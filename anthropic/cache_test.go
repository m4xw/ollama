package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

// cacheRequest builds a request whose prompt characters are exactly
// systemChars + len("user") + sum(blockChars), so breakpoint positions are
// predictable. Each entry in blocks is (chars, ttl); an empty ttl means the
// block carries no cache_control marker.
type block struct {
	chars int
	ttl   string // "" = no marker, "5m"/"1h" = marker with that TTL
}

func cacheRequest(t *testing.T, systemChars int, systemTTL string, blocks ...block) string {
	t.Helper()
	marker := func(ttl string) string {
		switch ttl {
		case "":
			return ""
		case CacheTTL5m:
			return `,"cache_control":{"type":"ephemeral"}`
		default:
			return `,"cache_control":{"type":"ephemeral","ttl":"` + ttl + `"}`
		}
	}

	var content []string
	for _, b := range blocks {
		content = append(content, `{"type":"text","text":"`+strings.Repeat("B", b.chars)+`"`+marker(b.ttl)+`}`)
	}
	return `{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"` + strings.Repeat("A", systemChars) + `"` + marker(systemTTL) + `}],` +
		`"messages":[{"role":"user","content":[` + strings.Join(content, ",") + `]}]}`
}

func mustScan(t *testing.T, body string) *PromptCacheLayout {
	t.Helper()
	var req MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	layout, err := ScanPromptCache(req)
	if err != nil {
		t.Fatalf("ScanPromptCache: %v", err)
	}
	return layout
}

func TestCacheControlRoundTrip(t *testing.T) {
	body := `{
	  "model": "m", "max_tokens": 16,
	  "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
	  "tools": [{"name": "t", "description": "d", "input_schema": {}, "cache_control": {"type": "ephemeral"}}],
	  "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]
	}`

	var req MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}

	if req.Tools[0].CacheControl == nil || req.Tools[0].CacheControl.Type != "ephemeral" {
		t.Fatalf("tool cache_control = %+v, want ephemeral", req.Tools[0].CacheControl)
	}
	if got := req.Messages[0].Content[0].CacheControl; got == nil || got.TTL != "" {
		t.Fatalf("message cache_control = %+v, want ephemeral with default ttl", got)
	}

	layout, err := ScanPromptCache(req)
	if err != nil {
		t.Fatal(err)
	}
	if got := layout.Breakpoints(); got != 3 {
		t.Fatalf("breakpoints = %d, want 3", got)
	}
	if !layout.ExtendedTTL() {
		t.Fatal("ExtendedTTL = false, want true for a 1h breakpoint")
	}
}

// Generated content blocks must never carry cache_control back to the client.
func TestCacheControlNotEmittedOnResponses(t *testing.T) {
	resp := ToMessagesResponse("msg_1", api.ChatResponse{
		Model:      "m",
		Message:    api.Message{Role: "assistant", Content: "hello"},
		Done:       true,
		DoneReason: "stop",
	})
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "cache_control") {
		t.Fatalf("response carries cache_control: %s", data)
	}
}

func TestScanPromptCacheRejectsInvalidMarkers(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "unknown type on system block",
			body: `{"model":"m","max_tokens":1,"system":[{"type":"text","text":"s","cache_control":{"type":"persistent"}}],"messages":[]}`,
			want: "cache_control.type",
		},
		{
			name: "unknown ttl on message block",
			body: `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"7d"}}]}]}`,
			want: "cache_control.ttl",
		},
		{
			name: "unknown type on tool",
			body: `{"model":"m","max_tokens":1,"tools":[{"name":"t","description":"d","input_schema":{},"cache_control":{"type":""}}],"messages":[]}`,
			want: "cache_control.type",
		},
		{
			name: "unknown ttl on top-level automatic caching",
			body: `{"model":"m","max_tokens":1,"cache_control":{"type":"ephemeral","ttl":"30m"},"messages":[{"role":"user","content":"x"}]}`,
			want: "cache_control.ttl",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req MessagesRequest
			if err := json.Unmarshal([]byte(tt.body), &req); err != nil {
				t.Fatal(err)
			}
			_, err := ScanPromptCache(req)
			if err == nil {
				t.Fatal("ScanPromptCache succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

func TestScanPromptCacheNilWithoutBreakpoints(t *testing.T) {
	layout := mustScan(t, `{"model":"m","max_tokens":1,"system":"plain","messages":[{"role":"user","content":"hi"}]}`)
	if layout != nil {
		t.Fatalf("layout = %+v, want nil when no cache_control is present", layout)
	}
	// Nil layouts must behave exactly like the pre-existing conversion.
	metrics := api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(4), EvalCount: 2}
	if diff := cmp.Diff(UsageFromMetrics(metrics), layout.UsageFromMetrics(metrics)); diff != "" {
		t.Fatalf("nil layout diverges from UsageFromMetrics (-want +got):\n%s", diff)
	}
}

// The invariant every usage report must satisfy:
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens == prompt tokens,
// and the per-TTL breakdown always sums to cache_creation_input_tokens.
func TestUsageSumInvariant(t *testing.T) {
	layouts := map[string]*PromptCacheLayout{
		"nil": nil,
		"single 5m breakpoint at 50%": mustScan(t, `{"model":"m","max_tokens":1,
			"system":[{"type":"text","text":"AAAAAAAAAA","cache_control":{"type":"ephemeral"}}],
			"messages":[{"role":"user","content":[{"type":"text","text":"BBBBBBBBBBBBBB"}]}]}`),
		"two breakpoints mixed ttl": mustScan(t, `{"model":"m","max_tokens":1,
			"system":[{"type":"text","text":"AAAAAAAAAA","cache_control":{"type":"ephemeral","ttl":"1h"}}],
			"messages":[{"role":"user","content":[
				{"type":"text","text":"BBBBBBBBBB","cache_control":{"type":"ephemeral"}},
				{"type":"text","text":"CCCCCCCCCC"}]}]}`),
		"automatic caching": mustScan(t, `{"model":"m","max_tokens":1,"cache_control":{"type":"ephemeral"},
			"messages":[{"role":"user","content":[{"type":"text","text":"AAAAAAAAAAAAAAAAAAAA"}]}]}`),
	}

	totals := []int{0, 1, 7, 100, 4096, 131072}
	for name, layout := range layouts {
		for _, total := range totals {
			for _, cached := range []int{0, 1, total / 3, total / 2, max(0, total-1), total, total + 5} {
				metrics := api.Metrics{PromptEvalCount: total, PromptEvalCachedCount: &cached, EvalCount: 3}
				usage := layout.UsageFromMetrics(metrics)

				if usage.InputTokens < 0 {
					t.Fatalf("%s total=%d cached=%d: input_tokens = %d, want >= 0", name, total, cached, usage.InputTokens)
				}
				if intValue(usage.CacheCreationInputTokens) < 0 {
					t.Fatalf("%s total=%d cached=%d: cache_creation = %d, want >= 0", name, total, cached, intValue(usage.CacheCreationInputTokens))
				}
				sum := usage.InputTokens + intValue(usage.CacheReadInputTokens) + intValue(usage.CacheCreationInputTokens)
				if sum != total {
					t.Fatalf("%s total=%d cached=%d: input(%d)+read(%d)+creation(%d) = %d, want %d",
						name, total, cached, usage.InputTokens, intValue(usage.CacheReadInputTokens),
						intValue(usage.CacheCreationInputTokens), sum, total)
				}
				if usage.CacheCreation != nil {
					byTTL := usage.CacheCreation.Ephemeral5mInputTokens + usage.CacheCreation.Ephemeral1hInputTokens
					if byTTL != intValue(usage.CacheCreationInputTokens) {
						t.Fatalf("%s total=%d cached=%d: cache_creation breakdown = %d, want %d",
							name, total, cached, byTTL, intValue(usage.CacheCreationInputTokens))
					}
				}
			}
		}
	}
}

func TestUsageColdAndWarmRequests(t *testing.T) {
	// 100 system chars + len("user") + 96 message chars = 200 chars, so the
	// breakpoint sits at exactly half the prompt: token 100 of 200.
	layout := mustScan(t, cacheRequest(t, 100, CacheTTL5m, block{chars: 96}))

	t.Run("cold", func(t *testing.T) {
		usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 200, PromptEvalCachedCount: testIntPtr(0), EvalCount: 5})
		if got := intValue(usage.CacheCreationInputTokens); got != 100 {
			t.Fatalf("cache_creation = %d, want 100 (the marked prefix)", got)
		}
		if usage.InputTokens != 100 {
			t.Fatalf("input_tokens = %d, want 100 (the tail past the breakpoint)", usage.InputTokens)
		}
		if usage.CacheCreation.Ephemeral5mInputTokens != 100 || usage.CacheCreation.Ephemeral1hInputTokens != 0 {
			t.Fatalf("cache_creation breakdown = %+v, want all 5m", usage.CacheCreation)
		}
	})

	t.Run("warm", func(t *testing.T) {
		// The runner reused the whole prefix and past it.
		usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 200, PromptEvalCachedCount: testIntPtr(199), EvalCount: 5})
		if got := intValue(usage.CacheCreationInputTokens); got != 0 {
			t.Fatalf("cache_creation = %d, want 0 when the prefix was a cache hit", got)
		}
		if got := intValue(usage.CacheReadInputTokens); got != 199 {
			t.Fatalf("cache_read = %d, want 199", got)
		}
		if usage.InputTokens != 1 {
			t.Fatalf("input_tokens = %d, want 1", usage.InputTokens)
		}
	})

	t.Run("partial hit inside the marked prefix", func(t *testing.T) {
		usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 200, PromptEvalCachedCount: testIntPtr(40), EvalCount: 5})
		if got := intValue(usage.CacheCreationInputTokens); got != 60 {
			t.Fatalf("cache_creation = %d, want 60 (prefix 100 minus the 40 already cached)", got)
		}
		if usage.InputTokens != 100 {
			t.Fatalf("input_tokens = %d, want 100", usage.InputTokens)
		}
	})
}

func TestUsagePerTTLAttribution(t *testing.T) {
	// 100 (1h) + len("user") + 96 (5m) + 100 (unmarked) = 300 chars, putting the
	// breakpoints at tokens 100 and 200 of a 300 token prompt.
	layout := mustScan(t, cacheRequest(t, 100, CacheTTL1h,
		block{chars: 96, ttl: CacheTTL5m},
		block{chars: 100}))

	usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 300, PromptEvalCachedCount: testIntPtr(0), EvalCount: 1})
	want := CacheCreation{Ephemeral5mInputTokens: 100, Ephemeral1hInputTokens: 100}
	if diff := cmp.Diff(&want, usage.CacheCreation); diff != "" {
		t.Fatalf("cache_creation (-want +got):\n%s", diff)
	}
	if got := intValue(usage.CacheCreationInputTokens); got != 200 {
		t.Fatalf("cache_creation_input_tokens = %d, want 200", got)
	}
	if usage.InputTokens != 100 {
		t.Fatalf("input_tokens = %d, want 100", usage.InputTokens)
	}

	// A cache hit that covers the 1h segment leaves only the 5m segment to write.
	usage = layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 300, PromptEvalCachedCount: testIntPtr(100), EvalCount: 1})
	want = CacheCreation{Ephemeral5mInputTokens: 100}
	if diff := cmp.Diff(&want, usage.CacheCreation); diff != "" {
		t.Fatalf("cache_creation after partial hit (-want +got):\n%s", diff)
	}
}

// Usage must serialize with Anthropic's field names, and omit the cache fields
// entirely for requests that asked for no caching.
func TestUsageCacheCreationJSON(t *testing.T) {
	tests := []struct {
		name   string
		layout *PromptCacheLayout
		want   string
	}{
		{
			name:   "no caching requested",
			layout: nil,
			want:   `{"input_tokens":6,"cache_read_input_tokens":4,"output_tokens":2}`,
		},
		{
			name: "caching requested",
			// 10 + len("user") + 6 = 20 chars: the breakpoint is token 5 of 10.
			layout: mustScan(t, cacheRequest(t, 10, CacheTTL5m, block{chars: 6})),
			want:   `{"input_tokens":5,"cache_creation_input_tokens":1,"cache_read_input_tokens":4,"cache_creation":{"ephemeral_5m_input_tokens":1,"ephemeral_1h_input_tokens":0},"output_tokens":2}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := tt.layout.UsageFromMetrics(api.Metrics{
				PromptEvalCount:       10,
				PromptEvalCachedCount: testIntPtr(4),
				EvalCount:             2,
			})
			data, err := json.Marshal(usage)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != tt.want {
				t.Fatalf("usage JSON = %s, want %s", data, tt.want)
			}
		})
	}
}

// A request that asked for caching always reports both counters, so clients can
// tell a cold cache apart from a server that does not report cache activity.
func TestUsageReportsZeroCountersWhenCachingRequested(t *testing.T) {
	layout := mustScan(t, `{"model":"m","max_tokens":1,
		"system":[{"type":"text","text":"AAAAA","cache_control":{"type":"ephemeral"}}],
		"messages":[{"role":"user","content":"BBBBB"}]}`)

	usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 10, EvalCount: 1})
	if usage.CacheReadInputTokens == nil || *usage.CacheReadInputTokens != 0 {
		t.Fatalf("cache_read_input_tokens = %v, want 0", usage.CacheReadInputTokens)
	}
	if usage.CacheCreationInputTokens == nil {
		t.Fatal("cache_creation_input_tokens = nil, want a reported count")
	}
}

// Follow-up runner calls grow the prompt; without a frozen anchor the scaled
// breakpoint offsets would drift forward and inflate cache_creation.
func TestUsageAnchorKeepsBreakpointStable(t *testing.T) {
	layout := mustScan(t, cacheRequest(t, 100, CacheTTL5m, block{chars: 96}))

	base := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 200, PromptEvalCachedCount: testIntPtr(0)})
	grown := layout.UsageFromMetricsAnchored(api.Metrics{PromptEvalCount: 600, PromptEvalCachedCount: testIntPtr(0)}, 200)

	if intValue(grown.CacheCreationInputTokens) != intValue(base.CacheCreationInputTokens) {
		t.Fatalf("anchored cache_creation = %d, want it pinned at %d",
			intValue(grown.CacheCreationInputTokens), intValue(base.CacheCreationInputTokens))
	}
	if grown.InputTokens != 500 {
		t.Fatalf("input_tokens = %d, want 500 (all prompt growth is plain input)", grown.InputTokens)
	}
}

func TestApplyKeepAliveExtendsForOneHourTTL(t *testing.T) {
	oneHour := mustScan(t, cacheRequest(t, 10, CacheTTL1h, block{chars: 6}))
	fiveMin := mustScan(t, cacheRequest(t, 10, CacheTTL5m, block{chars: 6}))

	tests := []struct {
		name      string
		layout    *PromptCacheLayout
		keepAlive string // OLLAMA_KEEP_ALIVE; "" means the built-in 5m default
		existing  *api.Duration
		want      *api.Duration
	}{
		{name: "1h ttl on default keep-alive", layout: oneHour, want: &api.Duration{Duration: time.Hour}},
		{name: "5m ttl left alone", layout: fiveMin},
		{name: "no caching left alone", layout: nil},
		{
			name:      "server already keeps models longer",
			layout:    oneHour,
			keepAlive: "24h",
		},
		{
			name:      "server keeps models forever",
			layout:    oneHour,
			keepAlive: "-1",
		},
		{
			name:     "explicit request keep-alive wins",
			layout:   oneHour,
			existing: &api.Duration{Duration: 90 * time.Second},
			want:     &api.Duration{Duration: 90 * time.Second},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Pin the variable: the ambient value on a developer machine would
			// otherwise decide the outcome.
			t.Setenv("OLLAMA_KEEP_ALIVE", tt.keepAlive)
			req := &api.ChatRequest{KeepAlive: tt.existing}
			applyKeepAlive(req, tt.layout)
			if diff := cmp.Diff(tt.want, req.KeepAlive); diff != "" {
				t.Fatalf("keep_alive (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFromMessagesRequestWithCache(t *testing.T) {
	var req MessagesRequest
	body := `{"model":"m","max_tokens":8,
		"system":[{"type":"text","text":"policy","cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"messages":[{"role":"user","content":"hello"}]}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_KEEP_ALIVE", "")

	chatReq, layout, err := FromMessagesRequestWithCache(req)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Breakpoints() != 1 {
		t.Fatalf("breakpoints = %d, want 1", layout.Breakpoints())
	}
	if chatReq.KeepAlive == nil || chatReq.KeepAlive.Duration != time.Hour {
		t.Fatalf("keep_alive = %v, want 1h for a 1h cache TTL", chatReq.KeepAlive)
	}
	// The system block still renders into the prompt unchanged.
	if len(chatReq.Messages) == 0 || chatReq.Messages[0].Content != "policy" {
		t.Fatalf("messages = %+v, want the system block preserved", chatReq.Messages)
	}
}

// More than the upstream limit of four breakpoints is tolerated, not rejected.
func TestScanPromptCacheToleratesManyBreakpoints(t *testing.T) {
	var blocks []string
	for range maxCacheBreakpoints + 3 {
		blocks = append(blocks, `{"type":"text","text":"chunk","cache_control":{"type":"ephemeral"}}`)
	}
	body := `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[` + strings.Join(blocks, ",") + `]}]}`

	layout := mustScan(t, body)
	if got := layout.Breakpoints(); got != maxCacheBreakpoints+3 {
		t.Fatalf("breakpoints = %d, want all %d honored", got, maxCacheBreakpoints+3)
	}
	if _, err := FromMessagesRequest(mustUnmarshalRequest(t, body)); err != nil {
		t.Fatalf("FromMessagesRequest rejected %d breakpoints: %v", maxCacheBreakpoints+3, err)
	}
}

func mustUnmarshalRequest(t *testing.T, body string) MessagesRequest {
	t.Helper()
	var req MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func TestStreamConverterReportsCacheCreation(t *testing.T) {
	// 10 system chars + len("user") + 6 message chars = 20 chars, so the
	// breakpoint is token 5 of a 10 token prompt.
	layout := mustScan(t, cacheRequest(t, 10, CacheTTL5m, block{chars: 6}))

	conv := NewStreamConverter("msg_123", "test-model", 0)
	conv.PromptCache = layout

	first := conv.Process(api.ChatResponse{
		Model:   "test-model",
		Message: api.Message{Role: "assistant", Content: "Hi"},
		Metrics: api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(2)},
	})
	start, ok := first[0].Data.(MessageStartEvent)
	if !ok {
		t.Fatalf("first event = %T, want MessageStartEvent", first[0].Data)
	}
	if got := intValue(start.Message.Usage.CacheCreationInputTokens); got != 3 {
		t.Fatalf("message_start cache_creation = %d, want 3 (breakpoint 5 minus 2 cached)", got)
	}
	if start.Message.Usage.InputTokens != 5 {
		t.Fatalf("message_start input_tokens = %d, want 5", start.Message.Usage.InputTokens)
	}

	final := conv.Process(api.ChatResponse{
		Model:      "test-model",
		Message:    api.Message{Role: "assistant", Content: " there"},
		Done:       true,
		DoneReason: "stop",
		Metrics:    api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(9), EvalCount: 4},
	})

	var delta MessageDeltaEvent
	for _, e := range final {
		if d, ok := e.Data.(MessageDeltaEvent); ok {
			delta = d
		}
	}
	// The real counts land in message_delta, replacing whatever message_start
	// reported before generation finished.
	if got := intValue(delta.Usage.CacheCreationInputTokens); got != 0 {
		t.Fatalf("message_delta cache_creation = %d, want 0 (the prefix was a hit)", got)
	}
	if got := intValue(delta.Usage.CacheReadInputTokens); got != 9 {
		t.Fatalf("message_delta cache_read = %d, want 9", got)
	}
	if delta.Usage.InputTokens != 1 {
		t.Fatalf("message_delta input_tokens = %d, want 1", delta.Usage.InputTokens)
	}
	if delta.Usage.CacheCreation == nil {
		t.Fatal("message_delta cache_creation breakdown missing")
	}
	sum := delta.Usage.InputTokens + intValue(delta.Usage.CacheReadInputTokens) + intValue(delta.Usage.CacheCreationInputTokens)
	if sum != 10 {
		t.Fatalf("message_delta usage sums to %d, want the 10 token prompt", sum)
	}
}

// With no metrics yet, message_start falls back to the request-side estimate.
// It must NOT invent a cache split: the Anthropic SDKs' message_delta usage has
// no `cache_creation` field, so a breakdown guessed here would survive as the
// client's final snapshot even after message_delta reports the real numbers.
func TestStreamConverterEstimateClaimsNoCacheSplit(t *testing.T) {
	layout := mustScan(t, cacheRequest(t, 10, CacheTTL5m, block{chars: 6}))

	conv := NewStreamConverter("msg_123", "test-model", 40)
	conv.PromptCache = layout

	events := conv.Process(api.ChatResponse{
		Model:   "test-model",
		Message: api.Message{Role: "assistant", Content: "Hi"},
	})
	usage := events[0].Data.(MessageStartEvent).Message.Usage
	if usage.InputTokens != 40 {
		t.Fatalf("estimated input_tokens = %d, want the full 40 token estimate", usage.InputTokens)
	}
	if usage.CacheCreationInputTokens != nil || usage.CacheCreation != nil || usage.CacheReadInputTokens != nil {
		t.Fatalf("estimate claims cache activity it cannot know: %+v", usage)
	}

	// The real numbers still arrive in message_delta.
	final := conv.Process(api.ChatResponse{
		Model:      "test-model",
		Message:    api.Message{Role: "assistant", Content: "!"},
		Done:       true,
		DoneReason: "stop",
		Metrics:    api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(8), EvalCount: 2},
	})
	var delta MessageDeltaEvent
	for _, e := range final {
		if d, ok := e.Data.(MessageDeltaEvent); ok {
			delta = d
		}
	}
	if got := intValue(delta.Usage.CacheReadInputTokens); got != 8 {
		t.Fatalf("message_delta cache_read = %d, want 8", got)
	}
	if got := intValue(delta.Usage.CacheCreationInputTokens); got != 0 {
		t.Fatalf("message_delta cache_creation = %d, want 0", got)
	}
}

// Regression: a malformed cache_control on a system block must be rejected just
// like the same marker on a tool or a message block, and a block that does not
// round-trip through ContentBlock must still contribute its characters.
func TestScanPromptCacheValidatesRawSystemMarkers(t *testing.T) {
	t.Run("malformed marker is rejected", func(t *testing.T) {
		for name, marker := range map[string]string{
			"ttl as number":    `{"type":"ephemeral","ttl":3600}`,
			"marker as string": `"ephemeral"`,
		} {
			t.Run(name, func(t *testing.T) {
				body := `{"model":"m","max_tokens":1,
					"system":[{"type":"text","text":"AAAAAAAAAA","cache_control":` + marker + `}],
					"messages":[{"role":"user","content":"BBBBBB"}]}`
				var req MessagesRequest
				if err := json.Unmarshal([]byte(body), &req); err != nil {
					t.Fatal(err)
				}
				if _, err := ScanPromptCache(req); err == nil {
					t.Fatal("ScanPromptCache accepted a malformed system cache_control")
				}
			})
		}
	})

	t.Run("odd block still counts toward the prefix", func(t *testing.T) {
		// The converter renders this block's text regardless of the field it
		// cannot decode, so dropping it would rescale the other breakpoint.
		body := `{"model":"m","max_tokens":1,
			"system":[{"type":"text","text":"` + strings.Repeat("A", 100) + `","citations":"nope"}],
			"messages":[{"role":"user","content":[
				{"type":"text","text":"` + strings.Repeat("B", 96) + `","cache_control":{"type":"ephemeral"}},
				{"type":"text","text":"` + strings.Repeat("C", 100) + `"}]}]}`
		layout := mustScan(t, body)
		usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 300, PromptEvalCachedCount: testIntPtr(0)})
		if got := intValue(usage.CacheCreationInputTokens); got != 200 {
			t.Fatalf("cache_creation = %d, want 200 (the odd system block must still count)", got)
		}
	})
}

// Regression: block types that render into the prompt must not score 0
// characters, or every breakpoint after them lands in the wrong place.
func TestBlockPromptCharsCoversRenderedBlocks(t *testing.T) {
	long := strings.Repeat("R", 300)
	tests := []struct {
		name  string
		block ContentBlock
		want  int
	}{
		{name: "text", block: ContentBlock{Type: "text", Text: ptr("hello")}, want: 5},
		{name: "thinking", block: ContentBlock{Type: "thinking", Thinking: ptr("hmm")}, want: 3},
		{
			name:  "tool_result string content",
			block: ContentBlock{Type: "tool_result", ToolUseID: "id", Content: long},
			want:  2 + 300,
		},
		{
			name: "web_search_tool_result",
			block: ContentBlock{Type: "web_search_tool_result", ToolUseID: "id", Content: []any{
				map[string]any{"type": "web_search_result", "title": long, "url": "u"},
			}},
		},
		{name: "image contributes no characters", block: ContentBlock{Type: "image", Source: &ImageSource{Type: "base64", Data: long}}, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := blockPromptChars(tt.block)
			switch tt.name {
			case "web_search_tool_result":
				if got < 300 {
					t.Fatalf("chars = %d, want the rendered result text (>=300) counted", got)
				}
			default:
				if got != tt.want {
					t.Fatalf("chars = %d, want %d", got, tt.want)
				}
			}
		})
	}
}

// Regression: a system value the converter drops must not inflate totalChars.
func TestScanPromptCacheIgnoresDroppedSystemShapes(t *testing.T) {
	body := `{"model":"m","max_tokens":1,
		"system":{"type":"text","text":"` + strings.Repeat("S", 400) + `"},
		"messages":[{"role":"user","content":[
			{"type":"text","text":"` + strings.Repeat("A", 16) + `","cache_control":{"type":"ephemeral"}},
			{"type":"text","text":"` + strings.Repeat("B", 20) + `"}]}]}`

	layout := mustScan(t, body)
	// Only the user turn reaches the model: 4 (role) + 16 = 20 of 40 chars.
	usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 100, PromptEvalCachedCount: testIntPtr(0)})
	if got := intValue(usage.CacheCreationInputTokens); got != 50 {
		t.Fatalf("cache_creation = %d, want 50 (the dropped system must not count)", got)
	}
}

// A layout with no characters cannot come out of ScanPromptCache, but the
// division in tokenOffsets must not panic if one is ever built another way.
func TestSplitWithNoCharactersDoesNotPanic(t *testing.T) {
	layout := &PromptCacheLayout{
		breakpoints: []cacheBreakpoint{{chars: 0, ttl: CacheTTL5m}},
		totalChars:  0,
	}
	usage := layout.UsageFromMetrics(api.Metrics{PromptEvalCount: 100, PromptEvalCachedCount: testIntPtr(10)})
	if usage.InputTokens != 90 {
		t.Fatalf("input_tokens = %d, want 90", usage.InputTokens)
	}
	if got := intValue(usage.CacheCreationInputTokens); got != 0 {
		t.Fatalf("cache_creation = %d, want 0", got)
	}
}

// Anthropic splits thinking across two fields: `thinking.type` is on/off and
// `output_config.effort` carries the depth. Clients send both, so the effort
// must survive a bare "enabled" -- otherwise every request collapses to plain
// `true` and the level is silently lost.
//
// Older clients (Claude Code) send neither: depth arrives as
// thinking.budget_tokens, and "off" arrives as no thinking block at all.
func TestThinkFromThinkingConfig(t *testing.T) {
	tests := []struct {
		name     string
		thinking *ThinkingConfig
		output   *OutputConfig
		want     any // nil means "leave at the model default"
	}{
		// No thinking block is how a budget_tokens client says "off"; leaving it
		// at the model default lets routes.go turn it back on.
		{name: "nothing requested", want: false},
		{name: "enabled only", thinking: &ThinkingConfig{Type: "enabled"}, want: true},
		{name: "adaptive only", thinking: &ThinkingConfig{Type: "adaptive"}, want: true},
		{name: "disabled only", thinking: &ThinkingConfig{Type: "disabled"}, want: false},
		{name: "effort only", output: &OutputConfig{Effort: "low"}, want: "low"},

		// The regression: effort alongside thinking.
		{
			name:     "effort wins over enabled",
			thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 8000},
			output:   &OutputConfig{Effort: "low"},
			want:     "low",
		},
		{
			name:     "effort wins over adaptive",
			thinking: &ThinkingConfig{Type: "adaptive"},
			output:   &OutputConfig{Effort: "max"},
			want:     "max",
		},
		// ...but an explicit disable still wins over everything.
		{
			name:     "disabled beats effort",
			thinking: &ThinkingConfig{Type: "disabled"},
			output:   &OutputConfig{Effort: "high"},
			want:     false,
		},

		{name: "effort is case and space insensitive", output: &OutputConfig{Effort: "  HIGH "}, want: "high"},
		{name: "xhigh clamps down to high", output: &OutputConfig{Effort: "xhigh"}, want: "high"},
		{name: "minimal clamps up to low", output: &OutputConfig{Effort: "minimal"}, want: "low"},
		{name: "none disables", output: &OutputConfig{Effort: "none"}, want: false},

		// An unrecognized effort must not discard a valid thinking.type.
		{
			name:     "unknown effort falls back to thinking.type",
			thinking: &ThinkingConfig{Type: "enabled"},
			output:   &OutputConfig{Effort: "turbo"},
			want:     true,
		},
		{name: "unknown effort alone disables", output: &OutputConfig{Effort: "turbo"}, want: false},

		// budget_tokens carries the level for clients that never send effort.
		// 8000 is Claude Code's lowest budget and 32000 is its high; both are
		// exact, so they must not sit on the wrong side of a bound.
		{name: "smallest budget is low", thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 8000}, want: "low"},
		{name: "below the smallest budget is still low", thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 4000}, want: "low"},
		{name: "mid budget is medium", thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 16000}, want: "medium"},
		{name: "high budget is high", thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 32000}, want: "high"},
		{name: "huge budget is max", thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 64000}, want: "max"},
		{name: "budget on adaptive counts too", thinking: &ThinkingConfig{Type: "adaptive", BudgetTokens: 32000}, want: "high"},
		{
			name:     "effort still wins over budget",
			thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 32000},
			output:   &OutputConfig{Effort: "low"},
			want:     "low",
		},
		{
			name:     "disabled beats budget",
			thinking: &ThinkingConfig{Type: "disabled", BudgetTokens: 32000},
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := thinkFromThinkingConfig(tt.thinking, tt.output, nil)
			if tt.want == nil {
				if got != nil {
					t.Fatalf("think = %v, want nil (model default)", got.Value)
				}
				return
			}
			if got == nil {
				t.Fatalf("think = nil, want %v", tt.want)
			}
			if got.Value != tt.want {
				t.Fatalf("think = %v, want %v", got.Value, tt.want)
			}
		})
	}
}

// End to end through the request conversion, in the shape Claude Code sends.
func TestFromMessagesRequestKeepsEffortAlongsideThinking(t *testing.T) {
	body := `{"model":"m","max_tokens":64,
		"thinking":{"type":"enabled","budget_tokens":8000},
		"output_config":{"effort":"low"},
		"messages":[{"role":"user","content":"hi"}]}`
	var req MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	chatReq, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if chatReq.Think == nil || chatReq.Think.Value != "low" {
		t.Fatalf("think = %v, want \"low\" (the effort must not be dropped)", chatReq.Think)
	}
}
