package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
)

// cacheBody builds a /v1/messages body whose prompt is 20 characters, with a
// cache_control breakpoint at the halfway mark: 10 system chars, then
// len("user") + 6 message chars.
func cacheBody(t *testing.T, stream bool, ttl string) string {
	t.Helper()
	marker := `{"type":"ephemeral"}`
	if ttl != "" {
		marker = `{"type":"ephemeral","ttl":"` + ttl + `"}`
	}
	streamField := "false"
	if stream {
		streamField = "true"
	}
	return `{"model":"test-model","max_tokens":100,"stream":` + streamField + `,
		"system":[{"type":"text","text":"AAAAAAAAAA","cache_control":` + marker + `}],
		"messages":[{"role":"user","content":[{"type":"text","text":"BBBBBB"}]}]}`
}

// respondOnce writes a single non-streaming ChatResponse with the given metrics.
func respondOnce(metrics api.Metrics) gin.HandlerFunc {
	return func(c *gin.Context) {
		data, _ := json.Marshal(api.ChatResponse{
			Model:      "test-model",
			Message:    api.Message{Role: "assistant", Content: "Hello there!"},
			Done:       true,
			DoneReason: "stop",
			Metrics:    metrics,
		})
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	}
}

func TestAnthropicNonStreamingReportsCacheCreation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", respondOnce(api.Metrics{
		PromptEvalCount:       10,
		PromptEvalCachedCount: testIntPtr(2),
		EvalCount:             5,
	}))

	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(cacheBody(t, false, "")))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	// Breakpoint at token 5, 2 tokens already cached: 3 written, 5 in the tail.
	if got := optionalIntValue(result.Usage.CacheCreationInputTokens); got != 3 {
		t.Fatalf("cache_creation_input_tokens = %d, want 3", got)
	}
	if got := optionalIntValue(result.Usage.CacheReadInputTokens); got != 2 {
		t.Fatalf("cache_read_input_tokens = %d, want 2", got)
	}
	if result.Usage.InputTokens != 5 {
		t.Fatalf("input_tokens = %d, want 5", result.Usage.InputTokens)
	}
	if result.Usage.CacheCreation == nil || result.Usage.CacheCreation.Ephemeral5mInputTokens != 3 {
		t.Fatalf("cache_creation breakdown = %+v, want 3 under the 5m TTL", result.Usage.CacheCreation)
	}
	if sum := promptTokens(result.Usage); sum != 10 {
		t.Fatalf("usage sums to %d, want the 10 token prompt", sum)
	}
}

// Requests with no cache_control must keep reporting exactly what they did
// before: no cache_creation fields at all.
func TestAnthropicWithoutCacheControlOmitsCacheCreation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", respondOnce(api.Metrics{PromptEvalCount: 10, EvalCount: 5}))

	body := `{"model":"test-model","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if strings.Contains(resp.Body.String(), "cache_creation") {
		t.Fatalf("response reports cache creation for an uncached request: %s", resp.Body.String())
	}
	if strings.Contains(resp.Body.String(), "cache_read_input_tokens") {
		t.Fatalf("response reports cache reads the runner never mentioned: %s", resp.Body.String())
	}
}

// Metrics on a non-final chunk. Only reachable via IncludeIntermediateMetrics;
// see TestAnthropicStreamingProductionChunkShape for the ordinary sequence.
func TestAnthropicStreamingReportsCacheCreation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Hello"},
				Metrics: api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(2)},
			},
			{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: " there!"},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(2), EvalCount: 5},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(cacheBody(t, true, "")))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", resp.Code, resp.Body.String())
	}

	events := parseSSEEvents(t, resp.Body.String())

	var start anthropic.MessageStartEvent
	var delta anthropic.MessageDeltaEvent
	for _, event := range events {
		switch event.event {
		case "message_start":
			if err := json.Unmarshal([]byte(event.data), &start); err != nil {
				t.Fatal(err)
			}
		case "message_delta":
			if err := json.Unmarshal([]byte(event.data), &delta); err != nil {
				t.Fatal(err)
			}
		}
	}

	if got := optionalIntValue(start.Message.Usage.CacheCreationInputTokens); got != 3 {
		t.Fatalf("message_start cache_creation_input_tokens = %d, want 3", got)
	}
	if got := optionalIntValue(delta.Usage.CacheCreationInputTokens); got != 3 {
		t.Fatalf("message_delta cache_creation_input_tokens = %d, want 3", got)
	}
	if got := optionalIntValue(delta.Usage.CacheReadInputTokens); got != 2 {
		t.Fatalf("message_delta cache_read_input_tokens = %d, want 2", got)
	}
	if delta.Usage.InputTokens != 5 {
		t.Fatalf("message_delta input_tokens = %d, want 5", delta.Usage.InputTokens)
	}
	if delta.Usage.CacheCreation == nil || delta.Usage.CacheCreation.Ephemeral5mInputTokens != 3 {
		t.Fatalf("message_delta cache_creation breakdown = %+v, want 3 under the 5m TTL", delta.Usage.CacheCreation)
	}
	sum := delta.Usage.InputTokens +
		optionalIntValue(delta.Usage.CacheReadInputTokens) +
		optionalIntValue(delta.Usage.CacheCreationInputTokens)
	if sum != 10 {
		t.Fatalf("message_delta usage sums to %d, want the 10 token prompt", sum)
	}
}

// The sequence a real client actually receives: server/routes.go zeroes the
// metrics on every non-final chunk, so message_start fires before any token
// count exists and must claim nothing about the cache. The real numbers arrive
// only in message_delta -- which is the event Anthropic SDK clients treat as
// authoritative for the final usage snapshot.
func TestAnthropicStreamingProductionChunkShape(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Hello"},
				Metrics: api.Metrics{}, // non-final chunks carry no prompt metrics
			},
			{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: " there!"},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, PromptEvalCachedCount: testIntPtr(8), EvalCount: 5},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(cacheBody(t, true, "")))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	var start anthropic.MessageStartEvent
	var delta anthropic.MessageDeltaEvent
	for _, event := range parseSSEEvents(t, resp.Body.String()) {
		switch event.event {
		case "message_start":
			if err := json.Unmarshal([]byte(event.data), &start); err != nil {
				t.Fatal(err)
			}
		case "message_delta":
			if err := json.Unmarshal([]byte(event.data), &delta); err != nil {
				t.Fatal(err)
			}
		}
	}

	// message_start knows nothing yet: it must not invent a split that SDK
	// clients would keep, since their message_delta schema drops cache_creation.
	if start.Message.Usage.CacheCreationInputTokens != nil ||
		start.Message.Usage.CacheCreation != nil ||
		start.Message.Usage.CacheReadInputTokens != nil {
		t.Fatalf("message_start claims cache activity before any metrics exist: %+v", start.Message.Usage)
	}
	if start.Message.Usage.InputTokens <= 0 {
		t.Fatalf("message_start input_tokens = %d, want the request-side estimate", start.Message.Usage.InputTokens)
	}

	// message_delta carries the truth: a warm 80% hit, nothing newly written.
	if got := optionalIntValue(delta.Usage.CacheReadInputTokens); got != 8 {
		t.Fatalf("message_delta cache_read = %d, want 8", got)
	}
	if got := optionalIntValue(delta.Usage.CacheCreationInputTokens); got != 0 {
		t.Fatalf("message_delta cache_creation = %d, want 0", got)
	}
	if delta.Usage.InputTokens != 2 {
		t.Fatalf("message_delta input_tokens = %d, want 2", delta.Usage.InputTokens)
	}
	sum := delta.Usage.InputTokens +
		optionalIntValue(delta.Usage.CacheReadInputTokens) +
		optionalIntValue(delta.Usage.CacheCreationInputTokens)
	if sum != 10 {
		t.Fatalf("message_delta usage sums to %d, want the 10 token prompt", sum)
	}
}

func TestAnthropicRejectsInvalidCacheControl(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) { c.Status(http.StatusOK) })

	body := `{"model":"test-model","max_tokens":100,
		"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","ttl":"7d"}}]}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", resp.Code, resp.Body.String())
	}
	var errResp anthropic.ErrorResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
		t.Fatal(err)
	}
	if errResp.Error.Type != "invalid_request_error" {
		t.Fatalf("error type = %q, want invalid_request_error", errResp.Error.Type)
	}
	if !strings.Contains(errResp.Error.Message, "cache_control.ttl") {
		t.Fatalf("error message = %q, want it to name cache_control.ttl", errResp.Error.Message)
	}
}

// A malformed marker must be rejected in every position. `system` decodes to
// []any rather than a typed struct, so it bypasses gin's JSON binding and has
// to be validated during the cache scan.
func TestAnthropicRejectsMalformedCacheControlEverywhere(t *testing.T) {
	gin.SetMode(gin.TestMode)

	bodies := map[string]string{
		"system": `{"model":"test-model","max_tokens":100,
			"system":[{"type":"text","text":"AAAA","cache_control":{"type":"ephemeral","ttl":3600}}],
			"messages":[{"role":"user","content":"hi"}]}`,
		"system marker not an object": `{"model":"test-model","max_tokens":100,
			"system":[{"type":"text","text":"AAAA","cache_control":"ephemeral"}],
			"messages":[{"role":"user","content":"hi"}]}`,
		"message block": `{"model":"test-model","max_tokens":100,
			"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","ttl":3600}}]}]}`,
		"tool": `{"model":"test-model","max_tokens":100,
			"tools":[{"name":"t","description":"d","input_schema":{},"cache_control":{"type":"ephemeral","ttl":3600}}],
			"messages":[{"role":"user","content":"hi"}]}`,
	}

	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			router := gin.New()
			router.Use(AnthropicMessagesMiddleware())
			router.POST("/v1/messages", respondOnce(api.Metrics{PromptEvalCount: 10, EvalCount: 1}))

			req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", resp.Code, resp.Body.String())
			}
		})
	}
}

// A 1h cache TTL must reach the runner as a keep_alive long enough to hold the
// KV cache that backs it.
func TestAnthropicExtendedTTLSetsKeepAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_KEEP_ALIVE", "")

	tests := []struct {
		name string
		ttl  string
		want *api.Duration
	}{
		{name: "1h ttl", ttl: "1h", want: &api.Duration{Duration: time.Hour}},
		{name: "default ttl", ttl: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var captured api.ChatRequest
			router := gin.New()
			router.Use(AnthropicMessagesMiddleware())
			router.POST("/v1/messages", func(c *gin.Context) {
				if err := json.NewDecoder(c.Request.Body).Decode(&captured); err != nil {
					t.Fatalf("decode forwarded request: %v", err)
				}
				respondOnce(api.Metrics{PromptEvalCount: 10, EvalCount: 1})(c)
			})

			req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(cacheBody(t, false, tt.ttl)))
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(httptest.NewRecorder(), req)

			switch {
			case tt.want == nil && captured.KeepAlive != nil:
				t.Fatalf("keep_alive = %v, want it left to the server default", captured.KeepAlive)
			case tt.want != nil && (captured.KeepAlive == nil || captured.KeepAlive.Duration != tt.want.Duration):
				t.Fatalf("keep_alive = %v, want %v", captured.KeepAlive, tt.want)
			}
		})
	}
}
