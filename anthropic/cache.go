package anthropic

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// Cache control types and TTL values accepted on `cache_control` markers.
const (
	CacheControlEphemeral = "ephemeral"

	CacheTTL5m = "5m"
	CacheTTL1h = "1h"
)

// maxCacheBreakpoints is the number of `cache_control` markers the upstream
// Anthropic API accepts per request. Ollama serves the whole prompt from a
// single llama.cpp KV slot, so extra breakpoints cost nothing and are honored
// rather than rejected; the limit is only used to log a compatibility warning.
const maxCacheBreakpoints = 100

// CacheControl marks the end of a cacheable prompt prefix.
type CacheControl struct {
	Type string `json:"type"`          // always "ephemeral"
	TTL  string `json:"ttl,omitempty"` // "5m" (default) or "1h"
}

// ttl returns the normalized TTL, defaulting to 5 minutes.
func (c *CacheControl) ttl() string {
	if c == nil || c.TTL == "" {
		return CacheTTL5m
	}
	return c.TTL
}

// validate rejects markers the Anthropic API would reject with a 400.
func (c *CacheControl) validate(where string) error {
	if c == nil {
		return nil
	}
	if c.Type != CacheControlEphemeral {
		return fmt.Errorf("cache_control.type %q is not supported (%s): expected %q", c.Type, where, CacheControlEphemeral)
	}
	switch c.TTL {
	case "", CacheTTL5m, CacheTTL1h:
		return nil
	default:
		return fmt.Errorf("cache_control.ttl %q is not supported (%s): expected %q or %q", c.TTL, where, CacheTTL5m, CacheTTL1h)
	}
}

// CacheCreation breaks cache_creation_input_tokens down by TTL. The two fields
// always sum to Usage.CacheCreationInputTokens.
type CacheCreation struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
}

// cacheBreakpoint is one `cache_control` marker, positioned by the cumulative
// character count of the rendered prompt prefix that ends at the marked block.
type cacheBreakpoint struct {
	chars int
	ttl   string
}

// PromptCacheLayout describes where a request's `cache_control` breakpoints sit
// within the prompt. Positions are recorded in characters because the exact
// token offsets are not known until the runner has tokenized the rendered
// prompt; the character ratios are then scaled by the real prompt token count
// reported in api.Metrics, which is far more accurate than estimating the
// prefix length on its own.
//
// A PromptCacheLayout is immutable once built and safe to share across
// goroutines.
type PromptCacheLayout struct {
	breakpoints []cacheBreakpoint // ascending by chars
	totalChars  int
	extendedTTL bool // at least one breakpoint asked for the 1h TTL
}

// Breakpoints reports how many `cache_control` markers the request carried.
func (l *PromptCacheLayout) Breakpoints() int {
	if l == nil {
		return 0
	}
	return len(l.breakpoints)
}

// ExtendedTTL reports whether any breakpoint requested the 1 hour TTL.
func (l *PromptCacheLayout) ExtendedTTL() bool {
	return l != nil && l.extendedTTL
}

// blockPromptChars reports how many prompt characters a content block
// contributes, mirroring what convertMessage actually renders into the Ollama
// request. It deliberately does NOT reuse countContentBlock, which exists to
// estimate a token count for /v1/messages/count_tokens and scores several block
// types that do reach the prompt as zero.
//
// Images are the one block type with no character analogue: they become
// api.ImageData and are tokenized by the vision encoder, so they contribute 0
// here and a breakpoint placed after a large image is positioned by the text
// around it. See docs/api/anthropic-compatibility.mdx.
func blockPromptChars(block ContentBlock) int {
	switch block.Type {
	case "text":
		if block.Text != nil {
			return len(*block.Text)
		}
	case "thinking":
		if block.Thinking != nil {
			return len(*block.Thinking)
		}
	case "tool_use", "server_tool_use":
		n := len(block.Name)
		if data, err := json.Marshal(block.Input); err == nil {
			n += len(data)
		}
		return n
	case "tool_result":
		return len(block.ToolUseID) + toolResultPromptChars(block.Content)
	case "web_search_tool_result":
		return len(block.ToolUseID) + len(formatWebSearchToolResultContent(block.Content))
	}
	return 0
}

// toolResultPromptChars mirrors the text convertToolResultContent produces,
// without decoding any images.
func toolResultPromptChars(content any) int {
	switch c := content.(type) {
	case nil:
		return 0
	case string:
		return len(c)
	case []any:
		total := 0
		for _, cb := range c {
			cbMap, ok := cb.(map[string]any)
			if !ok {
				continue
			}
			if cbMap["type"] == "text" {
				if t, ok := cbMap["text"].(string); ok {
					total += len(t)
				}
			}
		}
		return total
	default:
		if data, err := json.Marshal(content); err == nil {
			return len(data)
		}
		return 0
	}
}

// parseCacheControl reads a `cache_control` value out of a raw JSON-decoded
// map and validates it. A marker of the wrong shape is an error, matching what
// the typed tool and message paths produce when JSON binding fails.
func parseCacheControl(v any, where string) (*CacheControl, error) {
	if v == nil {
		return nil, nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("cache_control is not valid (%s): %w", where, err)
	}
	var cc CacheControl
	if err := json.Unmarshal(data, &cc); err != nil {
		return nil, fmt.Errorf("cache_control is not valid (%s): %w", where, err)
	}
	if err := cc.validate(where); err != nil {
		return nil, err
	}
	return &cc, nil
}

// cacheScanner accumulates prompt characters in Anthropic render order
// (tools, then system, then messages) and records breakpoint positions.
type cacheScanner struct {
	layout PromptCacheLayout
}

func (s *cacheScanner) add(chars int) {
	s.layout.totalChars += chars
}

func (s *cacheScanner) mark(cc *CacheControl) {
	ttl := cc.ttl()
	if ttl == CacheTTL1h {
		s.layout.extendedTTL = true
	}
	s.layout.breakpoints = append(s.layout.breakpoints, cacheBreakpoint{chars: s.layout.totalChars, ttl: ttl})
}

// ScanPromptCache validates every `cache_control` marker in the request and
// returns the resulting prompt cache layout. It returns a nil layout when the
// request carries no breakpoints, which leaves usage reporting unchanged.
func ScanPromptCache(r MessagesRequest) (*PromptCacheLayout, error) {
	var s cacheScanner

	// Render order is tools -> system -> messages.
	for i, t := range r.Tools {
		s.add(len(t.Name) + len(t.Description) + len(t.InputSchema))
		if err := t.CacheControl.validate(fmt.Sprintf("tools[%d]", i)); err != nil {
			return nil, err
		}
		if t.CacheControl != nil {
			s.mark(t.CacheControl)
		}
	}

	if err := scanSystemCache(&s, r.System); err != nil {
		return nil, err
	}

	for i, msg := range r.Messages {
		s.add(len(msg.Role))
		for j, block := range msg.Content {
			s.add(blockPromptChars(block))
			if err := block.CacheControl.validate(fmt.Sprintf("messages[%d].content[%d]", i, j)); err != nil {
				return nil, err
			}
			if block.CacheControl != nil {
				s.mark(block.CacheControl)
			}
		}
	}

	// A top-level `cache_control` requests automatic caching: the breakpoint
	// lands on the last cacheable block, i.e. at the very end of the prompt.
	if err := r.CacheControl.validate("cache_control"); err != nil {
		return nil, err
	}
	if r.CacheControl != nil {
		s.mark(r.CacheControl)
	}

	if len(s.layout.breakpoints) == 0 || s.layout.totalChars <= 0 {
		return nil, nil
	}

	layout := s.layout
	return &layout, nil
}

// scanSystemCache walks the system field. It mirrors FromMessagesRequest
// exactly: a string system prompt contributes its own characters, an array
// contributes the text of its `text` blocks, and any other shape is dropped by
// the converter and so contributes nothing.
func scanSystemCache(s *cacheScanner, system any) error {
	switch sys := system.(type) {
	case string:
		s.add(len(sys))
	case []ContentBlock:
		// Only reachable for Go-constructed requests; JSON always decodes to []any.
		for i, block := range sys {
			if block.Type == "text" {
				s.add(blockPromptChars(block))
			}
			if err := block.CacheControl.validate(fmt.Sprintf("system[%d]", i)); err != nil {
				return err
			}
			if block.CacheControl != nil {
				s.mark(block.CacheControl)
			}
		}
	case []any:
		for i, item := range sys {
			blockMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if blockMap["type"] == "text" {
				if text, ok := blockMap["text"].(string); ok {
					s.add(len(text))
				}
			}
			// Read the marker off the raw map rather than a round-tripped
			// ContentBlock: a block that fails to decode must still be counted
			// and still have its marker validated, or one odd block silently
			// rescales every other breakpoint.
			cc, err := parseCacheControl(blockMap["cache_control"], fmt.Sprintf("system[%d]", i))
			if err != nil {
				return err
			}
			if cc != nil {
				s.mark(cc)
			}
		}
	}
	// Any other shape (including nil) is dropped by the converter: no characters.
	return nil
}

// tokenOffsets converts the layout's character positions into token offsets,
// scaled by anchor (the token count of the prompt the layout describes) and
// clamped to total. The returned slice is non-decreasing.
func (l *PromptCacheLayout) tokenOffsets(anchor, total int) []int {
	offsets := make([]int, len(l.breakpoints))
	den := int64(l.totalChars) * 2
	for i, bp := range l.breakpoints {
		// Round to nearest rather than truncating so a breakpoint at the end
		// of the prompt lands on the last token instead of one short of it.
		// int64 throughout: chars*anchor overflows a 32-bit int on long prompts.
		offset := int(((int64(bp.chars)*int64(anchor))*2 + int64(l.totalChars)) / den)
		offset = min(max(offset, 0), total)
		if i > 0 && offset < offsets[i-1] {
			offset = offsets[i-1]
		}
		offsets[i] = offset
	}
	return offsets
}

// split partitions a prompt of total tokens, of which cached tokens were served
// from the KV cache, into Anthropic's three buckets. anchor is the token count
// of the prompt the layout was built from; pass total when they are the same.
//
// The result always satisfies input + cached + creation == total.
func (l *PromptCacheLayout) split(anchor, total, cached int) (input, creation int, byTTL CacheCreation) {
	// totalChars <= 0 cannot come out of ScanPromptCache, but guard it anyway:
	// tokenOffsets divides by it, and a panic here would take down a request.
	if l == nil || total <= 0 || l.totalChars <= 0 {
		return max(0, total-cached), 0, CacheCreation{}
	}
	if anchor <= 0 {
		anchor = total
	}

	offsets := l.tokenOffsets(anchor, total)

	// Tokens between the end of the cache hit and each breakpoint are written
	// to the cache by this request. prev tracks the high-water mark so the
	// per-breakpoint segments never overlap.
	prev := cached
	for i, offset := range offsets {
		if offset <= prev {
			continue
		}
		segment := offset - prev
		prev = offset
		creation += segment
		if l.breakpoints[i].ttl == CacheTTL1h {
			byTTL.Ephemeral1hInputTokens += segment
		} else {
			byTTL.Ephemeral5mInputTokens += segment
		}
	}

	return total - cached - creation, creation, byTTL
}

// UsageFromMetrics converts runner metrics into Anthropic usage, attributing
// newly prefilled tokens to cache_creation_input_tokens up to the request's
// last `cache_control` breakpoint. It is nil-safe: a nil layout reports no
// cache creation at all.
func (l *PromptCacheLayout) UsageFromMetrics(metrics api.Metrics) Usage {
	return l.usageFromMetrics(metrics, 0)
}

// UsageFromMetricsAnchored is UsageFromMetrics with an explicit anchor, for
// callers that accumulate metrics across several runner calls. Freezing the
// anchor at the first observed prompt size keeps the breakpoint token offsets
// from drifting as server-side follow-up turns grow the prompt.
func (l *PromptCacheLayout) UsageFromMetricsAnchored(metrics api.Metrics, anchor int) Usage {
	return l.usageFromMetrics(metrics, anchor)
}

func (l *PromptCacheLayout) usageFromMetrics(metrics api.Metrics, anchor int) Usage {
	total := max(0, metrics.PromptEvalCount)

	var cached *int
	if metrics.PromptEvalCachedCount != nil {
		count := min(max(0, *metrics.PromptEvalCachedCount), total)
		cached = &count
	} else if l != nil {
		// A request that asked for caching always gets both counters, even
		// when the runner reports no cache hits: clients rely on them to tell
		// "cold cache" apart from "caching not wired up".
		zero := 0
		cached = &zero
	}

	usage := Usage{
		InputTokens:          total - intValue(cached),
		CacheReadInputTokens: cached,
		OutputTokens:         metrics.EvalCount,
	}

	if l == nil {
		return usage
	}

	input, creation, byTTL := l.split(anchor, total, intValue(cached))
	usage.InputTokens = input
	usage.CacheCreationInputTokens = &creation
	usage.CacheCreation = &byTTL
	return usage
}

// applyKeepAlive extends the model's keep-alive to cover a 1 hour cache TTL.
//
// The Anthropic TTL is the only part of prompt caching with a local meaning
// beyond accounting: the KV cache lives in the loaded model, so unloading the
// model destroys it. Ollama's default keep-alive is 5 minutes, which matches
// the default ephemeral TTL exactly; a client that asked for `ttl: "1h"` is
// telling us it expects the prefix to survive a gap that long.
//
// This only ever extends. If the server is configured with a keep-alive of an
// hour or more (including "forever"), it is left alone.
func applyKeepAlive(req *api.ChatRequest, l *PromptCacheLayout) {
	if req == nil || !l.ExtendedTTL() || req.KeepAlive != nil {
		return
	}
	if envconfig.KeepAlive() >= time.Hour {
		return
	}
	req.KeepAlive = &api.Duration{Duration: time.Hour}
}
