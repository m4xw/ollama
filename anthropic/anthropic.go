package anthropic

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/types/model"
)

// Error types matching Anthropic API
type Error struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type ErrorResponse struct {
	Type      string `json:"type"` // always "error"
	Error     Error  `json:"error"`
	RequestID string `json:"request_id,omitempty"`
}

// NewError creates a new ErrorResponse with the appropriate error type based on HTTP status code
func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusUnauthorized:
		etype = "authentication_error"
	case http.StatusForbidden:
		etype = "permission_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	case http.StatusTooManyRequests:
		etype = "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		etype = "overloaded_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{
		Type:      "error",
		Error:     Error{Type: etype, Message: message},
		RequestID: generateID("req"),
	}
}

// Request types

// MessagesRequest represents an Anthropic Messages API request
type MessagesRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	Messages      []MessageParam  `json:"messages"`
	System        any             `json:"system,omitempty"` // string or []map[string]any (JSON-decoded ContentBlock)
	Stream        bool            `json:"stream,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Tools         []Tool          `json:"tools,omitempty"`
	ToolChoice    *ToolChoice     `json:"tool_choice,omitempty"`
	Thinking      *ThinkingConfig `json:"thinking,omitempty"`
	Metadata      *Metadata       `json:"metadata,omitempty"`
	OutputConfig  *OutputConfig   `json:"output_config,omitempty"`

	// CacheControl requests automatic prompt caching: the breakpoint is placed
	// on the last cacheable block rather than at an explicit marker.
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type OutputConfig struct {
	Effort string `json:"effort,omitempty"`
}

// MessageParam represents a message in the request
type MessageParam struct {
	Role    string         `json:"role"`    // "user" or "assistant"
	Content []ContentBlock `json:"content"` // always []ContentBlock; plain strings are normalized on unmarshal
}

func (m *MessageParam) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role

	var s string
	if err := json.Unmarshal(raw.Content, &s); err == nil {
		m.Content = []ContentBlock{{Type: "text", Text: &s}}
		return nil
	}

	return json.Unmarshal(raw.Content, &m.Content)
}

// ContentBlock represents a content block in a message.
// Text and Thinking use pointers so they serialize as the field being present (even if empty)
// only when set, which is required for SDK streaming accumulation.
type ContentBlock struct {
	Type string `json:"type"` // text, image, tool_use, tool_result, thinking, server_tool_use, web_search_tool_result

	// For text blocks - pointer so field only appears when set (SDK requires it for accumulation)
	Text *string `json:"text,omitempty"`

	// For text blocks with citations
	Citations []Citation `json:"citations,omitempty"`

	// For image blocks
	Source *ImageSource `json:"source,omitempty"`

	// For tool_use and server_tool_use blocks
	ID    string                        `json:"id,omitempty"`
	Name  string                        `json:"name,omitempty"`
	Input api.ToolCallFunctionArguments `json:"input,omitzero"`

	// For tool_result and web_search_tool_result blocks
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // string, []ContentBlock, []WebSearchResult, or WebSearchToolResultError
	IsError   bool   `json:"is_error,omitempty"`

	// For thinking blocks - pointer so field only appears when set (SDK requires it for accumulation)
	Thinking  *string `json:"thinking,omitempty"`
	Signature string  `json:"signature,omitempty"`

	// CacheControl marks this block as the end of a cacheable prompt prefix.
	// Only meaningful on request blocks; never set on generated output.
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// Citation represents a citation in a text block
type Citation struct {
	Type           string `json:"type"` // "web_search_result_location"
	URL            string `json:"url"`
	Title          string `json:"title"`
	EncryptedIndex string `json:"encrypted_index,omitempty"`
	CitedText      string `json:"cited_text,omitempty"`
}

// WebSearchResult represents a single web search result
type WebSearchResult struct {
	Type             string `json:"type"` // "web_search_result"
	URL              string `json:"url"`
	Title            string `json:"title"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
	PageAge          string `json:"page_age,omitempty"`
}

// WebSearchToolResultError represents an error from web search
type WebSearchToolResultError struct {
	Type      string `json:"type"` // "web_search_tool_result_error"
	ErrorCode string `json:"error_code"`
}

// ImageSource represents the source of an image
type ImageSource struct {
	Type      string `json:"type"` // "base64"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type        string          `json:"type,omitempty"` // "custom" for user-defined tools, or "web_search_20250305" for web search
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`

	// Web search specific fields
	MaxUses int `json:"max_uses,omitempty"`

	// CacheControl marks this tool as the end of a cacheable prompt prefix.
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// ToolChoice controls how the model uses tools
type ToolChoice struct {
	Type                   string `json:"type"` // "auto", "any", "tool", "none"
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// ThinkingConfig controls extended thinking
type ThinkingConfig struct {
	Type         string `json:"type"` // "enabled" or "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// Metadata for the request
type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}

// Response types

// MessagesResponse represents an Anthropic Messages API response
type MessagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"` // "message"
	Role         string         `json:"role"` // "assistant"
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason,omitempty"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// Usage contains token usage information. InputTokens is the uncached
// remainder only: total prompt size is InputTokens + CacheCreationInputTokens +
// CacheReadInputTokens.
type Usage struct {
	InputTokens              int            `json:"input_tokens"`
	CacheCreationInputTokens *int           `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int           `json:"cache_read_input_tokens,omitempty"`
	CacheCreation            *CacheCreation `json:"cache_creation,omitempty"`
	OutputTokens             int            `json:"output_tokens"`
}

// UsageFromMetrics separates total prompt tokens into uncached and cache-read
// counts. Requests that carry `cache_control` breakpoints should report through
// [PromptCacheLayout.UsageFromMetrics] instead so newly prefilled tokens are
// also attributed to cache_creation_input_tokens.
func UsageFromMetrics(metrics api.Metrics) Usage {
	return (*PromptCacheLayout)(nil).UsageFromMetrics(metrics)
}

func intValue(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

// Streaming event types

// MessageStartEvent is sent at the start of streaming
type MessageStartEvent struct {
	Type    string           `json:"type"` // "message_start"
	Message MessagesResponse `json:"message"`
}

// ContentBlockStartEvent signals the start of a content block
type ContentBlockStartEvent struct {
	Type         string       `json:"type"` // "content_block_start"
	Index        int          `json:"index"`
	ContentBlock ContentBlock `json:"content_block"`
}

// ContentBlockDeltaEvent contains incremental content updates
type ContentBlockDeltaEvent struct {
	Type  string `json:"type"` // "content_block_delta"
	Index int    `json:"index"`
	Delta Delta  `json:"delta"`
}

// Delta represents an incremental update
type Delta struct {
	Type        string `json:"type"` // "text_delta", "input_json_delta", "thinking_delta", "signature_delta"
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

// ContentBlockStopEvent signals the end of a content block
type ContentBlockStopEvent struct {
	Type  string `json:"type"` // "content_block_stop"
	Index int    `json:"index"`
}

// MessageDeltaEvent contains updates to the message
type MessageDeltaEvent struct {
	Type  string       `json:"type"` // "message_delta"
	Delta MessageDelta `json:"delta"`
	Usage DeltaUsage   `json:"usage"`
}

// MessageDelta contains stop information
type MessageDelta struct {
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// DeltaUsage contains cumulative token usage
type DeltaUsage struct {
	InputTokens              int            `json:"input_tokens"`
	CacheCreationInputTokens *int           `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int           `json:"cache_read_input_tokens,omitempty"`
	CacheCreation            *CacheCreation `json:"cache_creation,omitempty"`
	OutputTokens             int            `json:"output_tokens"`
}

// DeltaUsageFrom projects a Usage onto the streaming message_delta shape.
func DeltaUsageFrom(u Usage) DeltaUsage {
	return DeltaUsage{
		InputTokens:              u.InputTokens,
		CacheCreationInputTokens: u.CacheCreationInputTokens,
		CacheReadInputTokens:     u.CacheReadInputTokens,
		CacheCreation:            u.CacheCreation,
		OutputTokens:             u.OutputTokens,
	}
}

// MessageStopEvent signals the end of the message
type MessageStopEvent struct {
	Type string `json:"type"` // "message_stop"
}

// PingEvent is a keepalive event
type PingEvent struct {
	Type string `json:"type"` // "ping"
}

// StreamErrorEvent is an error during streaming
type StreamErrorEvent struct {
	Type  string `json:"type"` // "error"
	Error Error  `json:"error"`
}

// FromMessagesRequest converts an Anthropic MessagesRequest to an Ollama api.ChatRequest.
// An optional thinking descriptor preserves model-defined effort names for rendering.
func FromMessagesRequest(r MessagesRequest, thinking ...*model.Thinking) (*api.ChatRequest, error) {
	req, _, err := FromMessagesRequestWithCache(r, thinking...)
	return req, err
}

// FromMessagesRequestWithCache converts an Anthropic MessagesRequest to an
// Ollama api.ChatRequest and also returns the request's prompt cache layout,
// which the caller needs to report cache_creation_input_tokens. The layout is
// nil when the request carries no `cache_control` breakpoints.
// An optional thinking descriptor preserves model-defined effort names for rendering.
func FromMessagesRequestWithCache(r MessagesRequest, thinking ...*model.Thinking) (*api.ChatRequest, *PromptCacheLayout, error) {
	logutil.Trace("anthropic: converting request", "req", TraceMessagesRequest(r))

	promptCache, err := ScanPromptCache(r)
	if err != nil {
		logutil.Trace("anthropic: invalid cache_control", "err", err)
		return nil, nil, err
	}
	if n := promptCache.Breakpoints(); n > maxCacheBreakpoints {
		// Upstream rejects this with a 400. Ollama serves the whole prompt from
		// one KV slot, so every breakpoint is honored instead.
		slog.Debug("anthropic: request exceeds upstream cache_control breakpoint limit",
			"breakpoints", n, "limit", maxCacheBreakpoints)
	}

	var messages []api.Message

	if r.System != nil {
		switch sys := r.System.(type) {
		case string:
			if sys != "" {
				messages = append(messages, api.Message{Role: "system", Content: sys})
			}
		case []any:
			// System can be an array of content blocks
			var content strings.Builder
			for _, block := range sys {
				if blockMap, ok := block.(map[string]any); ok {
					if blockMap["type"] == "text" {
						if text, ok := blockMap["text"].(string); ok {
							content.WriteString(text)
						}
					}
				}
			}
			if content.Len() > 0 {
				messages = append(messages, api.Message{Role: "system", Content: content.String()})
			}
		}
	}

	for i, msg := range r.Messages {
		converted, err := convertMessage(msg)
		if err != nil {
			logutil.Trace("anthropic: message conversion failed", "index", i, "role", msg.Role, "err", err)
			return nil, nil, err
		}
		messages = append(messages, converted...)
	}

	options := make(map[string]any)

	options["num_predict"] = r.MaxTokens

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	}

	if r.TopK != nil {
		options["top_k"] = *r.TopK
	}

	if len(r.StopSequences) > 0 {
		options["stop"] = r.StopSequences
	}

	var tools api.Tools
	hasBuiltinWebSearch := false
	for _, t := range r.Tools {
		if strings.HasPrefix(t.Type, "web_search") {
			hasBuiltinWebSearch = true
			break
		}
	}

	for _, t := range r.Tools {
		// Anthropic built-in web_search maps to Ollama function name "web_search".
		// If a user-defined tool also uses that name in the same request, drop the
		// user-defined one to avoid ambiguous tool-call routing.
		if hasBuiltinWebSearch && !strings.HasPrefix(t.Type, "web_search") && t.Name == "web_search" {
			logutil.Trace("anthropic: dropping colliding custom web_search tool", "tool", TraceTool(t))
			continue
		}

		tool, _, err := convertTool(t)
		if err != nil {
			return nil, nil, err
		}
		tools = append(tools, tool)
	}

	var modelThinking *model.Thinking
	if len(thinking) > 0 {
		modelThinking = thinking[0]
	}
	think := thinkFromThinkingConfig(r.Thinking, r.OutputConfig, modelThinking)

	stream := r.Stream
	convertedRequest := &api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Options:  options,
		Stream:   &stream,
		Tools:    tools,
		Think:    think,
	}
	applyKeepAlive(convertedRequest, promptCache)

	logutil.Trace("anthropic: converted request", "req", TraceChatRequest(convertedRequest),
		"cache_breakpoints", promptCache.Breakpoints())

	return convertedRequest, promptCache, nil
}

// thinkFromThinkingConfig derives the Ollama think value from Anthropic's two
// thinking controls.
//
// `thinking.type` carries only on/off; the depth lives in
// `output_config.effort`. Real clients send BOTH (Claude Code sends
// `thinking: {type: "enabled"}` alongside an effort), so the effort has to win
// over a bare "enabled" — otherwise every request collapses to a plain
// `true` and the caller's effort setting is silently discarded. That was the
// behavior that made this endpoint diverge from /v1/chat/completions, which
// derives the level straight from `reasoning_effort`.
//
// An explicit `thinking: {type: "disabled"}` still wins over everything: it is
// the only way for a client to turn thinking off outright.
//
// Clients that predate output_config send neither field. Claude Code is one:
// it puts the depth in `thinking.budget_tokens` and signals "off" by omitting
// the thinking block, so both of those have to be honored here as well.
// modelThinking, when valid, carries the model's own effort names: a matching
// output_config.effort is then passed through exactly instead of being mapped
// onto the legacy low/medium/high/max levels.
func thinkFromThinkingConfig(thinking *ThinkingConfig, output *OutputConfig, modelThinking *model.Thinking) *api.ThinkValue {
	if thinking != nil && thinking.Type == "disabled" {
		return &api.ThinkValue{Value: false}
	}

	if modelThinking != nil && modelThinking.Valid() && output != nil && output.Effort != "" {
		return &api.ThinkValue{Value: output.Effort}
	}

	effort := ""
	if output != nil {
		effort = strings.ToLower(strings.TrimSpace(output.Effort))
	}

	budget := 0
	thinkingType := ""
	if thinking != nil {
		budget = thinking.BudgetTokens
		thinkingType = thinking.Type
	}
	logutil.Trace("anthropic: thinking controls", "type", thinkingType, "budget_tokens", budget, "effort", effort)

	if think := thinkFromEffort(effort); think != nil {
		return think
	}

	// A missing thinking block is how budget_tokens clients turn thinking off.
	// Claude Code never sends output_config; it sends
	// `thinking: {type: "enabled", budget_tokens: N}` when thinking is on and
	// omits the block entirely when it is off. Without this, "off" fell through
	// to the model default, which routes.go turns back into `true` for any
	// thinking-capable model — so the client could never switch thinking off.
	//
	// This is checked after the effort branch so a request that carries only
	// output_config.effort still gets its level.
	if thinking == nil {
		return &api.ThinkValue{Value: false}
	}

	switch thinking.Type {
	// "adaptive" is the current on-mode for Claude 4.6+ models; "enabled" is
	// the older budget_tokens form.
	case "enabled", "adaptive":
		if think := thinkFromBudgetTokens(thinking.BudgetTokens); think != nil {
			return think
		}
		return &api.ThinkValue{Value: true}
	}

	// Nothing requested: leave thinking at the model's default.
	return nil
}

// thinkFromBudgetTokens maps a budget_tokens thinking budget onto Ollama's
// think levels. Clients that predate output_config.effort — Claude Code among
// them — express depth only as a token budget, so without this every level
// from that client collapses to a plain `true`.
//
// The thresholds are anchored on the budgets Claude Code actually sends: 8000
// for its lowest level and 32000 for high. Each bound is inclusive of the
// anchor below it, so a level lands on the same think value whether the client
// sends the budget exactly or pads it slightly.
//
// The client has five levels and Ollama has four, so the bottom two collapse:
// 8000 is the smallest budget on the wire and "low" is the smallest think
// value, leaving nothing below it to map to.
//
// A zero or negative budget means the client sent no budget at all, which
// leaves the caller to fall back to a plain `true`.
func thinkFromBudgetTokens(budget int) *api.ThinkValue {
	switch {
	case budget <= 0:
		return nil
	case budget <= 8000:
		return &api.ThinkValue{Value: "low"}
	case budget < 32000:
		return &api.ThinkValue{Value: "medium"}
	case budget < 64000:
		return &api.ThinkValue{Value: "high"}
	default:
		return &api.ThinkValue{Value: "max"}
	}
}

// thinkFromEffort maps an Anthropic effort level onto Ollama's think levels,
// which are one tier coarser: Anthropic's xhigh sits between high and max.
//
// This mirrors openai.thinkFromReasoningEffort with one deliberate difference:
// xhigh clamps DOWN to high here and UP to max there. That predates this
// function and is pinned by TestFromMessagesRequest_WithOutputConfigEffortXHighMapsToHigh.
//
// An unrecognized effort is left alone rather than rejected: unlike
// reasoning_effort, effort is not the only thinking control on this endpoint,
// and a 400 would break a request whose thinking.type is perfectly valid.
func thinkFromEffort(effort string) *api.ThinkValue {
	switch effort {
	case "none":
		return &api.ThinkValue{Value: false}
	case "minimal":
		return &api.ThinkValue{Value: "low"}
	case "low", "medium", "high", "max":
		return &api.ThinkValue{Value: effort}
	case "xhigh", "ultra":
		return &api.ThinkValue{Value: "high"}
	default:
		return nil
	}
}

// convertMessage converts an Anthropic MessageParam to Ollama api.Message(s)
func convertMessage(msg MessageParam) ([]api.Message, error) {
	var messages []api.Message
	role := strings.ToLower(msg.Role)

	var textContent strings.Builder
	var images []api.ImageData
	var toolCalls []api.ToolCall
	var thinking string
	var toolResults []api.Message
	textBlocks := 0
	imageBlocks := 0
	toolUseBlocks := 0
	toolResultBlocks := 0
	serverToolUseBlocks := 0
	webSearchToolResultBlocks := 0
	thinkingBlocks := 0
	unknownBlocks := 0

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			textBlocks++
			if block.Text != nil {
				textContent.WriteString(*block.Text)
			}

		case "image":
			imageBlocks++
			if block.Source == nil {
				logutil.Trace("anthropic: invalid image source", "role", role)
				return nil, errors.New("invalid image source")
			}

			decoded, err := resolveImageSource(block.Source)
			if err != nil {
				logutil.Trace("anthropic: unsupported image source", "role", role, "source_type", block.Source.Type, "error", err)
				return nil, err
			}
			images = append(images, decoded)

		case "tool_use":
			toolUseBlocks++
			if block.ID == "" {
				logutil.Trace("anthropic: tool_use block missing id", "role", role)
				return nil, errors.New("tool_use block missing required 'id' field")
			}
			if block.Name == "" {
				logutil.Trace("anthropic: tool_use block missing name", "role", role)
				return nil, errors.New("tool_use block missing required 'name' field")
			}
			toolCalls = append(toolCalls, api.ToolCall{
				ID: block.ID,
				Function: api.ToolCallFunction{
					Name:      block.Name,
					Arguments: block.Input,
				},
			})

		case "tool_result":
			toolResultBlocks++
			resultContent, resultImages, err := convertToolResultContent(block.Content)
			if err != nil {
				logutil.Trace("anthropic: invalid tool_result content", "role", role, "error", err)
				return nil, err
			}

			toolResults = append(toolResults, api.Message{
				Role:       "tool",
				Content:    resultContent,
				Images:     resultImages,
				ToolCallID: block.ToolUseID,
			})

		case "thinking":
			thinkingBlocks++
			if block.Thinking != nil {
				thinking = *block.Thinking
			}

		case "server_tool_use":
			serverToolUseBlocks++
			toolCalls = append(toolCalls, api.ToolCall{
				ID: block.ID,
				Function: api.ToolCallFunction{
					Name:      block.Name,
					Arguments: block.Input,
				},
			})

		case "web_search_tool_result":
			webSearchToolResultBlocks++
			toolResults = append(toolResults, api.Message{
				Role:       "tool",
				Content:    formatWebSearchToolResultContent(block.Content),
				ToolCallID: block.ToolUseID,
			})
		default:
			unknownBlocks++
		}
	}

	if role == "user" && len(toolResults) > 0 {
		messages = append(messages, toolResults...)
	}

	if textContent.Len() > 0 || len(images) > 0 || len(toolCalls) > 0 || thinking != "" {
		m := api.Message{
			Role:      role,
			Content:   textContent.String(),
			Images:    images,
			ToolCalls: toolCalls,
			Thinking:  thinking,
		}
		messages = append(messages, m)
	}

	// Add tool results as separate messages.
	if role != "user" || len(toolResults) == 0 {
		messages = append(messages, toolResults...)
	}
	logutil.Trace("anthropic: converted block message",
		"role", role,
		"blocks", len(msg.Content),
		"text", textBlocks,
		"image", imageBlocks,
		"tool_use", toolUseBlocks,
		"tool_result", toolResultBlocks,
		"server_tool_use", serverToolUseBlocks,
		"web_search_result", webSearchToolResultBlocks,
		"thinking", thinkingBlocks,
		"unknown", unknownBlocks,
		"messages", TraceAPIMessages(messages),
	)

	return messages, nil
}

func formatWebSearchToolResultContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []WebSearchResult:
		var resultContent strings.Builder
		for _, item := range c {
			if item.Type != "web_search_result" {
				continue
			}
			fmt.Fprintf(&resultContent, "- %s: %s\n", item.Title, item.URL)
		}
		return resultContent.String()
	case []any:
		var resultContent strings.Builder
		for _, item := range c {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch itemMap["type"] {
			case "web_search_result":
				title, _ := itemMap["title"].(string)
				url, _ := itemMap["url"].(string)
				fmt.Fprintf(&resultContent, "- %s: %s\n", title, url)
			case "web_search_tool_result_error":
				errorCode, _ := itemMap["error_code"].(string)
				if errorCode == "" {
					return "web_search_tool_result_error"
				}
				return "web_search_tool_result_error: " + errorCode
			}
		}
		return resultContent.String()
	case map[string]any:
		if c["type"] == "web_search_tool_result_error" {
			errorCode, _ := c["error_code"].(string)
			if errorCode == "" {
				return "web_search_tool_result_error"
			}
			return "web_search_tool_result_error: " + errorCode
		}
		data, err := json.Marshal(c)
		if err != nil {
			return ""
		}
		return string(data)
	case WebSearchToolResultError:
		if c.ErrorCode == "" {
			return "web_search_tool_result_error"
		}
		return "web_search_tool_result_error: " + c.ErrorCode
	default:
		data, err := json.Marshal(c)
		if err != nil {
			return ""
		}
		return string(data)
	}
}

// convertTool converts an Anthropic Tool to an Ollama api.Tool, returning true if it's a server tool
func convertTool(t Tool) (api.Tool, bool, error) {
	if strings.HasPrefix(t.Type, "web_search") {
		props := api.NewToolPropertiesMap()
		props.Set("query", api.ToolProperty{
			Type:        api.PropertyType{"string"},
			Description: "The search query to look up on the web",
		})
		return api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "web_search",
				Description: "Search the web for current information. Use this to find up-to-date information about any topic.",
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Required:   []string{"query"},
					Properties: props,
				},
			},
		}, true, nil
	}

	var params api.ToolFunctionParameters
	if len(t.InputSchema) > 0 {
		if err := json.Unmarshal(t.InputSchema, &params); err != nil {
			logutil.Trace("anthropic: invalid tool schema", "tool", t.Name, "err", err)
			return api.Tool{}, false, fmt.Errorf("invalid input_schema for tool %q: %w", t.Name, err)
		}
	}

	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		},
	}, false, nil
}

// ToMessagesResponse converts an Ollama api.ChatResponse to an Anthropic MessagesResponse
func ToMessagesResponse(id string, r api.ChatResponse) MessagesResponse {
	return ToMessagesResponseWithCache(id, r, nil)
}

// ToMessagesResponseWithCache is ToMessagesResponse with the request's prompt
// cache layout, so usage reports cache_creation_input_tokens.
func ToMessagesResponseWithCache(id string, r api.ChatResponse, promptCache *PromptCacheLayout) MessagesResponse {
	var content []ContentBlock

	if r.Message.Thinking != "" {
		content = append(content, ContentBlock{
			Type:     "thinking",
			Thinking: ptr(r.Message.Thinking),
		})
	}

	if r.Message.Content != "" {
		content = append(content, ContentBlock{
			Type: "text",
			Text: ptr(r.Message.Content),
		})
	}

	for _, tc := range r.Message.ToolCalls {
		content = append(content, ContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: tc.Function.Arguments,
		})
	}

	stopReason := mapStopReason(r.DoneReason, len(r.Message.ToolCalls) > 0)

	return MessagesResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      r.Model,
		Content:    content,
		StopReason: stopReason,
		Usage:      promptCache.UsageFromMetrics(r.Metrics),
	}
}

// mapStopReason converts Ollama done_reason to Anthropic stop_reason
func mapStopReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_use"
	}

	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		if reason != "" {
			return "stop_sequence"
		}
		return ""
	}
}

// StreamConverter manages state for converting Ollama streaming responses to Anthropic format
type StreamConverter struct {
	ID    string
	Model string

	// PromptCache is the request's `cache_control` layout, or nil when the
	// request asked for no caching. Set it before the first Process call.
	PromptCache *PromptCacheLayout

	firstWrite           bool
	contentIndex         int
	inputTokens          int
	cacheReadTokens      *int
	cacheCreationTokens  *int
	cacheCreation        *CacheCreation
	outputTokens         int
	estimatedInputTokens int // Estimated tokens from request (used when actual metrics are 0)
	thinkingStarted      bool
	thinkingDone         bool
	textStarted          bool
	toolCallsSent        map[string]bool
}

func NewStreamConverter(id, model string, estimatedInputTokens int) *StreamConverter {
	return &StreamConverter{
		ID:                   id,
		Model:                model,
		firstWrite:           true,
		estimatedInputTokens: estimatedInputTokens,
		toolCallsSent:        make(map[string]bool),
	}
}

// recordUsage latches the prompt-side token counts from a converted usage.
func (c *StreamConverter) recordUsage(usage Usage) {
	c.inputTokens = usage.InputTokens
	c.cacheReadTokens = usage.CacheReadInputTokens
	c.cacheCreationTokens = usage.CacheCreationInputTokens
	c.cacheCreation = usage.CacheCreation
}

// StreamEvent represents a streaming event to be sent to the client
type StreamEvent struct {
	Event string
	Data  any
}

// Process converts an Ollama ChatResponse to Anthropic streaming events
func (c *StreamConverter) Process(r api.ChatResponse) []StreamEvent {
	var events []StreamEvent

	if c.firstWrite {
		c.firstWrite = false
		// Use actual metrics if available, otherwise use estimate
		usage := c.PromptCache.UsageFromMetrics(r.Metrics)
		// The runner reports PromptEvalCachedCount only once prefill has
		// finished, so the first streamed chunk usually carries no cached
		// count even when it already carries a prompt size. Attributing the
		// whole prefix to cache_creation here would stick: the Anthropic
		// SDKs' message_delta usage has no `cache_creation` field, and
		// message_start is where the spec puts the prompt-side counts, so a
		// client reading it keeps the fabricated split for the whole request
		// and sees a 0% hit rate on a warm cache. Report the prompt size as
		// plain input_tokens and claim nothing about the cache; message_delta
		// carries the real split once the runner reports it.
		if r.Metrics.PromptEvalCachedCount == nil {
			inputTokens := usage.InputTokens + intValue(usage.CacheReadInputTokens) + intValue(usage.CacheCreationInputTokens)
			if inputTokens == 0 {
				inputTokens = c.estimatedInputTokens
			}
			usage = Usage{InputTokens: inputTokens}
		}
		c.recordUsage(usage)

		events = append(events, StreamEvent{
			Event: "message_start",
			Data: MessageStartEvent{
				Type: "message_start",
				Message: MessagesResponse{
					ID:      c.ID,
					Type:    "message",
					Role:    "assistant",
					Model:   c.Model,
					Content: []ContentBlock{},
					Usage: Usage{
						InputTokens:              c.inputTokens,
						CacheCreationInputTokens: c.cacheCreationTokens,
						CacheReadInputTokens:     c.cacheReadTokens,
						CacheCreation:            c.cacheCreation,
						OutputTokens:             0,
					},
				},
			},
		})
	}

	if r.Message.Thinking != "" && !c.thinkingDone {
		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
			c.textStarted = false
		}

		if !c.thinkingStarted {
			c.thinkingStarted = true
			events = append(events, StreamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type:     "thinking",
						Thinking: ptr(""),
					},
				},
			})
		}

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:     "thinking_delta",
					Thinking: r.Message.Thinking,
				},
			},
		})
	}

	if r.Message.Content != "" {
		if c.thinkingStarted && !c.thinkingDone {
			c.thinkingDone = true
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
		}

		if !c.textStarted {
			c.textStarted = true
			events = append(events, StreamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type: "text",
						Text: ptr(""),
					},
				},
			})
		}

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type: "text_delta",
					Text: r.Message.Content,
				},
			},
		})
	}

	for _, tc := range r.Message.ToolCalls {
		if c.toolCallsSent[tc.ID] {
			continue
		}

		// Close thinking block if still open (thinking → tool_use without text in between)
		if c.thinkingStarted && !c.thinkingDone {
			c.thinkingDone = true
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
		}

		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
			c.textStarted = false
		}

		argsJSON, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			slog.Error("failed to marshal tool arguments", "error", err, "tool_id", tc.ID)
			continue
		}
		events = append(events, StreamEvent{
			Event: "content_block_start",
			Data: ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: c.contentIndex,
				ContentBlock: ContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: api.NewToolCallFunctionArguments(),
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:        "input_json_delta",
					PartialJSON: string(argsJSON),
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})

		c.toolCallsSent[tc.ID] = true
		c.contentIndex++
	}

	if r.Done {
		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
		} else if c.thinkingStarted && !c.thinkingDone {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
		}

		usage := c.PromptCache.UsageFromMetrics(r.Metrics)
		c.recordUsage(usage)
		c.outputTokens = usage.OutputTokens
		stopReason := mapStopReason(r.DoneReason, len(c.toolCallsSent) > 0)

		events = append(events, StreamEvent{
			Event: "message_delta",
			Data: MessageDeltaEvent{
				Type: "message_delta",
				Delta: MessageDelta{
					StopReason: stopReason,
				},
				Usage: DeltaUsage{
					InputTokens:              c.inputTokens,
					CacheCreationInputTokens: c.cacheCreationTokens,
					CacheReadInputTokens:     c.cacheReadTokens,
					CacheCreation:            c.cacheCreation,
					OutputTokens:             c.outputTokens,
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "message_stop",
			Data: MessageStopEvent{
				Type: "message_stop",
			},
		})
	}

	return events
}

// generateID generates a unique ID with the given prefix using crypto/rand
func generateID(prefix string) string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		// Fallback to time-based ID if crypto/rand fails
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return fmt.Sprintf("%s_%x", prefix, b)
}

// GenerateMessageID generates a unique message ID
func GenerateMessageID() string {
	return generateID("msg")
}

func resolveImageSource(source *ImageSource) (api.ImageData, error) {
	if source.Type != "base64" {
		return nil, fmt.Errorf("invalid image source type: %s. Only base64 images are supported.", source.Type)
	}

	decoded, err := base64.StdEncoding.DecodeString(source.Data)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 image data: %w", err)
	}

	return decoded, nil
}

func convertToolResultContent(content any) (string, []api.ImageData, error) {
	switch c := content.(type) {
	case nil:
		return "", nil, nil
	case string:
		return c, nil, nil
	case []any:
		var text strings.Builder
		var images []api.ImageData

		for _, cb := range c {
			cbMap, ok := cb.(map[string]any)
			if !ok {
				continue
			}

			switch cbMap["type"] {
			case "text":
				if t, ok := cbMap["text"].(string); ok {
					text.WriteString(t)
				}
			case "image":
				rawSource, ok := cbMap["source"].(map[string]any)
				if !ok {
					return "", nil, errors.New("invalid tool_result image source")
				}

				var source ImageSource
				if rawType, ok := rawSource["type"].(string); ok {
					source.Type = rawType
				}
				if rawMediaType, ok := rawSource["media_type"].(string); ok {
					source.MediaType = rawMediaType
				}
				if rawData, ok := rawSource["data"].(string); ok {
					source.Data = rawData
				}

				img, err := resolveImageSource(&source)
				if err != nil {
					return "", nil, err
				}
				images = append(images, img)
			}
		}

		return text.String(), images, nil
	default:
		return "", nil, nil
	}
}

// ptr returns a pointer to the given string value
func ptr(s string) *string {
	return &s
}

// CountTokensRequest represents an Anthropic count_tokens request
type CountTokensRequest struct {
	Model    string          `json:"model"`
	Messages []MessageParam  `json:"messages"`
	System   any             `json:"system,omitempty"`
	Tools    []Tool          `json:"tools,omitempty"`
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// EstimateInputTokens estimates input tokens from a MessagesRequest (reuses CountTokensRequest logic)
func EstimateInputTokens(req MessagesRequest) int {
	return EstimateCountTokens(CountTokensRequest{
		Model:    req.Model,
		Messages: req.Messages,
		System:   req.System,
		Tools:    req.Tools,
		Thinking: req.Thinking,
	})
}

// CountTokensResponse represents an Anthropic count_tokens response
type CountTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

// EstimateCountTokens returns a rough estimate of tokens (len/4).
// TODO: Replace with actual tokenization via Tokenize API for accuracy.
// Current len/4 heuristic is a rough approximation (~4 chars/token average).
func EstimateCountTokens(req CountTokensRequest) int {
	var totalLen int

	// Count system prompt
	totalLen += countAnyContent(req.System)

	for _, msg := range req.Messages {
		// Count role (always present)
		totalLen += len(msg.Role)
		// Count content
		totalLen += countAnyContent(msg.Content)
	}

	for _, tool := range req.Tools {
		totalLen += len(tool.Name) + len(tool.Description) + len(tool.InputSchema)
	}

	// Return len/4 as rough token estimate, minimum 1 if there's any content
	tokens := totalLen / 4
	if tokens == 0 && (len(req.Messages) > 0 || req.System != nil) {
		tokens = 1
	}
	return tokens
}

func countAnyContent(content any) int {
	if content == nil {
		return 0
	}

	switch c := content.(type) {
	case string:
		return len(c)
	case []ContentBlock:
		total := 0
		for _, block := range c {
			total += countContentBlock(block)
		}
		return total
	case []any:
		total := 0
		for _, item := range c {
			data, err := json.Marshal(item)
			if err != nil {
				continue
			}
			var block ContentBlock
			if err := json.Unmarshal(data, &block); err == nil {
				total += countContentBlock(block)
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

func countContentBlock(block ContentBlock) int {
	total := 0
	if block.Text != nil {
		total += len(*block.Text)
	}
	if block.Thinking != nil {
		total += len(*block.Thinking)
	}
	if block.Type == "tool_use" || block.Type == "tool_result" {
		if data, err := json.Marshal(block); err == nil {
			total += len(data)
		}
	}
	return total
}

// OllamaWebSearchRequest represents a request to the Ollama web search API
type OllamaWebSearchRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// OllamaWebSearchResult represents a single search result from Ollama API
type OllamaWebSearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

// OllamaWebSearchResponse represents the response from the Ollama web search API
type OllamaWebSearchResponse struct {
	Results []OllamaWebSearchResult `json:"results"`
}

var WebSearchEndpoint = "https://ollama.com/api/web_search"

func WebSearch(ctx context.Context, query string, maxResults int) (*OllamaWebSearchResponse, error) {
	if internalcloud.Disabled() {
		logutil.TraceContext(ctx, "anthropic: web search blocked", "reason", "cloud_disabled")
		return nil, errors.New(internalcloud.DisabledError("web search is unavailable"))
	}

	if maxResults <= 0 {
		maxResults = 5
	}
	if maxResults > 10 {
		maxResults = 10
	}

	reqBody := OllamaWebSearchRequest{
		Query:      query,
		MaxResults: maxResults,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal web search request: %w", err)
	}

	searchURL, err := url.Parse(WebSearchEndpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to parse web search URL: %w", err)
	}
	logutil.TraceContext(ctx, "anthropic: web search request",
		"query", TraceTruncateString(query),
		"max_results", maxResults,
		"url", searchURL.String(),
	)

	q := searchURL.Query()
	q.Set("ts", strconv.FormatInt(time.Now().Unix(), 10))
	searchURL.RawQuery = q.Encode()

	signature := ""
	if strings.EqualFold(searchURL.Hostname(), "ollama.com") {
		challenge := fmt.Sprintf("%s,%s", http.MethodPost, searchURL.RequestURI())
		signature, err = auth.Sign(ctx, []byte(challenge))
		if err != nil {
			return nil, fmt.Errorf("failed to sign web search request: %w", err)
		}
	}
	logutil.TraceContext(ctx, "anthropic: web search auth", "signed", signature != "")

	req, err := http.NewRequestWithContext(ctx, "POST", searchURL.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create web search request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("web search request failed: %w", err)
	}
	defer resp.Body.Close()
	logutil.TraceContext(ctx, "anthropic: web search response", "status", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("web search returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var searchResp OllamaWebSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode web search response: %w", err)
	}
	logutil.TraceContext(ctx, "anthropic: web search results", "count", len(searchResp.Results))

	return &searchResp, nil
}

func ConvertOllamaToAnthropicResults(ollamaResults *OllamaWebSearchResponse) []WebSearchResult {
	var results []WebSearchResult
	for _, r := range ollamaResults.Results {
		results = append(results, WebSearchResult{
			Type:  "web_search_result",
			URL:   r.URL,
			Title: r.Title,
		})
	}
	return results
}
