package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and media that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, media []llm.MediaData, _ error) {
	var system []api.Message

	// TODO: This is only a truncation heuristic; llama-server handles the
	// actual image/media inputs. Replace this with projector/model-aware media
	// token accounting so image history is neither over-packed nor over-trimmed.
	// Clip images are represented as 768 tokens, each an embedding.
	imageNumTokens := 768

	lastMsgIdx := len(msgs) - 1
	currMsgIdx := 0

	if truncate {
		// Remove messages from the front until it fits in context, at the
		// stable cut points truncationStarts picks
		for _, i := range truncationStarts(msgs, opts.NumCtx) {
			// Collect system messages from the portion we're about to skip
			system = make([]api.Message, 0)
			for j := range i {
				if msgs[j].Role == "system" {
					system = append(system, msgs[j])
				}
			}

			p, err := renderPrompt(m, append(system, msgs[i:]...), tools, think)
			if err != nil {
				return "", nil, err
			}

			s, err := tokenize(ctx, p)
			if err != nil {
				return "", nil, err
			}

			ctxLen := len(s)
			if m.ProjectorPaths != nil {
				for _, msg := range msgs[i:] {
					ctxLen += imageNumTokens * len(msg.Images)
				}
			}

			if ctxLen <= opts.NumCtx {
				currMsgIdx = i
				break
			}

			// Must always include at least the last message
			if i == lastMsgIdx {
				currMsgIdx = lastMsgIdx
				break
			}
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[currMsgIdx:]))
	}

	renderMsgs, media, err := imageTaggedMessages(m, msgs, currMsgIdx, false)
	if err != nil {
		return "", nil, err
	}

	// truncate any messages that do not fit into the context window
	p, err := renderPrompt(m, append(system, renderMsgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, err
	}

	return p, media, nil
}

// truncationBlocks is how many cut points a context's worth of conversation
// has; charsPerToken turns that into a size in characters. Neither has to be
// accurate - whether a cut fits is decided by tokenizing - they only have to
// stay the same from one turn to the next.
const (
	truncationBlocks        = 4
	truncationCharsPerToken = 4
)

// truncationStarts returns the message indexes a conversation that does not
// fit in numCtx may start at, in the order to try them: 0 (nothing removed)
// first, the last message last.
//
// Removing just enough to fit would cut one more message every turn, and every
// such cut changes the prompt right after the system prompt: the whole rest of
// it is re-processed, turn after turn, and models whose memory cannot be
// rewound (recurrent, hybrid) do not even keep the system prompt. So the cuts
// are placed every numCtx/truncationBlocks tokens of conversation, counted from
// its start. Where they fall depends only on the messages before them, which
// do not change as the conversation grows, so a conversation keeps its cut
// until it outgrows it and then drops a whole block at once: the prompt is
// re-processed once per block, not once per turn.
//
// A cut never starts the prompt at a tool result, which would lose the call it
// answers. If no cut fits, every message after the last one is tried in turn,
// as before.
func truncationStarts(msgs []api.Message, numCtx int) []int {
	starts := []int{0}
	if len(msgs) == 0 {
		return starts
	}

	block := max(numCtx*truncationCharsPerToken/truncationBlocks, 1)
	next := block
	size := 0
	last := 0
	for i, msg := range msgs {
		if i > 0 && size >= next && msg.Role != "tool" && msg.Role != "system" {
			starts = append(starts, i)
			last = i
			for next <= size {
				next += block
			}
		}
		if msg.Role != "system" {
			size += truncationMessageSize(msg)
		}
	}

	for i := last + 1; i < len(msgs); i++ {
		starts = append(starts, i)
	}
	return starts
}

// truncationMessageSize estimates a message's size in characters. Thinking is
// left out: clients and renderers drop it from earlier turns inconsistently,
// and counting it would move the cuts when they do.
func truncationMessageSize(msg api.Message) int {
	size := len(msg.Content)
	for _, call := range msg.ToolCalls {
		size += len(call.Function.Name) + len(call.Function.Arguments.String())
	}
	// images are counted like chatPrompt counts them, 768 tokens each
	return size + len(msg.Images)*768*truncationCharsPerToken
}

func imageTaggedMessages(m *Model, msgs []api.Message, start int, clearImages bool) ([]api.Message, []llm.MediaData, error) {
	renderMsgs := slices.Clone(msgs)
	var media []llm.MediaData

	for cnt, msg := range renderMsgs[start:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return nil, nil, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			mediaData := llm.NewMediaData(len(media), i)
			media = append(media, mediaData)

			if m.Config.Renderer != "" {
				continue
			}

			// The prompt marker is still image-named for compatibility with
			// existing templates and llama-server media marker replacement.
			imgTag := fmt.Sprintf("[img-%d]", mediaData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}

		if m.Config.Renderer == "" {
			renderMsgs[start+cnt].Content = prefix + prompt
		}
		if clearImages {
			renderMsgs[start+cnt].Images = nil
		}
	}

	return renderMsgs, media, nil
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if m.Config.Renderer != "" {
		rendererName := resolveRendererName(m)
		rendered, err := renderers.RenderWithRenderer(rendererName, msgs, tools, think)
		if err != nil {
			return "", err
		}
		return rendered, nil
	}

	var b bytes.Buffer
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := m.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", err
	}
	return b.String(), nil
}
