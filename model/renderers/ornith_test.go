package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestOrnithRendererMatchesAssistantHistoryThinkBlocks(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "Say hello."},
		{Role: "assistant", Content: "Hello."},
		{Role: "user", Content: "Now say bye."},
	}

	got, err := RenderWithRenderer("ornith", msgs, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
<think>

</think>

Hello.<|im_end|>
<|im_start|>user
Now say bye.<|im_end|>
<|im_start|>assistant
<think>
`
	if got != want {
		t.Fatalf("unexpected Ornith render\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

func TestOrnithRendererKeepsAssistantThinkBlocksWhenThinkingDisabled(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "Say hello."},
		{
			Role:     "assistant",
			Thinking: "Keep it short.",
			Content:  "Hello.",
		},
		{Role: "user", Content: "Now say bye."},
	}

	got, err := RenderWithRenderer("ornith", msgs, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
<think>
Keep it short.
</think>

Hello.<|im_end|>
<|im_start|>user
Now say bye.<|im_end|>
<|im_start|>assistant
<think>

</think>

`
	if got != want {
		t.Fatalf("unexpected Ornith render with thinking disabled\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

// Ornith must honor the requested reasoning effort. Before this was wired up
// every level rendered a byte-identical prompt, so setting an effort had no
// effect on the model at all.
func TestOrnithRendererHonorsReasoningEffort(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "What is 17*23?"}}

	render := func(t *testing.T, think *api.ThinkValue) string {
		t.Helper()
		got, err := RenderWithRenderer("ornith", msgs, nil, think)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}

	neutral := render(t, &api.ThinkValue{Value: "medium"})
	low := render(t, &api.ThinkValue{Value: "low"})
	high := render(t, &api.ThinkValue{Value: "high"})

	if low == neutral {
		t.Fatal("low effort renders the same prompt as medium; the level is not reaching the model")
	}
	if high == neutral {
		t.Fatal("high effort renders the same prompt as medium; the level is not reaching the model")
	}
	if low == high {
		t.Fatal("low and high effort render the same prompt")
	}
	if !strings.Contains(low, "Reasoning effort is set to low") {
		t.Fatalf("low effort prompt missing its instruction:\n%q", low)
	}
	if !strings.Contains(high, "Reasoning effort is set to xhigh") {
		t.Fatalf("high effort prompt missing its instruction:\n%q", high)
	}

	// Unset and plain-on must keep rendering exactly as before: Ornith has no
	// xhigh default of its own, unlike the qwen3.8 renderer.
	unset := render(t, nil)
	if unset != neutral {
		t.Fatalf("unset effort changed the default prompt:\n--- unset ---\n%q\n--- neutral ---\n%q", unset, neutral)
	}
	if on := render(t, &api.ThinkValue{Value: true}); on != neutral {
		t.Fatalf("think=true changed the default prompt:\n%q", on)
	}
}
