package llm

import (
	"slices"
	"strconv"
	"testing"

	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

func TestYaRNExtendable(t *testing.T) {
	tests := []struct {
		name string
		yarn string
		kv   gguftest.KV
		want bool
	}{
		{"qwen35moe", "1", gguftest.KV{"general.architecture": "qwen35moe", "qwen35moe.context_length": uint32(262144)}, true},
		{"disabled", "", gguftest.KV{"general.architecture": "qwen35moe", "qwen35moe.context_length": uint32(262144)}, false},
		{"scaling none", "1", gguftest.KV{"general.architecture": "llama", "llama.context_length": uint32(8192), "llama.rope.scaling.type": "none"}, true},
		{"already yarn", "1", gguftest.KV{"general.architecture": "qwen3", "qwen3.context_length": uint32(131072), "qwen3.rope.scaling.type": "yarn", "qwen3.rope.scaling.factor": float32(4)}, false},
		{"linear factor", "1", gguftest.KV{"general.architecture": "llama", "llama.context_length": uint32(8192), "llama.rope.scaling.factor": float32(2)}, false},
		{"compat scaled", "1", gguftest.KV{"general.architecture": "gpt-oss", "gpt-oss.context_length": uint32(131072)}, false},
		{"embedding", "1", gguftest.KV{"general.architecture": "bert", "bert.context_length": uint32(512), "bert.pooling_type": uint32(1)}, false},
		{"no context length", "1", gguftest.KV{"general.architecture": "llama"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_YARN", tt.yarn)
			if got := YaRNExtendable(loadTestGGUF(t, tt.kv)); got != tt.want {
				t.Fatalf("YaRNExtendable = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestYaRNArgs(t *testing.T) {
	if args := yarnArgs(0, 1048576); args != nil {
		t.Fatalf("no orig ctx: got %v", args)
	}
	if args := yarnArgs(262144, 262144); args != nil {
		t.Fatalf("within training ctx: got %v", args)
	}

	want := []string{"--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "262144"}
	if args := yarnArgs(262144, 1048576); !slices.Equal(args, want) {
		t.Fatalf("got %v, want %v", args, want)
	}
}

// llama.cpp parses --rope-scale with stof, inverts it and re-derives
// n_ctx_train in float32; the slot must not end up below num_ctx.
func TestYaRNScaleReachesNumCtx(t *testing.T) {
	for _, numCtx := range []int{262145, 300000, 500000, 999999, 1000000, 1010000, 1048315, 1048576} {
		args := yarnArgs(262144, numCtx)
		parsed, err := strconv.ParseFloat(args[3], 32)
		if err != nil {
			t.Fatal(err)
		}
		if got := yarnTrainContext(262144, float32(parsed)); got < uint32(numCtx) {
			t.Errorf("num_ctx %d: llama.cpp n_ctx_train %d", numCtx, got)
		}
	}
}
