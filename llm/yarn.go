package llm

import (
	"math"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
)

// yarnCompatScaledArches have rope scaling injected by llama/compat at load
// time, so their GGUF metadata does not show it. Extending them again would
// replace that scaling rather than stack on it.
var yarnCompatScaledArches = map[string]bool{
	"gemma3":  true,
	"gpt-oss": true,
}

// YaRNExtendable reports whether an explicit num_ctx above the training
// context of f is served by YaRN-scaling RoPE (OLLAMA_YARN) instead of being
// clamped to the training context.
//
// Models that already ship rope scaling are left alone: their context_length
// is already the extended one, and a second factor would scale twice.
func YaRNExtendable(f *gguf.Model) bool {
	if f == nil || !envconfig.YaRN() {
		return false
	}
	kv := f.KV()
	if kv.ContextLength() == 0 || kv.Has("pooling_type") || kv.String("decision.type") != "" {
		return false
	}
	if yarnCompatScaledArches[kv.Architecture()] {
		return false
	}
	switch strings.ToLower(kv.String("rope.scaling.type")) {
	case "", "none":
	default:
		return false
	}
	return !kv.Has("rope.scaling.factor") && !kv.Has("rope.scaling.original_context_length")
}

// yarnOrigContext returns the training context to YaRN-extend from when a
// load of f asks for numCtx tokens per slot, or 0 when no scaling applies.
func yarnOrigContext(f *gguf.Model, numCtx int) int {
	if !YaRNExtendable(f) {
		return 0
	}
	trainCtx := int(f.KV().ContextLength())
	if numCtx <= trainCtx {
		return 0
	}
	return trainCtx
}

// yarnArgs scales RoPE by numCtx/origCtx. llama.cpp then raises the model's
// n_ctx_train to numCtx (llama_init_from_model), so llama-server's per-slot
// cap at the training context no longer binds.
func yarnArgs(origCtx, numCtx int) []string {
	if origCtx <= 0 || numCtx <= origCtx {
		return nil
	}
	return []string{
		"--rope-scaling", "yarn",
		"--rope-scale", strconv.FormatFloat(float64(yarnScale(origCtx, numCtx)), 'g', -1, 32),
		"--yarn-orig-ctx", strconv.Itoa(origCtx),
	}
}

// yarnScale returns the smallest float32 factor for which llama.cpp's
// float32 n_ctx_train = origCtx / (1 / factor), truncated, reaches numCtx.
// The plain quotient can land one token short (1048315 over 262144 gives
// 1048314), and llama-server would then cap the slot below num_ctx.
func yarnScale(origCtx, numCtx int) float32 {
	factor := float32(numCtx) / float32(origCtx)
	for i := 0; i < 64 && yarnTrainContext(origCtx, factor) < uint32(numCtx); i++ {
		factor = math.Nextafter32(factor, float32(math.Inf(1)))
	}
	return factor
}

// yarnTrainContext mirrors llama_init_from_model's re-adjusted n_ctx_train.
func yarnTrainContext(origCtx int, factor float32) uint32 {
	freqScale := float32(1 / factor)
	return uint32(float32(float32(origCtx) / freqScale))
}
