package llm

import (
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/ml"
)

// hostExpertsPlan is where a Mixture-of-Experts model's experts live and how
// much VRAM is spent on caching them. A MoE whose experts do not fit next to
// the dense layers is better served by --cpu-moe than by llama-server's --fit:
// fit drops whole layers to the CPU, which takes the dense attention / SSM
// weights along with the experts, while --cpu-moe keeps every dense layer on
// the GPU and only streams the few experts a token routes to from host memory.
// The VRAM that is left afterwards goes to --moe-cache-mib, an LRU of hot
// experts on the GPU.
type hostExpertsPlan struct {
	enabled  bool
	cacheMiB int
	reason   string
}

// expertTensorBytes is the size of the routed expert weights (*_exps tensors).
// The shared expert (*_shexp) is dense per token and stays on the GPU.
func expertTensorBytes(f *gguf.Model) uint64 {
	var n uint64
	for _, t := range f.Tensors().Items() {
		if strings.Contains(t.Name, "_exps.") {
			n += uint64(t.NumBytes())
		}
	}
	return n
}

// attentionKVBytes estimates the KV cache of the attention layers for one
// context. Hybrid models (Nemotron-H, Jamba, ...) only have a KV cache on the
// layers whose head_count_kv is non-zero; the recurrent state of the SSM
// layers is small and goes into the headroom.
func attentionKVBytes(f *gguf.Model, kvCacheType string, numCtx uint64) uint64 {
	kv := f.KV()

	// the metadata helpers prefix the architecture themselves
	value := kv.KeyValue("attention.head_count_kv")
	counts := value.Uints()
	if len(counts) == 0 {
		for _, n := range value.Ints() {
			counts = append(counts, uint64(max(n, 0)))
		}
	}

	var layers, heads uint64
	if len(counts) > 0 {
		for _, n := range counts {
			if n > 0 {
				layers++
				heads = max(heads, n)
			}
		}
	} else {
		layers = kv.BlockCount()
		heads = kv.Uint("attention.head_count_kv", kv.HeadCountMax())
	}
	if layers == 0 || heads == 0 {
		return 0
	}

	headDim := kv.Uint("attention.key_length", 0)
	if headDim == 0 {
		if n := kv.HeadCountMax(); n > 0 {
			headDim = kv.EmbeddingLength() / n
		}
	}

	var bytesPerElement float64
	switch kvCacheType {
	case "q4_0":
		bytesPerElement = 0.5625
	case "q8_0":
		bytesPerElement = 1.0625
	case "", "f16", "bf16":
		bytesPerElement = 2
	default:
		bytesPerElement = 4
	}

	return uint64(float64(2*layers*heads*headDim*numCtx) * bytesPerElement)
}

func fileSize(path string) uint64 {
	if path == "" {
		return 0
	}
	if info, err := os.Stat(path); err == nil {
		return uint64(info.Size())
	}
	return 0
}

// planHostExperts decides --cpu-moe and --moe-cache-mib for a model.
func planHostExperts(f *gguf.Model, gpus []ml.DeviceInfo, opts api.Options, numParallel int, kvCacheType, draftModelPath string) hostExpertsPlan {
	expertBytes := expertTensorBytes(f)
	if expertBytes == 0 || len(gpus) == 0 || opts.NumGPU == 0 {
		return hostExpertsPlan{}
	}

	// one cache per GPU is split like the layers; a single GPU is the case
	// this is tuned for, so the smallest device bounds the plan
	var free, total uint64
	for i, gpu := range gpus {
		if i == 0 || gpu.FreeMemory < free {
			free = gpu.FreeMemory
			total = gpu.TotalMemory
		}
	}
	if free == 0 {
		free = total
	}

	const GiB = 1 << 30
	// the input embeddings are looked up on the CPU whatever the offload
	denseBytes := f.Tensors().Size() - expertBytes - f.Tensors().Size("token_embd.")
	// compute buffers, the CUDA context, the SSM state, cache slot maps
	const headroom = 5 * GiB / 4

	plan := hostExpertsPlan{}
	switch envconfig.CPUMoE() {
	case "0":
		return plan
	case "1":
		plan.enabled = true
		plan.reason = "OLLAMA_CPU_MOE=1"
	default:
		if denseBytes+expertBytes+headroom > free {
			plan.enabled = true
			plan.reason = "experts do not fit VRAM"
		}
	}
	if !plan.enabled {
		return plan
	}

	kvBytes := attentionKVBytes(f, kvCacheType, uint64(opts.NumCtx)*uint64(max(numParallel, 1)))
	draftBytes := fileSize(draftModelPath)

	if plan.cacheMiB = envconfig.MoECacheMiB(); plan.cacheMiB < 0 {
		spent := denseBytes + kvBytes + draftBytes + headroom
		if free > spent {
			plan.cacheMiB = int((free - spent) >> 20)
			plan.cacheMiB -= plan.cacheMiB % 128
		}
		if plan.cacheMiB < 256 {
			plan.cacheMiB = 0
		}
	}

	slog.Info("keeping MoE experts in host memory",
		"reason", plan.reason,
		"experts", format.HumanBytes2(expertBytes),
		"dense", format.HumanBytes2(denseBytes),
		"kv", format.HumanBytes2(kvBytes),
		"draft", format.HumanBytes2(draftBytes),
		"vram_free", format.HumanBytes2(free),
		"moe_cache_mib", plan.cacheMiB)
	return plan
}

// hostExpertsArgs are the llama-server flags of a plan.
func hostExpertsArgs(plan hostExpertsPlan) []string {
	if !plan.enabled {
		return nil
	}
	args := []string{"--cpu-moe"}
	if plan.cacheMiB > 0 {
		args = append(args, "--moe-cache-mib", strconv.Itoa(plan.cacheMiB))
	}
	return args
}
