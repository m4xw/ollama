package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// On-disk prompt cache.
//
// llama-server's prompt cache (--cache-ram) lives in host memory and dies with
// the process, so an unload or a restart throws away every prefill. When
// OLLAMA_PROMPT_CACHE_DIR is set, Ollama mirrors each slot's state to that
// directory through llama-server's /slots save/restore endpoints: a slot is
// written after every completion and read back once the server is ready again.
//
// What llama.cpp serializes is the sequence state plus its tokens - not the
// slot's context checkpoints. A restored slot can therefore be *extended* (the
// client re-sends the same prefix and appends to it), but a prompt that
// diverges inside the restored range still needs a full re-prefill on models
// whose memory cannot be rewound, exactly as it would with the RAM cache.
//
// Entries live for the model's effective keep-alive: the same signal that says
// how long the client expects the prefix to stay warm, including the hour that
// an Anthropic `cache_control` of "1h" asks for.

// promptCacheStateVersion is embedded in every entry name. Bump it when the
// meaning of an entry changes so old files are ignored rather than restored.
const promptCacheStateVersion = 1

// promptCacheRequestTimeout bounds a single save or restore. Multi-hundred-MiB
// states are normal, so this is generous; it exists to keep a wedged runner
// from leaking a goroutine forever.
const promptCacheRequestTimeout = 5 * time.Minute

// promptCacheStatusTimeout bounds the slot-state query a chat-path save makes
// to find out which slots moved.
const promptCacheStatusTimeout = 30 * time.Second

// promptCacheFlushTimeout bounds how long an unload waits for an in-flight save
// to finish before it stops the runner anyway.
const promptCacheFlushTimeout = 30 * time.Second

// promptCacheStore is the per-runner handle on the on-disk cache. It is nil
// when OLLAMA_PROMPT_CACHE_DIR is unset.
type promptCacheStore struct {
	dir   string // directory holding the entries
	key   string // identity of the runner configuration that wrote them
	slots int    // number of slots the runner was started with

	// ttl is the effective keep-alive in nanoseconds, refreshed by the
	// scheduler as requests arrive. Zero means entries expire immediately,
	// which disables saving.
	ttl atomic.Int64

	// saving marks a slot whose save is still in flight, so a burst of short
	// completions cannot queue up several multi-hundred-MiB writes; dirty
	// records that the slot moved on while that write was running.
	saving []atomic.Bool
	dirty  []atomic.Bool

	// mirrored records, per slot, the state that was last written out. It is
	// how a save decides whether a slot has moved, on the chat path where the
	// response does not say which slot served the request.
	mirroredMu sync.Mutex
	mirrored   []string

	// restored guards the read-back. The scheduler waits for the runner twice
	// (llm.Load, then again on the goroutine that publishes it), and reading a
	// multi-hundred-MiB state a second time is pure latency. It is replaced
	// when the process restarts, whose slots start empty again.
	restored atomic.Pointer[sync.Once]
}

// newPromptCacheStore returns the store for a runner, or nil when the on-disk
// cache is disabled or its directory cannot be created.
func newPromptCacheStore(launch llamaServerLaunchConfig) *promptCacheStore {
	dir := envconfig.PromptCacheDir()
	if dir == "" {
		return nil
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Warn("disabling on-disk prompt cache", "dir", dir, "error", err)
		return nil
	}

	slots := max(launch.numParallel, 1)
	store := &promptCacheStore{
		dir:      dir,
		key:      promptCacheIdentity(launch),
		slots:    slots,
		saving:   make([]atomic.Bool, slots),
		dirty:    make([]atomic.Bool, slots),
		mirrored: make([]string, slots),
	}
	store.ttl.Store(int64(envconfig.KeepAlive()))
	store.armRestore()
	return store
}

// promptCacheSlotSavePath is the value passed to llama-server's
// --slot-save-path. llama-server concatenates it with the request's filename
// without inserting a separator, so it has to end in one.
func promptCacheSlotSavePath(dir string) string {
	return strings.TrimRight(dir, `/\`) + string(os.PathSeparator)
}

// promptCacheIdentity derives a short key from everything that changes the
// layout of a saved state. A configuration change yields a new key, so stale
// entries are simply never looked at (and are pruned once they expire).
func promptCacheIdentity(launch llamaServerLaunchConfig) string {
	h := sha256.New()
	fmt.Fprintf(h, "v%d\n", promptCacheStateVersion)
	fmt.Fprintf(h, "model=%s\n", launch.modelPath)
	fmt.Fprintf(h, "arch=%s\n", launch.modelArch)
	fmt.Fprintf(h, "ctx=%d\n", launch.opts.NumCtx)
	fmt.Fprintf(h, "parallel=%d\n", launch.numParallel)
	fmt.Fprintf(h, "kv=%s\n", launch.kvCacheType)
	fmt.Fprintf(h, "embedding=%t\n", launch.embedding)
	fmt.Fprintf(h, "draft=%s|%s\n", launch.draftType, launch.config.DraftModelPath)
	for _, adapter := range launch.adapters {
		fmt.Fprintf(h, "lora=%s\n", adapter)
	}
	for _, projector := range launch.projectors {
		fmt.Fprintf(h, "mmproj=%s\n", projector)
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// promptCacheFilePrefix is shared by every entry so that pruning can recognize
// files this cache owns without touching anything else in the directory.
const promptCacheFilePrefix = "ollama-prompt-"

// promptCacheFilename names the entry for one slot. llama-server validates the
// name and joins it onto --slot-save-path itself, so it must stay a plain
// basename.
func promptCacheFilename(key string, slot int) string {
	return fmt.Sprintf("%s%s-%d.bin", promptCacheFilePrefix, key, slot)
}

// promptCacheTempSuffix marks the scratch entry a save writes before it takes
// the place of the live one.
const promptCacheTempSuffix = ".tmp"

// promptCacheTempFilename names that scratch entry. It keeps the shared prefix
// so pruning still recognizes the file as this cache's own.
func promptCacheTempFilename(key string, slot int) string {
	return promptCacheFilename(key, slot) + promptCacheTempSuffix
}

// promptCacheMaxTTL caps how far into the future an entry can be stamped. A
// keep-alive of "forever" is a keep-alive for this process, not a promise to
// hold a slot state on disk indefinitely.
const promptCacheMaxTTL = 30 * 24 * time.Hour

// promptCacheExpiry is when an entry written now under ttl stops being valid.
// A ttl of zero or less expires immediately, which is what a keep-alive of zero
// ("unload as soon as the request finishes") means for the cache too.
func promptCacheExpiry(now time.Time, ttl time.Duration) time.Time {
	if ttl <= 0 {
		return now
	}
	return now.Add(min(ttl, promptCacheMaxTTL))
}

// promptCacheExpired reports whether an entry stamped with expiresAt is spent.
//
// The expiry is stamped onto the file when it is written, rather than judged
// from its age when it is read, because the reader does not know the keep-alive
// the writer had: a later request with a shorter keep-alive would otherwise
// throw away an entry that was promised a longer life.
func promptCacheExpired(expiresAt, now time.Time) bool {
	return !now.Before(expiresAt)
}

func (p *promptCacheStore) keepAlive() time.Duration {
	if p == nil {
		return 0
	}
	return time.Duration(p.ttl.Load())
}

func (p *promptCacheStore) setKeepAlive(d time.Duration) {
	if p == nil {
		return
	}
	p.ttl.Store(int64(d))
}

// armRestore lets the next restore run, and is called for every start of the
// llama-server process.
func (p *promptCacheStore) armRestore() {
	if p == nil {
		return
	}
	p.restored.Store(&sync.Once{})
}

func (p *promptCacheStore) path(slot int) string {
	return filepath.Join(p.dir, promptCacheFilename(p.key, slot))
}

// prune removes spent entries left by this and by earlier runners, including
// entries belonging to other models: each carries its own stamped expiry, so
// pruning never cuts an entry's life short.
func (p *promptCacheStore) prune(now time.Time) {
	entries, err := os.ReadDir(p.dir)
	if err != nil {
		slog.Debug("prompt cache prune failed", "dir", p.dir, "error", err)
		return
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), promptCacheFilePrefix) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if strings.HasSuffix(entry.Name(), promptCacheTempSuffix) {
			// A scratch entry is only stamped once its save has finished, so
			// age is all there is to go on: one younger than a save's timeout
			// may still be being written.
			if now.Sub(info.ModTime()) < promptCacheRequestTimeout {
				continue
			}
		} else if !promptCacheExpired(info.ModTime(), now) {
			continue
		}
		name := filepath.Join(p.dir, entry.Name())
		if err := os.Remove(name); err != nil && !os.IsNotExist(err) {
			slog.Debug("prompt cache entry could not be removed", "file", name, "error", err)
			continue
		}
		slog.Debug("removed expired prompt cache entry", "file", name)
	}
}

// PromptCacheIdleUnload reports how long this runner may sit idle before it is
// worth unloading it for its memory. It is zero unless the on-disk cache is on,
// because without it an unload throws the prefill away.
func (s *llamaServerRunner) PromptCacheIdleUnload() time.Duration {
	if s.promptCache == nil {
		return 0
	}
	return envconfig.PromptCacheIdleUnload()
}

// flushPromptCache waits for a save that is still in flight, so an unload does
// not kill llama-server in the middle of writing an entry. The write itself is
// already crash-safe (it lands on a scratch file that is renamed into place),
// so this only decides whether the newest state makes it to disk at all.
func (s *llamaServerRunner) flushPromptCache() {
	p := s.promptCache
	if p == nil {
		return
	}

	deadline := time.Now().Add(promptCacheFlushTimeout)
	for slot := range p.slots {
		for p.saving[slot].Load() {
			if time.Now().After(deadline) {
				slog.Debug("prompt cache save still in flight at unload, giving up on it", "slot", slot)
				return
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
}

// promptCacheSlotState is the part of llama-server's GET /slots reply that says
// whether a slot holds something worth writing out, and whether it has moved
// since the last time it was.
type promptCacheSlotState struct {
	ID            int  `json:"id"`
	IDTask        int  `json:"id_task"`
	NPromptTokens int  `json:"n_prompt_tokens"`
	IsProcessing  bool `json:"is_processing"`
}

// fingerprint identifies the state a slot is holding. The task id changes with
// every request a slot serves, so a slot that has served something new since
// its last save never looks unchanged.
func (st promptCacheSlotState) fingerprint() string {
	return fmt.Sprintf("%d:%d", st.IDTask, st.NPromptTokens)
}

// claimMirrored reports whether a slot holding fp still needs to be written
// out, and records fp as mirrored when it does.
func (p *promptCacheStore) claimMirrored(slot int, fp string) bool {
	p.mirroredMu.Lock()
	defer p.mirroredMu.Unlock()
	if p.mirrored[slot] == fp {
		return false
	}
	p.mirrored[slot] = fp
	return true
}

// releaseMirrored forgets a slot's recorded state, so a failed save is retried
// after the next request rather than being taken for done.
func (p *promptCacheStore) releaseMirrored(slot int) {
	p.mirroredMu.Lock()
	defer p.mirroredMu.Unlock()
	if slot < len(p.mirrored) {
		p.mirrored[slot] = ""
	}
}

// slotStates asks llama-server what its slots are holding.
func (s *llamaServerRunner) slotStates(ctx context.Context) ([]promptCacheSlotState, error) {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/slots", s.port)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	res, err := s.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	payload, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= 400 {
		return nil, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(payload)}
	}

	var out []promptCacheSlotState
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil, fmt.Errorf("error unmarshalling slots response: %w", err)
	}
	return out, nil
}

// savePromptCacheChanged mirrors every slot whose state has moved since it was
// last written out. Only the native completion response says which slot served
// a request; on the chat path the slots are asked instead, which also covers a
// slot that llama-server filled behind Ollama's back.
func (s *llamaServerRunner) savePromptCacheChanged() {
	p := s.promptCache
	if p == nil || p.keepAlive() <= 0 {
		return
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), promptCacheStatusTimeout)
		states, err := s.slotStates(ctx)
		cancel()
		if err != nil {
			slog.Debug("prompt cache could not read slot states", "error", err)
			return
		}

		for _, st := range states {
			if st.IsProcessing || st.NPromptTokens <= 0 || st.ID < 0 || st.ID >= p.slots {
				continue
			}
			if !p.claimMirrored(st.ID, st.fingerprint()) {
				continue
			}
			s.savePromptCache(st.ID)
		}
	}()
}

// SetPromptCacheTTL records the effective keep-alive for this runner, which is
// also how long its on-disk prompt cache entries stay valid. The scheduler
// calls it as requests arrive; it is a no-op when the on-disk cache is off.
func (s *llamaServerRunner) SetPromptCacheTTL(d time.Duration) {
	s.promptCache.setKeepAlive(d)
}

type promptCacheSlotResponse struct {
	IDSlot    int    `json:"id_slot"`
	Filename  string `json:"filename"`
	NSaved    int    `json:"n_saved"`
	NRestored int    `json:"n_restored"`
	NWritten  int64  `json:"n_written"`
	NRead     int64  `json:"n_read"`
	Timings   struct {
		SaveMS    float64 `json:"save_ms"`
		RestoreMS float64 `json:"restore_ms"`
	} `json:"timings"`
}

// slotAction drives llama-server's POST /slots/{id}?action=... endpoint.
func (s *llamaServerRunner) slotAction(ctx context.Context, slot int, action, filename string) (promptCacheSlotResponse, error) {
	var out promptCacheSlotResponse

	body, err := json.Marshal(map[string]string{"filename": filename})
	if err != nil {
		return out, err
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=%s", s.port, slot, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return out, err
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(req)
	if err != nil {
		return out, err
	}
	defer res.Body.Close()

	payload, err := io.ReadAll(res.Body)
	if err != nil {
		return out, err
	}
	if res.StatusCode >= 400 {
		return out, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(payload)}
	}
	if err := json.Unmarshal(payload, &out); err != nil {
		return out, fmt.Errorf("error unmarshalling slot %s response: %w", action, err)
	}
	return out, nil
}

// savePromptCache mirrors a slot to disk after a completion. It runs in the
// background: llama-server defers the save until the slot is free again, and
// the caller should not wait on a multi-hundred-MiB write.
func (s *llamaServerRunner) savePromptCache(slot int) {
	p := s.promptCache
	if p == nil || slot < 0 || slot >= p.slots {
		return
	}
	if p.keepAlive() <= 0 {
		// The model is not meant to stay warm, so neither is its prefix.
		return
	}
	// A save in flight is left to finish, but the slot is marked dirty so the
	// state it misses is written once it returns: the entry has to end up
	// holding the last completion, not an arbitrary earlier one.
	p.dirty[slot].Store(true)
	if !p.saving[slot].CompareAndSwap(false, true) {
		slog.Debug("prompt cache save already in flight, will repeat when it finishes", "slot", slot)
		return
	}

	go func() {
		defer p.saving[slot].Store(false)

		for p.dirty[slot].CompareAndSwap(true, false) {
			ctx, cancel := context.WithTimeout(context.Background(), promptCacheRequestTimeout)
			start := time.Now()
			// The write goes to a scratch entry that is renamed over the live
			// one once it is complete. Killing the process is the case this
			// cache exists for, and a kill during a write would otherwise
			// leave a truncated entry where a good one used to be.
			tmp := filepath.Join(p.dir, promptCacheTempFilename(p.key, slot))
			res, err := s.slotAction(ctx, slot, "save", promptCacheTempFilename(p.key, slot))
			cancel()
			if err != nil {
				slog.Debug("prompt cache save failed", "slot", slot, "error", err)
				p.releaseMirrored(slot)
				if err := os.Remove(tmp); err != nil && !os.IsNotExist(err) {
					slog.Debug("prompt cache scratch entry could not be removed", "file", tmp, "error", err)
				}
				return
			}

			// Stamp the entry with when it stops being valid, so a later load
			// under a shorter keep-alive cannot cut it short. Stamping before
			// the rename keeps an entry from ever being visible unstamped,
			// which a concurrent prune would read as expired.
			expiry := promptCacheExpiry(time.Now(), p.keepAlive())
			if err := os.Chtimes(tmp, expiry, expiry); err != nil {
				slog.Debug("prompt cache entry could not be stamped", "slot", slot, "error", err)
			}

			if err := os.Rename(tmp, p.path(slot)); err != nil {
				slog.Debug("prompt cache entry could not be published", "slot", slot, "error", err)
				if err := os.Remove(tmp); err != nil && !os.IsNotExist(err) {
					slog.Debug("prompt cache scratch entry could not be removed", "file", tmp, "error", err)
				}
				return
			}

			slog.Debug("saved prompt cache to disk",
				"slot", slot,
				"tokens", res.NSaved,
				"bytes", res.NWritten,
				"expires", expiry,
				"duration", time.Since(start),
			)
		}
	}()
}

// restorePromptCache loads any unexpired entries back into their slots. It is
// called once, after llama-server reports ready and before the first request,
// because a restore is refused while a slot is processing.
func (s *llamaServerRunner) restorePromptCache(ctx context.Context) {
	p := s.promptCache
	if p == nil {
		return
	}
	if once := p.restored.Load(); once != nil {
		once.Do(func() { s.restorePromptCacheOnce(ctx) })
	}
}

func (s *llamaServerRunner) restorePromptCacheOnce(ctx context.Context) {
	p := s.promptCache

	now := time.Now()

	for slot := range p.slots {
		name := p.path(slot)
		info, err := os.Stat(name)
		if err != nil {
			continue
		}
		if promptCacheExpired(info.ModTime(), now) {
			slog.Debug("ignoring expired prompt cache entry", "file", name, "expired", info.ModTime())
			continue
		}

		start := time.Now()
		reqCtx, cancel := context.WithTimeout(ctx, promptCacheRequestTimeout)
		res, err := s.slotAction(reqCtx, slot, "restore", filepath.Base(name))
		cancel()
		if err != nil {
			// Only discard an entry llama-server actually rejected - a state
			// written by an incompatible build will never load, so retrying it
			// on every start is waste. A transport error or a cancelled load
			// says nothing about the entry, and must leave it alone.
			var status api.StatusError
			if errors.As(err, &status) && status.StatusCode < 500 {
				slog.Info("prompt cache entry was rejected, discarding", "file", name, "error", err)
				if err := os.Remove(name); err != nil && !os.IsNotExist(err) {
					slog.Debug("prompt cache entry could not be removed", "file", name, "error", err)
				}
				continue
			}
			slog.Debug("prompt cache entry could not be restored", "file", name, "error", err)
			continue
		}

		slog.Info("restored prompt cache from disk",
			"slot", slot,
			"tokens", res.NRestored,
			"bytes", res.NRead,
			"duration", time.Since(start),
		)
	}

	p.prune(now)
}
