package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func testLaunchConfig() llamaServerLaunchConfig {
	opts := api.DefaultOptions()
	opts.NumCtx = 8192
	return llamaServerLaunchConfig{
		modelPath:   "/models/blobs/sha256-abc",
		modelArch:   "qwen35",
		opts:        opts,
		numParallel: 1,
		kvCacheType: "f16",
	}
}

func TestPromptCacheIdentity(t *testing.T) {
	base := testLaunchConfig()

	if got, want := promptCacheIdentity(base), promptCacheIdentity(testLaunchConfig()); got != want {
		t.Errorf("identity is not stable: %q != %q", got, want)
	}

	if len(promptCacheIdentity(base)) != 16 {
		t.Errorf("identity length = %d, want 16", len(promptCacheIdentity(base)))
	}

	cases := map[string]func(*llamaServerLaunchConfig){
		"model":       func(l *llamaServerLaunchConfig) { l.modelPath = "/models/blobs/sha256-def" },
		"context":     func(l *llamaServerLaunchConfig) { l.opts.NumCtx = 16384 },
		"parallel":    func(l *llamaServerLaunchConfig) { l.numParallel = 4 },
		"kv type":     func(l *llamaServerLaunchConfig) { l.kvCacheType = "q8_0" },
		"adapter":     func(l *llamaServerLaunchConfig) { l.adapters = []string{"/adapters/a"} },
		"projector":   func(l *llamaServerLaunchConfig) { l.projectors = []string{"/mmproj"} },
		"draft model": func(l *llamaServerLaunchConfig) { l.config.DraftModelPath = "/models/draft" },
	}

	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			changed := testLaunchConfig()
			mutate(&changed)
			if promptCacheIdentity(changed) == promptCacheIdentity(base) {
				t.Errorf("changing %s did not change the identity", name)
			}
		})
	}
}

func TestPromptCacheFilename(t *testing.T) {
	name := promptCacheFilename("0123456789abcdef", 3)

	if !strings.HasPrefix(name, promptCacheFilePrefix) {
		t.Errorf("filename %q does not carry the prune prefix", name)
	}

	// llama-server rejects anything that is not a plain basename, and joins the
	// name onto --slot-save-path without sanitizing it.
	if filepath.Base(name) != name || strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		t.Errorf("filename %q is not a plain basename", name)
	}

	if other := promptCacheFilename("0123456789abcdef", 4); other == name {
		t.Error("slots share a filename")
	}
}

func TestPromptCacheSlotSavePath(t *testing.T) {
	sep := string(os.PathSeparator)

	for _, dir := range []string{"/tmp/cache", "/tmp/cache/", `/tmp/cache\`} {
		got := promptCacheSlotSavePath(dir)
		if !strings.HasSuffix(got, sep) {
			t.Errorf("promptCacheSlotSavePath(%q) = %q, want a trailing separator", dir, got)
		}
		if strings.HasSuffix(strings.TrimSuffix(got, sep), sep) {
			t.Errorf("promptCacheSlotSavePath(%q) = %q, want exactly one trailing separator", dir, got)
		}
	}
}

func TestPromptCacheExpiry(t *testing.T) {
	written := time.Now().Add(-10 * time.Minute)

	cases := []struct {
		name string
		ttl  time.Duration
		want bool // expired ten minutes after it was written
	}{
		{"within ttl", time.Hour, false},
		{"past ttl", 5 * time.Minute, true},
		{"zero keep alive", 0, true},
		{"negative", -time.Second, true},
		{"forever is capped", time.Duration(math.MaxInt64), false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			expiry := promptCacheExpiry(written, tc.ttl)
			if got := promptCacheExpired(expiry, written.Add(10*time.Minute)); got != tc.want {
				t.Errorf("entry written with ttl %v: expired = %v, want %v", tc.ttl, got, tc.want)
			}
			if expiry.After(written.Add(promptCacheMaxTTL)) {
				t.Errorf("expiry %v exceeds the cap of %v", expiry, promptCacheMaxTTL)
			}
		})
	}
}

func TestPromptCachePruneKeepsUnexpiredAndForeignFiles(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()

	// Entries carry their expiry as the file's modification time.
	write := func(name string, expiredFor time.Duration) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("state"), 0o644); err != nil {
			t.Fatal(err)
		}
		mod := now.Add(-expiredFor)
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatal(err)
		}
		return path
	}

	fresh := write(promptCacheFilename("aaaaaaaaaaaaaaaa", 0), -time.Hour)
	stale := write(promptCacheFilename("aaaaaaaaaaaaaaaa", 1), time.Minute)
	otherModel := write(promptCacheFilename("bbbbbbbbbbbbbbbb", 0), -24*time.Hour)
	foreign := write("some-other-file.bin", time.Hour)

	store := &promptCacheStore{dir: dir, key: "aaaaaaaaaaaaaaaa", slots: 1, saving: make([]atomic.Bool, 1), dirty: make([]atomic.Bool, 1)}
	store.setKeepAlive(time.Minute)
	store.prune(now)

	if _, err := os.Stat(fresh); err != nil {
		t.Errorf("entry that has not reached its expiry was removed: %v", err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("expired entry survived: %v", err)
	}
	if _, err := os.Stat(otherModel); err != nil {
		t.Errorf("another model's entry was removed before its own expiry: %v", err)
	}
	if _, err := os.Stat(foreign); err != nil {
		t.Errorf("unrelated file was removed: %v", err)
	}
}

func TestPromptCacheNilStore(t *testing.T) {
	var store *promptCacheStore
	if got := store.keepAlive(); got != 0 {
		t.Errorf("keepAlive on nil store = %v, want 0", got)
	}
	store.setKeepAlive(time.Hour)

	// A runner with the on-disk cache disabled must stay inert.
	s := &llamaServerRunner{}
	s.SetPromptCacheTTL(time.Hour)
	s.savePromptCache(0)
	s.restorePromptCache(t.Context())
}

func TestRestorePromptCacheReadsOncePerProcessStart(t *testing.T) {
	var restores atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		restores.Add(1)
		fmt.Fprint(w, `{"id_slot":0,"n_restored":42,"n_read":128}`)
	}))
	defer srv.Close()

	port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	store := &promptCacheStore{dir: dir, key: "aaaaaaaaaaaaaaaa", slots: 1, saving: make([]atomic.Bool, 1), dirty: make([]atomic.Bool, 1)}
	store.setKeepAlive(time.Hour)
	store.armRestore()

	entry := store.path(0)
	if err := os.WriteFile(entry, []byte("state"), 0o644); err != nil {
		t.Fatal(err)
	}
	expiry := time.Now().Add(time.Hour)
	if err := os.Chtimes(entry, expiry, expiry); err != nil {
		t.Fatal(err)
	}

	s := &llamaServerRunner{port: port, promptCache: store, client: srv.Client()}

	// The scheduler waits for the runner twice per start.
	s.restorePromptCache(t.Context())
	s.restorePromptCache(t.Context())
	if got := restores.Load(); got != 1 {
		t.Errorf("restores after one start = %d, want 1", got)
	}

	// A process restart brings up empty slots, so the entry is read again.
	store.armRestore()
	s.restorePromptCache(t.Context())
	if got := restores.Load(); got != 2 {
		t.Errorf("restores after a restart = %d, want 2", got)
	}
}

func TestRestorePromptCacheDiscardsOnlyRejectedEntries(t *testing.T) {
	cases := []struct {
		name    string
		handler http.HandlerFunc
		kept    bool
	}{
		{
			name: "restored",
			handler: func(w http.ResponseWriter, r *http.Request) {
				if got := r.URL.Query().Get("action"); got != "restore" {
					t.Errorf("action = %q, want restore", got)
				}
				var body map[string]string
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if !strings.HasPrefix(body["filename"], promptCacheFilePrefix) {
					t.Errorf("filename = %q, want a prompt cache entry name", body["filename"])
				}
				fmt.Fprint(w, `{"id_slot":0,"n_restored":42,"n_read":128}`)
			},
			kept: true,
		},
		{
			name: "rejected by llama-server",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprint(w, `{"error":{"message":"Unable to restore slot: invalid slot save file"}}`)
			},
			kept: false,
		},
		{
			name: "server error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				fmt.Fprint(w, `{"error":{"message":"busy"}}`)
			},
			kept: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()

			port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
			if err != nil {
				t.Fatal(err)
			}

			dir := t.TempDir()
			store := &promptCacheStore{dir: dir, key: "aaaaaaaaaaaaaaaa", slots: 1, saving: make([]atomic.Bool, 1), dirty: make([]atomic.Bool, 1)}
			store.setKeepAlive(time.Hour)
			store.armRestore()

			entry := store.path(0)
			if err := os.WriteFile(entry, []byte("state"), 0o644); err != nil {
				t.Fatal(err)
			}
			expiry := time.Now().Add(time.Hour)
			if err := os.Chtimes(entry, expiry, expiry); err != nil {
				t.Fatal(err)
			}

			s := &llamaServerRunner{port: port, promptCache: store, client: srv.Client()}
			s.restorePromptCache(t.Context())

			_, err = os.Stat(entry)
			if tc.kept && err != nil {
				t.Errorf("entry was removed: %v", err)
			}
			if !tc.kept && !os.IsNotExist(err) {
				t.Errorf("rejected entry survived: %v", err)
			}
		})
	}
}
