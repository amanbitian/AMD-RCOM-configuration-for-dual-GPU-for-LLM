# Changelog

Notable changes to this repo, in reverse-chronological order. See [CLIENT_API.md](CLIENT_API.md)
for a walkthrough of the client-integration flow and [API_CONTRACT.md](API_CONTRACT.md) for the
exhaustive field-by-field request/response reference of every endpoint. For the 2026-09-27
gap-fixing work specifically, [optimised.md](optimised.md) has the narrative version — how
each bug below was actually found and confirmed, not just what changed.

## 2026-09-27

### Added — KV-cache-aware VRAM feasibility check for deploy

The deploy "final verdict" (`/api/deploy`, `/api/clients/deploy`) only ever checked model
*file size* against lane VRAM capacity — never the context window actually being requested,
including one grown by `input_tokens`/`max_output_tokens` (see the sizing-guarantee entry
below). A deploy could return a clean `200` and still OOM or silently spill to shared memory
once `llama-server` actually allocated a large KV cache. Also newly discovered while fixing
this: `single_large_model` deploys never ran *any* capacity check at all, file-size or
otherwise.

Fixed by reading each model's own GGUF metadata (new [dual_gpu_setup/gguf.py](dual_gpu_setup/gguf.py),
a dependency-free, tensor-data-skipping header reader) and estimating real KV-cache VRAM cost
with the standard per-layer transformer formula (`estimate_kv_cache_gb` in
[dual_gpu_setup/orchestrator.py](dual_gpu_setup/orchestrator.py)). `DeploymentManager._validate_capacity`
(replacing `_validate_single_fit`) now checks `model_file_gb + kv_cache_gb` against lane
capacity for all three deploy modes, including `single_large_model`.

This was validated against real GGUF files on disk, not just synthetic ones, before shipping:
- A genuine head-dimension bug was caught this way — deriving `head_dim` from
  `embedding_length / head_count` gave a non-integer (213.33) for a real Qwen3.5 model file.
  Fixed by preferring the GGUF's own explicit `attention.key_length`/`value_length` fields,
  falling back to the derived value only when those are absent and the division is exact.
- The live-deployed Qwen3.8-27B model turned out to be a hybrid state-space/attention
  architecture (`*.ssm.*` GGUF keys present) where most layers don't have a conventional KV
  cache at all. The estimator detects this and returns "not confidently estimable" rather than
  a wrong number — the pre-existing file-size-only check still applies for those, unchanged.
- Real Gemma GGUF files (`gemma3`, `gemma4`) confirmed a sliding-window attention pattern:
  `gemma4` exposes an explicit per-layer `attention.sliding_window_pattern` array, which the
  estimator uses to cap those layers' KV cost at the sliding window instead of full context
  (confirmed smaller than the naive full-context estimate on the real file: 1.37 GB vs. what a
  no-sliding-window assumption would have given). `gemma3` exposes only a scalar
  `sliding_window` with no per-layer pattern — the local/global ratio is then unknown, so the
  estimator falls back to treating every layer as full context (an over-estimate, the safe
  direction for a feasibility gate, never an under-estimate).

Retraction: an earlier message in this conversation claimed `policy.parallel` (concurrent
request slots) was also missing from this math and listed it as a second confirmed gap. Before
writing any code for it, `llama-server --help` was checked directly on this machine:
`--ctx-size` is documented as the shared/total KV pool size, not multiplied per slot by
default. Since this repo always passes an explicit `parallel` value (default `1`, never
`-1`/auto) and no current model config uses `parallel > 1`, there was no way to confirm an
actual bug, so no fix was written for it — it was dropped rather than build something on an
unverified premise.

Files: [dual_gpu_setup/gguf.py](dual_gpu_setup/gguf.py) (new),
[dual_gpu_setup/orchestrator.py](dual_gpu_setup/orchestrator.py) (`estimate_kv_cache_gb`),
[app.py](app.py) (`DeploymentManager._validate_capacity`), [tests/](tests/) (new, see below).

### Fixed — config loader crashed on any TOML that omitted an optional field

`load_config` used `PolicyConfig.some_field` (and the same for `ProjectConfig`/
`DiscoveryConfig`) as the fallback default in `policy_raw.get("some_field", PolicyConfig.some_field)`.
For a `@dataclass(slots=True)`, accessing a field on the *class* returns the slot descriptor
object, not the default value — only *instance* access (`PolicyConfig().some_field`) does. This
never showed up before because every real TOML in this repo happened to specify every field
explicitly; it surfaced while writing a minimal test config that omitted most of `[policy]`,
which crashed with `TypeError: ... not a 'member_descriptor'` instead of just using the
documented default. Fixed by building one default instance per config section
(`ProjectConfig()`, `DiscoveryConfig()`, `PolicyConfig()`) and reading fallbacks from those.

Files: [dual_gpu_setup/config.py](dual_gpu_setup/config.py), [tests/test_config.py](tests/test_config.py).

### Added — Automated test suite

There was no automated test coverage at all in this repo before this. Added `tests/` (pytest,
37 tests, all passing) covering: the GGUF metadata reader (round-trips scalars/strings/arrays,
rejects bad magic and truncated files, `wanted_keys` skip-mode matches a full parse), the KV-
cache estimator and `resolve_context_window`/`ctx_for` (including the exact dense-model math,
the head-dim bug above as a regression test, the SSM-hybrid and sliding-window cases), the
config loader (including the slots-default bug above as a regression test), `RunStore`
migrations/client registration/`categories`/`category_models` aggregation/pruning, and
`DeploymentManager._validate_capacity` end-to-end across all three deploy modes using synthetic
GGUF fixtures (a `write_gguf` test helper in [tests/conftest.py](tests/conftest.py) — so the
suite never depends on any real model file being present on disk). `pytest` added as a
`dev` optional dependency in [pyproject.toml](pyproject.toml); run with `pytest` from the repo
root (basetemp is pinned into the repo via `addopts` since this environment's default system
temp dir hit a permissions error).

Files: [tests/](tests/) (new), [pyproject.toml](pyproject.toml), [.gitignore](.gitignore).

### Added — Optional data retention

There was no way to bound the run database's growth — `metric_samples` gets a row every ~2s
per active server for the lifetime of every run, forever, with no pruning anywhere. Default
behavior is unchanged (keep everything forever, matching this project's stated priority of
retaining as much telemetry as possible) but an explicit opt-in is now available:
`RunStore.prune_older_than(days)` deletes `runs` (and, via the existing `ON DELETE CASCADE`
foreign key, their `metric_samples`) and `deployments` older than the given number of days.
Wired up as `python app.py --prune-older-than-days N`, which runs once at startup (compatible
with `--check` for a pure maintenance invocation) and then continues normally.

Files: [app.py](app.py) (`RunStore.prune_older_than`, CLI flag), [tests/test_runstore.py](tests/test_runstore.py).

### Fixed — `task_label` normalization

`task_label` (see the Analytics entry below) was stored exactly as given, so
`"Resume_Extraction"` and `"resume_extraction"` would silently appear as two different rows on
the Analytics tab. Now lowercased and stripped before storage.

Files: [app.py](app.py) (`ChatService.chat`).

### Added — Task-category analytics (dashboard Analytics tab)

`/api/chat` accepts an optional `task_label` (freeform string, e.g. `"relevance"`,
`"resume_extraction"`, `"fraud_d2_fusion"`) to tag what kind of work a call was. Stored on the
run alongside `client_id`, and rolled up in `/api/dashboard` as two new fields:

- `categories` — same metrics as the existing `models` breakdown (runs, success rate, avg
  decode/prefill tok/s, avg latency, peak VRAM), grouped by `task_label` instead of model.
- `category_models` — one row per (category, model) pair actually seen, so results can be
  compared across models within the same task type.

The dashboard gained a new **Analytics** nav tab ([dashboard.html](dashboard.html)) with two
dropdown filters (category, model) over a breakdown table fed by `category_models`, plus a
weighted-average KPI strip for whatever's currently filtered. A dedicated tab was chosen over
adding filters to the existing Overview tab, which was already dense (5 KPIs, model bars,
token donut, activity chart, resource peaks, recent runs). Runs made without a `task_label`
are excluded from the two new breakdowns but keep counting normally in `models`/`summary`, so
nothing already working changes.

New `task_label` column on `runs` (additive migration, backward-compatible). The
`renderAnalytics()` filtering/weighted-average logic was verified in isolation with Node
before shipping (filtering, cross-tab math, and the empty-state message all checked against
synthetic data).

Files: [app.py](app.py) (`RunStore.dashboard`, `ChatService.chat`/`_run_one`),
[dashboard.html](dashboard.html), [CLIENT_API.md](CLIENT_API.md), [API_CONTRACT.md](API_CONTRACT.md).

### Added — Context-window sizing guarantee for deploy

`/api/deploy` and `/api/clients/deploy` accept optional `input_tokens` + `max_output_tokens`
(required together). When given, the resolved `context_window` is grown to guarantee
`input_tokens + max_output_tokens + policy.safety_tokens <= context_window`, even above
whatever `context_window` was requested or the policy default would have picked. If that
requirement exceeds `policy.ctx_max`, the deploy is rejected with `400` up front instead of
silently loading a server that fails requests mid-run with `exceed_context_size_error` — the
exact failure hit running resume extraction (~19k-token prompts) against a 8,192-token
deployment earlier the same day. New `policy.safety_tokens` config field (default `256`).

Files: [dual_gpu_setup/config.py](dual_gpu_setup/config.py),
[dual_gpu_setup/orchestrator.py](dual_gpu_setup/orchestrator.py) (`resolve_context_window`),
[app.py](app.py), [CLIENT_API.md](CLIENT_API.md), [API_CONTRACT.md](API_CONTRACT.md).

### Added — On-demand service launcher

Previously a client could only use the Client Integration API once `app.py` was already
running — there was no way to trigger the service to start, and no HTTP endpoint could ever
fill that gap (a stopped service has nothing listening to receive a "start" request).
[launch_service.py](launch_service.py) closes it from outside the HTTP layer: a one-shot
command (not an always-on daemon) that a client runs before it starts calling the API. It
probes `--host:--port`, exits immediately if the service is already healthy (idempotent), and
otherwise spawns `app.py` as a detached background process, waits for it to report healthy,
prints its `base_url`, and exits — the service keeps running after the launcher exits; the
launcher itself does not stay resident. Verified end-to-end: cold start, idempotent re-run
against an already-running instance, and confirmation that the launcher process exits while
`app.py` survives independently.

### Added — Formal API contract

[API_CONTRACT.md](API_CONTRACT.md) documents every HTTP endpoint `app.py` exposes (not just
the client-facing ones): for each one, exactly which fields are required vs. optional on the
request, their types/defaults/validation rules, the full response shape, and every error
status the endpoint can return. This is the exhaustive reference; [CLIENT_API.md](CLIENT_API.md)
remains the narrative walkthrough for the client-integration flow specifically.

### Added — Client Integration API

Other local codebases can now drive the orchestrator over HTTP instead of only through the
dashboard UI, while the UI keeps working exactly as before.

- `POST /api/clients/register` — a consuming repo registers `project_name`, optional
  `github_repo`, and an `output_path` where its own copy of run metrics gets written.
- `POST /api/clients/deploy` — pass `model`, optional `gpu`, optional `context_window`. If `gpu`
  is omitted, the service picks the smallest lane the model fits (or its pinned lane) and
  deploys through the same fit/pin/capacity validation the dashboard's "Load deployment" button
  uses, so the feasibility verdict is authoritative rather than guessed client-side.
- `POST /api/chat` now accepts an optional `client_id`. The response is unchanged, plus a new
  `client_delivery` field confirming a JSON-line record was appended to that client's
  `output_path`. A write failure there never fails the chat call itself.
- `GET /api/clients/{id}/runs` — a client can re-fetch its own run history from this repo's
  store.
- `GET /api/clients/schema` — machine-readable list of every metric field shared with clients.
- New `clients` SQLite table and a `client_id` column on `runs` (both additive migrations,
  backward-compatible with existing databases).

Files: [app.py](app.py) (`RunStore`, `DeploymentManager.deploy_for_client`,
`ChatService._deliver_to_client`, new HTTP routes), [CLIENT_API.md](CLIENT_API.md) (new).

### Fixed / Optimized — GPU and CPU telemetry collection

The resource sampler that tracks per-model GPU/CPU/RAM usage had two real problems, both
confirmed by direct measurement rather than assumed:

- **Redundant process spawns.** `gpu_process_metrics()` spawned a new PowerShell process (with
  three separate `Get-Counter` queries) *per active GPU server*, called by the background
  sampler every 2 seconds and twice more per chat request (baseline + final capture). Measured
  cost: ~4.05s for the 3 separate `Get-Counter` calls per server — with two active GPUs, the
  sampler could not keep up with its own 2-second interval.
  - Collapsed to one PowerShell call per sampling tick covering every active server's PID
    (`gpu_process_metrics_batch` in [dual_gpu_setup/server.py](dual_gpu_setup/server.py)),
    instead of one call per server.
  - Combined the three `Get-Counter` queries into a single call
    (`Get-Counter -Counter @(...)`), which roughly halves the remaining PDH cold-start cost
    again (measured: 4.05s → 2.09s).
  - Net effect measured end-to-end: two servers' worth of GPU stats dropped from ~8.7s to
    ~2.3s total.
- **PID cross-matching.** The instance filter `-like 'pid_123*'` had no boundary after the
  number, so pid `123` could match GPU counters belonging to pid `1234`, `12300`, etc.
  (confirmed in real PowerShell: `'pid_1234_luid...' -like 'pid_123*'` → `True`). Anchored the
  pattern to `pid_{pid}_*`, which correctly excludes cross-matches while still matching the
  real pid.

### Added — More retained telemetry

Additional fields that were already being fetched from Windows APIs but discarded, now
surfaced (all additive — no existing field, table column, or dashboard behavior changed):

- Per-process: `peak_rss_bytes`, `peak_private_bytes`, `page_fault_count` per sample.
- Per-run rollup (`resources_json`, also what clients receive via the Client Integration API):
  `peak_process_private_bytes`, `process_page_fault_delta` (faults incurred *during that run*,
  not the lifetime counter), `peak_system_pagefile_used_bytes`.
- System-wide: `pagefile_total_bytes`.

Files: [app.py](app.py) (`ResourceSampler._process_metrics`, `._system_metrics`,
`ChatService._resource_summary`), [dual_gpu_setup/server.py](dual_gpu_setup/server.py).
