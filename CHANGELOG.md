# Changelog

## 2026-10-17 — Measured: where the context cache lives

Documentation only; no code changed. With Qwen3-8B at a 262,144-token context on the R9700 (a 36.9 GB cache on
a 32 GB card), the default silently spilled 11.4 GB into system RAM and decoded at 77 tok/s at a 2k prompt and
32 tok/s at 16k. `--no-kv-offload` put the cache in RAM (24 tok/s at 2k, 7 tok/s at 16k). `--cache-type-k q8_0
--cache-type-v q8_0` kept it on the GPU at 85 and 72 tok/s. `--fit on` changed nothing because context and layers
are set explicitly. The repo has no deploy parameter for this and its capacity check counts the cache at 16-bit
size in VRAM, so large contexts are refused. The table and the known issue are in
[app_overview.md](app_overview.md#where-the-context-cache-lives-kv-cache).

## 2026-10-17 — app_overview.md

New [app_overview.md](app_overview.md): the single description of what the app is and is not, where it is
verified, its architecture, every feature, the HTTP API map, the configuration reference, the files it writes,
measured speed, time and memory on the reference machine, the rules for applications that use it, known
issues, security posture and test status. It states what was measured or tested and marks the rest not
verified. README and API_CONTRACT now point to it, and it says to be updated in the same change as any
behaviour change.

## 2026-10-17 — Control the service from the UI

**Clients release the GPU when their task is done.** New contract clause (API_CONTRACT.md, "Releasing the
GPUs when a task is done"; also CLIENT_API.md step 7, CODING_AGENTS.md and the README): an application that
loaded a model must unload it when its task finishes, on success and on failure. The service does not do
this itself, and has no idle timeout. To make that safe on a shared service, `POST /api/unload` accepts
`{"lane": ..., "model": ...}`: it stops the lane's model only if `model` is still what serves it, so a
client cannot stop a model another application loaded there since. Lane-scoped unloads now reply
`released: true/false`. `model` without `lane` is a `400`.

**Unload while a model is answering.** Measured on a model writing a long answer: Unload said nothing for
30.8 s (the drain timeout), then cut the answer. Nothing in the UI showed a request was running, an
Unload error appeared only on the Chat page, and there was no way to stop sooner. Now each server reports
`inflight`, and a bar on every page says "Waiting for N running request(s) to finish (up to 30 s)" with
a **Stop now** button (`POST /api/unload {"force": true}`). The same scenario now ends in 3.8 s when Stop
now is pressed after 3 s. Cancelling one answer while keeping the model loaded is not built.

`launch_service.py` started the service but never opened it: it always passed `--no-browser`, which
suits a program calling it. It now opens the UI when run from a terminal (also when the service is
already running) and stays silent when no terminal is attached; `--open-browser` / `--no-browser`
force either.

New **Service** page in the dashboard, backed by new endpoints (see API_CONTRACT.md, "Service control"):
- **Restart / stop the service** (`POST /api/service/restart`, `/stop`). A restart stops the models,
  starts a replacement process that waits for this one to exit before binding the port
  (`--replace-pid`), and can load the same models again. The deployment is saved with its
  coding-agent mode, and a saved deployment older than 10 minutes is ignored. The page reconnects by
  itself.
- **Settings** (`GET`/`POST /api/settings`): the loading, memory and residency options in `[policy]`,
  validated, applied to the running service and saved into the config file with its comments intact
  (the first save keeps `<config>.bak`). The residency guard now reads its setting on every tick, so
  it can be switched live.
- **Unload one GPU** (`POST /api/unload {"lane": ...}`), or only the leftovers (`{"orphans_only": true}`).
- **Orphaned servers.** A llama-server left on a lane port by an instance that was killed is now reported
  (`orphans` in `/api/status`, a banner in the UI) and stopped by Unload. Two things were wrong without
  this: the new service did not know the leftover existed, so it kept holding VRAM and RAM, and its VRAM
  made the free-VRAM check refuse a load that was fine once it was gone. A load now stops such a leftover
  first (`kill_busy_ports`).
- The state-changing control endpoints require a JSON content type (`415` otherwise), so a web page cannot
  trigger a restart or stop with a blind form POST.

The test suite can no longer see or stop real servers. Its example config uses the real lane ports, and
the restart test's `unload()` found and stopped a llama-server a developer had running on 8081. `conftest.py`
now hides listeners on lane ports from every test and refuses to terminate a real process. Verified with
decoy `llama-server.exe` processes on both ports: none survived before the guard, all survived the full suite after.

Found by testing the restart against the real service: a deployment loaded on demand is recorded with a
`models` list even for one GPU, which `deploy()` rejected, so the first restart did not reload it. The
saved deployment is now converted to a deployable one.

Verified on spare ports against real processes: restart with reload (new PID, old one exited, the model
loaded again), settings saved and still in force after a restart, an orphan found, listed and stopped,
a load that replaced an orphan, unloading one GPU, and a full stop that left no process behind. The new
dashboard functions were also run against a fake DOM in Node (rendering, escaping, the typed settings
payload, the reconnect after a restart). The page was not opened in a real browser here.
## 2026-10-17 — Models stay on the GPU; on-demand model manager

**Why llama-server used so much RAM.** Measured on Qwen3-8B (4.7 GB) on the R9700, with one flag
changed per run. llama.cpp itself places everything on the GPU (verbose log: 4,455 MiB weights and
2,304 MiB KV cache on ROCm0, 334 MiB token embeddings on the CPU). Two things hold host RAM anyway:
- After load, the process holds about one model's worth of RAM (4.78 GB) in every load mode. It
  matches the uploaded weights, not the KV cache, so it looks like the driver's host-side copy of
  data uploaded from RAM. With `--load-mode none` it is still committed but no longer resident
  (0.62 GB resident instead of 4.64 GB).
- llama.cpp's prompt cache (`--cache-ram`, 8 GiB default) keeps every new prompt in RAM: +4.1 GB
  after 20 distinct prompts, and +6.8 GB after 150 on gpt-oss-20b, still growing.

New `[policy]` defaults `load_mode = "none"` and `cache_ram_mib = 0` hold 0.62 GB resident and flat
after 20 requests, against 8.77 GB with the old flags. Decode speed is unchanged (96 tok/s), load is
faster (1.5 s vs 3.1 s), and a same-conversation follow-up still reprocesses only its new tokens
(from the GPU KV cache). Flags apply only to builds that have them; a model's `extra_args` win.

**Windows moves idle model VRAM into RAM.** This is the big, intermittent RAM use. A loaded Qwen3.8-27B
(16.85 GB) on the R9700 read **0 GB** resident while idle. Its next request took **10.3 s** (10.2 s
"prompt" for a 13-token prompt) as Windows copied it back into VRAM, and system free RAM rose by
about 15 GB once it was back. A second request took 0.2 s.
- Ruled out as triggers: idle time alone (it stayed resident for 3 min idle), `--list-devices`, and
  loading another model on either GPU.
- The trigger is not identified. Evictions coincided with lower free system RAM (about 15 GB vs 27 GB).
- The 9070 XT model was never evicted during the observations.

**Residency guard** (`residency_guard`, `residency_check_seconds`, default on, every 15 s). Each idle
server whose weights are no longer resident gets a 1-token request on a reserved warm-up slot. That
pages the weights back in the background instead of during the next real request.
- Servers with one slot now start with `--parallel 2 --kv-unified`. The two slots share one KV
  pool, so there is no extra KV VRAM, and each slot keeps the full context.
- Real requests are pinned to slot 0, so they still queue one at a time.
- Measured: a warm-up on slot 1 ran immediately while slot 0 was busy. A pinned second request
  queued behind the first and did not error.
- A warm-up on the only slot made the next turn reprocess 2,510 tokens. On slot 1, the next turn
  reprocessed 10.
- `/api/status` servers report `resident_gb`, `evictions` and `rewarmed_at`.

**GPU-only admission** (`vram_admission`, `vram_headroom_gb`). A server starts only when its weights,
estimated KV cache and headroom fit the VRAM actually free on its GPU(s). Windows would otherwise back
the shortfall with system RAM.
- On Windows, free VRAM is total VRAM (DXGI) minus what the OS counts as in use by every process.
- HIP's `--list-devices` figure is not used on Windows. It is per process: it reported an R9700
  holding a 16.85 GB model as 32,472 of 32,624 MiB free, and a 9070 XT with 3 GB really free as
  15.8 GB free.
- Linux keeps `--list-devices`.
- The check waits up to 20 s for a just-stopped server's VRAM to free, then refuses with 503.
- It applies to `/api/deploy`, the client deploy endpoints and on-demand loads.

**On-demand model manager** (`auto_load_models`). A gateway request whose `model` is a catalog model
is served by that model:
- If it is not loaded, it goes to a GPU where it fits the VRAM obtainable now. That is free VRAM
  plus what evicting our own idle model there releases. It prefers a free GPU (smallest first),
  else replaces the least recently used idle model.
- If it fits nowhere, the request is refused with 503 **before** anything is evicted.
- A model no single GPU can hold spans both GPUs.
- Routed requests claim their server, so a concurrent load cannot evict a model between routing
  and sending.
- A busy model is never stopped mid-request. If every candidate stays busy for the drain timeout,
  the request gets 503 and should retry.
- `dgpu:<lane>` and unknown model names route as before.
- `GET /v1/models` now also lists requestable catalog models.

Verified on R9700 + 9070 XT through a second app instance, with another app's Qwen3.8-27B and
gemma-4-12B loaded:
- Four on-demand loads answered correctly. Qwen3-8B went to the R9700 because the 9070 XT lacked
  room, and the least recently used idle model was replaced.
- Two concurrent requests for different models both succeeded.
- A 15.7 GB model was refused in 2.1 s ("r9700 needs 16.7 GB, 16.3 GB obtainable") with nothing
  evicted.
- Our servers held 0.61 to 0.64 GB of resident RAM, and unload stopped only our servers.

Not yet seen live: the guard re-warming a model. No eviction happened to our servers during the
40 s check. The re-warm path is covered by tests.

## 2026-10-17 — Official llama.cpp builds instead of LM Studio's runtime

Single-lane servers now run the official llama.cpp ROCm release b11699
(`discovery.backend_server_binaries`) instead of LM Studio's bundled fork (runtime 2.55.0, which
reports `build 1`). Compared head-to-head on gpt-oss-20b with identical flags, b11699 was faster on
both cards: prefill +3.6% to +8.2%, decode +4.1% to +7.1%. The LM Studio fallback is unchanged:
remove the config line and LM Studio's newest runtime is used again.

- **Split models stay on LM Studio** (`discovery.multi_gpu_runtime = "lmstudio"`). b11699 answers
  correctly on either card alone, but produces garbage whenever a model spans R9700 + 9070 XT.
  Tested and still broken: Qwen3-8B and Qwen3-32B; 1:1 and 2:1 splits; either device order;
  `--split-mode layer` and `tensor`; flash attention on and off; integrated GPU hidden; peer copies
  disabled. `--split-mode row` does not start. LM Studio's runtime answers correctly on the same
  splits. Re-run this check before setting `"configured"` with a newer build.
- **Build is recorded:** each server's `runtime` (binary, source, version, build, commit) appears in
  `/api/status`, every run's stored configuration, CLI `load.json` and `dual-gpu inspect`.
- **Flag compatibility:** each binary is probed once (`--version`, `--help`). Builds with `--fit` get
  `--fit off`, because this repo sizes context and offload itself. A renamed `--no-mmap` in
  `extra_args` becomes `--load-mode none` on builds that dropped it. A server that exits during
  startup now quotes its own last log lines in the error.
- A configured build no longer gets LM Studio's vendor DLL folder on `PATH`; the release zip
  bundles its own HIP runtime.

Verified on the R9700 + 9070 XT machine through `app.py`:
- single-GPU, parallel and split deployments, each answering a fixed question correctly
- `/api/chat` compare
- the coding gateway: chat, lane-pinned streaming, autocomplete and models
- run history, dashboard, LM Studio GPU, lease, hardware, project and usage APIs, and unload

Not covered by this change: the deploy capacity check still rejects gpt-oss-120b (59 GB) as too
large for 43.2 GB of VRAM, because it does not count the expert weights `--n-cpu-moe` keeps on the
CPU.

## 2026-10-13 — Pick LM Studio's GPUs over the API

`GET`/`POST /api/lmstudio/gpus` read and set which GPUs LM Studio's own llama.cpp runtime may use, by
editing `load.gpuSplitConfig.disabledGpus` in LM Studio's `hardware-config.json` (see
[API_CONTRACT.md](API_CONTRACT.md#get--post-apilmstudiogpus)). Other fields are preserved, the first
original is backed up, and an unparseable file is never overwritten. It changes the saved setting only:
already-loaded models keep their placement. `POST /api/lmstudio/release` unloads LM Studio's models
(and optionally stops its server) so the GPUs and RAM are free first; it refuses while a model is busy.
Both POSTs require `Content-Type: application/json`. Checked on the R9700 + 9070 XT machine: release,
set R9700-only, restart the server, reload — the 27B model landed entirely on the R9700.

## 2026-10-11 — Training telemetry, GPU leases, checkpoint evaluation, DDP metrics

New **Training** and **Hardware** dashboard tabs and a stdlib-only `TrainingTracker` (plus a Hugging Face
`Trainer` callback) that streams loss, learning rate, gradient norm, token accounting, step time, GPU
phase timing and memory from a training script. Metrics are per window, never per microbatch, and the
client never blocks or fails training (JSONL spool + replay when the dashboard is unreachable).

- **Host layer:** `hostos/` moves all Windows-only code (ctypes, PowerShell, netstat, taskkill) behind
  one interface and adds Linux (`/proc`, DRM fdinfo, signals). Native Linux `llama-server` builds work
  through `discovery.backend_server_binaries`.
- **GPU identity and telemetry:** lanes accept `pci_bus`; an independent 1 Hz amdgpu sysfs sampler (with
  optional AMD SMI) reports utilisation, VRAM, three temperatures, power, clocks and the negotiated PCIe
  link. Missing sensors are `null` with a reason, never `0`. amdgpu resets/page faults are captured from
  the kernel log.
- **Leases:** exclusive, expiring, token-protected claims on one or more lanes (all-or-nothing). Inference
  deploys onto a leased lane get HTTP 409, training cannot take a lane inference holds, and process
  termination verifies PID start time first. New `/api/leases` endpoints.
- **Reliability:** runs are `stalled` on step progress (not heartbeats), rank loss in DDP is detected,
  checkpoints are valid only once a manifest is committed, and a resume continuity report checks
  optimizer/scheduler/RNG state, LR schedule and loss continuity.
- **Evaluation pipeline:** bounded, coalescing checkpoint queue; adapter compatibility checks; native,
  GGUF F16 and quantized GGUF levels on a spare lane with explicit failures and server changeover timing;
  best-checkpoint selection on the deployed score (not training loss) and regression detection.
- **Experiments and analysis:** equal-token-budget comparison with seed-noise gating, batch/context/LoRA/
  gradient-checkpointing/precision benchmark sweeps (recommendations only), thermal degradation,
  cross-GPU contention, energy per million tokens, MFU estimate, memory forecaster with measured profiles,
  alerts, DDP straggler and scaling-efficiency analysis, rocprofv3 artifact attachments.
- **Storage:** pooled SQLite connections, incremental per-run aggregates, SQL-side downsampling, idempotent
  schema upgrades. Free-form payloads are scrubbed of personal data before storage.

Not verified on real hardware here: the Linux sysfs/AMD SMI readings, the LoRA to GGUF conversion and
`llama-server` changeover, and the DDP numerical-equivalence check (skipped without torch). All are
covered by fakes and run on the training machine without code changes.

## 2026-10-09 — Project-aware dual coding-agent mode

Added a coding-agent preset without removing or changing the existing evaluation modes or metric
schema. Runs can now carry project, workload, agent-role, session and coding-task attribution.
Projects, agent sessions, coding tasks and tool events are persisted in SQLite, with project and
GPU breakdowns exposed in the existing dashboard.

Added a streaming OpenAI-compatible gateway at `/v1/.../chat/completions`, including lane- and
role-pinned URLs suitable for running Cline and OpenCode concurrently. A registered client id used
as the API key supplies project attribution. Coding prompt and completion content is redacted from
storage by default while usage, prefill/decode timing, latency and resource telemetry are retained.

Dashboard totals now aggregate the complete matching history rather than inheriting the 500-row
cap from `recent()`. Run-history queries gained pagination and project/workload filters. The default
web bind was tightened to `127.0.0.1`; LAN access remains available with an explicit
`--host 0.0.0.0`.

## 2026-09-28 — `deploy_lane` now honors the documented join key, and the client docs cover the queue

Found by auditing this repo against the first client that actually built the two-queue scheduler
`deploy_lane` was added for (`talent-llm-eval`'s `main.py`).

### Fixed — `profile.models[]` silently stopped echoing caller strings after a lane replace

`deploy()` stores the caller's own raw profile as `active_profile`, so `profile.models[].model` is
the exact string the caller requested — which both `API_CONTRACT.md` and `CLIENT_API.md` document
in bold as the *only* reliable join key for matching your requested models back to lanes
(`servers[].model` is the resolved short display name and never matches a catalog id).
`deploy_lane_for_client` instead rebuilt `active_profile` from its live handles using
`ServerHandle.model.name` — the resolved short name — so the documented join key was wrong for
exactly the endpoint the documented multi-model scheduler depends on. A client following the docs
would match nothing, with no error.

Fixed with `DeploymentManager._requested_model_names_by_lane()`: a lane replacement now carries
this call's own `model` string for the lane it replaces, and forwards the previously recorded
request strings for the lanes it preserves. A lane whose request string was never recorded (a
direct `/api/deploy` that omitted `lane`) still falls back to the resolved name for that entry.
Three regression tests in `tests/test_lane_policy.py` cover the replaced lane, the preserved lane,
and the fallback.

### Fixed — `API_CONTRACT.md` documented error codes `check_fit` never returns

The `check_fit` entry claimed `400` for an unknown model/lane and `409` while a deployment change
was in flight. Neither happens: the endpoint takes no deployment lock (so never `409`), and its
own `except AppError` converts unknown lanes, unknown models, missing model files, unpaired
`input_tokens`/`max_output_tokens` and sub-256 contexts into `{"fits": false, "reason": ...}` at
`200`. The section also contradicted itself two paragraphs later. Rewritten to describe the real
behavior, including the consequence worth designing around: a typo'd `lane` returns `fits: false`
for every model tested, which is indistinguishable from a real "nothing fits this GPU" verdict.

### Documented — the independent per-lane queue, end to end

`check_fit` was absent from `CLIENT_API.md` entirely and `deploy_lane` had a single trailing
sentence, so the walkthrough described no way to keep both GPUs busy across a batch — the whole
point of both endpoints. Added step 1b (plan placement with `check_fit` before loading anything,
and the lane-key trap above) and step 2c (the full sequence: register once, bucket the list, start
each lane with `deploy_lane`, one pinned worker thread per lane, re-`deploy_lane` as each lane
frees up). Also newly documented: `deploy_lane` works from an idle service, so a scheduler does
**not** need a `deploy_parallel` first; a deployment is not owned by a `client_id`, which only
controls run attribution at chat time; and the whole-system resource fields
(`peak_system_ram_used_bytes`, `peak_system_pagefile_used_bytes`) are not comparable between
parallel and sequential runs, because one lane loading a model inflates the other lane's reading.

`README.md` mentioned neither endpoint. Its "How Scheduling Works" section describes the `dual-gpu`
CLI's own both-ends queue, which is unavailable over HTTP — now labeled as such, and pointed at
`CLIENT_API.md` step 2c for the client-side equivalent.

## 2026-09-27 — Independent per-lane model replacement

Added `POST /api/clients/deploy_lane`. It stops and replaces only the requested GPU lane while
preserving every other active lane, enabling work-conserving schedulers where a faster GPU loads
its next assigned model immediately instead of waiting at a fixed-pair barrier. Concurrent lane
replacement requests are serialized; chat traffic on the other lane continues during the load.

## 2026-09-27 — Context-aware smallest-fit GPU placement

Client auto-placement and parallel placement now use the full capacity verdict at the requested
context (model file plus estimated KV cache), not relative model size within a pair. Every model
that fits the 16 GB `9070xt` at 51,200 tokens stays on that lane. Only models that fail that check
are assigned to the 32 GB `r9700`. Parallel deploy accepts one model from each class; two small
models or two large-only models must run sequentially on their common lane instead of spreading
across both GPUs.

Removed the example config's stale `gpt-oss-20b-GGUF` pin to `r9700`, allowing the same fit rule
to decide its lane. Added regression tests for small/small rejection, cross-class pairing, and
context-aware single-model auto-placement.

Notable changes to this repo, in reverse-chronological order. See [CLIENT_API.md](CLIENT_API.md)
for a walkthrough of the client-integration flow and [API_CONTRACT.md](API_CONTRACT.md) for the
exhaustive field-by-field request/response reference of every endpoint. For the 2026-09-27
gap-fixing work specifically, [optimised.md](optimised.md) has the narrative version — how
each bug below was actually found and confirmed, not just what changed.

## 2026-09-27

### Fixed — `ResourceSampler` popped a new console window every 2 seconds

`app.py` is normally launched detached/without a console (see `launch_service.py`'s
`DETACHED_PROCESS` flag, meant to let it run as an invisible background service). Its
`ResourceSampler` background thread polls GPU/process metrics via `gpu_process_metrics_batch`
(PowerShell `Get-Counter`) every `interval_seconds` (2s) for the entire life of the process,
and `dual_gpu_setup/server.py`'s handful of other helper-process calls (`netstat`, `tasklist`,
`taskkill`, the `llama-server` launch itself, plus `dual_gpu_setup/lmstudio.py`'s
`list_devices`) never passed `creationflags=subprocess.CREATE_NO_WINDOW`. On Windows, a child
process spawned by a parent with no console of its own gets a brand-new console window unless
told not to — so every 2-second sample popped one. Harmless functionally, but a visible,
repeating nuisance for anyone running the service.

Fixed by adding a module-level `_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform ==
"win32" else 0` to both `dual_gpu_setup/server.py` and `dual_gpu_setup/lmstudio.py`, passed via
`creationflags=` on every `subprocess.run`/`subprocess.Popen` call in those two files.
`dual_gpu_setup/tasks.py` (the separate CLI `run`/`plan`/`inspect` command path, always run in
the user's own foreground terminal, never detached) was deliberately left unchanged — that path
never lacks a console in the first place, so the bug doesn't apply there.

Verified live: stopped the running (pre-fix) service, confirmed no console flashes with it
fully stopped, restarted it with the fix via `launch_service.py`, and had the reporting
codebase's user confirm directly that the flashing stopped.

### Added — `POST /api/clients/deploy_parallel`: auto-lane-picking parallel deploy

New `DeploymentManager.deploy_parallel_for_client()` + `POST /api/clients/deploy_parallel`.
Given a plain list of model names (one per configured lane, today 2), auto-assigns each the
best-fitting lane (largest lane first, each time taking the largest remaining model that still
fits it) and deploys via the *existing* `deploy()`/`_validate_profile("parallel_models", ...)`
path — no duplicated capacity/KV-cache/pin-lane validation. Previously a caller had to already
know which model belongs on which lane to use `parallel_models` mode at all; this is the
2-model equivalent of `deploy_for_client`'s existing single-model auto-lane-pick. Documented in
[API_CONTRACT.md](API_CONTRACT.md).

Built for `talent-llm-eval`'s automatic dual-GPU batch evaluation (pairs models 2-at-a-time,
deploys+runs each pair across both lanes concurrently instead of one lane at a time), but the
endpoint itself is generic — any client of this service's Client Integration API can call it.

**A real integration bug surfaced and was documented, not just fixed client-side:** the first
integration attempt matched requested model names against `servers[].model`, which is the
orchestrator's *resolved short display name* (e.g. `"gemma-3-270m-it-Q8_0"`), not the full
catalog id a caller requests with — every match silently failed and the client fell back to
sequential every time, with no visible error. `profile.models[]` (which echoes the exact
requested strings back, paired with their resolved lane) is the correct join key. Documented as
its own callout in `API_CONTRACT.md`'s `/api/clients/deploy_parallel` section — this ambiguity
was pre-existing in every deploy endpoint's response shape (`/api/deploy`, `/api/clients/deploy`
too), just newly surfaced by having two models to disambiguate between at once.

Verified live end-to-end, not just at the code level: deployed `gemma-3-270m-it-Q8_0` +
`SmolLM2-360M.Q8_0` in parallel (best-fit correctly put the larger SmolLM2 on the larger `r9700`
lane, the smaller gemma on `9070xt`), then ran real concurrent chat completions against both --
both threads started and finished together (not one-then-the-other), each returned distinct
real `timing`/`resources` telemetry (different tokens/sec, different GPU utilization, different
VRAM), confirming genuinely separate backends, not one lane silently double-serving both.

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
