# Dual GPU Studio: App Overview

**This file is the source of truth for what the app is, what it can do, and what to expect from it.**
If another document disagrees with this one about scope or behaviour, this one is meant to win, and the
disagreement is a bug to fix. For exact request and response shapes, [API_CONTRACT.md](API_CONTRACT.md) is
the authority. When behaviour changes, update this file in the same change (see
[Keeping this document true](#16-keeping-this-document-true)).

Last checked against the code: **2026-10-17** (390 automated tests passing, 1 skipped).

## Contents

1. [What the app is, and is not](#1-what-the-app-is-and-is-not)
2. [Where it runs, and what is verified](#2-where-it-runs-and-what-is-verified)
3. [How it is put together](#3-how-it-is-put-together)
4. [Five ways to use it](#4-five-ways-to-use-it)
5. [Features](#5-features)
6. [HTTP API map](#6-http-api-map)
7. [Configuration reference](#7-configuration-reference)
8. [Files the app writes](#8-files-the-app-writes)
9. [What to expect: measured behaviour](#9-what-to-expect-measured-behaviour)
10. [Rules for applications that use the service](#10-rules-for-applications-that-use-the-service)
11. [Limits, non-goals and known issues](#11-limits-non-goals-and-known-issues)
12. [Security posture](#12-security-posture)
13. [Testing and verification status](#13-testing-and-verification-status)
14. [Operating cheat-sheet](#14-operating-cheat-sheet)
15. [Where the other documents fit](#15-where-the-other-documents-fit)
16. [Keeping this document true](#16-keeping-this-document-true)

---

## 1. What the app is, and is not

**It is** a local service that runs `llama-server` (llama.cpp) on **two GPUs** of one Windows or Linux machine and
lets people and other applications use them without each project re-implementing GPU handling. It does five
jobs:

1. **Manages model servers.** It finds the GPUs, starts and stops one `llama-server` per GPU (or one
   across both), checks the model fits before loading, and keeps the weights in VRAM.
2. **Serves models over HTTP.** An OpenAI-compatible gateway (chat, autocomplete) plus a deployment API,
   so editors, coding agents, eval harnesses and backends all use the same two GPUs.
3. **Records everything.** Every request's token counts, timings, GPU and RAM use are stored and shown in a
   dashboard.
4. **Shares the GPUs with training.** GPU leases stop inference from landing on a GPU that is training, and
   a training tracker and evaluation pipeline report to the same dashboard.
5. **Controls itself from a UI.** Load, unload, restart, stop and change settings from the browser.

**It is not:**

- **Not an evaluator.** It runs your task commands against a model, or serves a model to your code. It does
  not score models itself.
- **Not an inference engine.** Generation is done by llama.cpp's `llama-server`. This app launches,
  watches and routes to it.
- **Not a trainer.** It never runs a training loop. It receives metrics from one and protects its GPUs.
- **Not multi-machine.** One machine, two GPUs ("lanes").
- **Not multi-user safe.** There is no authentication (see [section 12](#12-security-posture)).
- **Not a model host that unloads for you.** A loaded model stays on the GPU until something unloads it
  (see [section 10](#10-rules-for-applications-that-use-the-service)).

## 2. Where it runs, and what is verified

### Reference machine (everything marked "measured" was measured here)

| Item | Value |
|---|---|
| OS | Windows 11 Pro |
| GPUs | AMD Radeon AI PRO R9700 (32 GB), AMD Radeon RX 9070 XT (16 GB); a CPU-integrated Radeon also exists and is never used |
| RAM | 93.6 GB |
| `llama-server` | official llama.cpp **b11699** ROCm build (`F:\llama.cpp\b11699-rocm`), plus LM Studio's bundled ROCm runtime 2.55.0 |
| Python | 3.11 or newer; the service itself uses only the standard library plus SQLite |
| Models | GGUF files under `G:\LLM models` (about 100 models) |

### Status legend used in this file

- **Verified on hardware:** run against the real GPUs on the reference machine.
- **Tested with fakes:** covered by automated tests that simulate the hardware.
- **Not verified:** written, but never run against the real thing.

| Area | Status |
|---|---|
| Single-GPU, two-model parallel and split-model serving on Windows + ROCm | Verified on hardware |
| OpenAI gateway, on-demand loading, eviction, VRAM admission | Verified on hardware |
| Service restart, stop, settings, orphan handling, unload and Stop now | Verified on hardware (spare ports) |
| Dashboard pages | Page logic run against a fake DOM in Node; **not opened in a real browser after the latest changes** |
| Residency guard re-warming a model | Tested with fakes. **Not yet seen to trigger live** |
| Ubuntu/Linux support (`/proc`, sysfs, AMD SMI, DRM fdinfo, POSIX process control) | Tested with fakes; **not yet verified on the real two-GPU Linux machine** |
| GPU hardware sensors on Windows (busy %, temperatures, power, clocks, fan, VRAM, PCIe link, via AMD ADL) | Verified on hardware (R9700 and 9070 XT). The sensor ids were identified by behaviour; see 5.6 and 11 |
| Vulkan backend | Code path exists; not benchmarked or verified here |
| Training tracker, leases, evaluation pipeline | Tested with fakes and real subprocess/HTTP runs; the LoRA-to-GGUF conversion, real `llama-server` changeover during evaluation, Linux GPU sensors and the DDP numerical check are **not verified** |

## 3. How it is put together

```
   your apps / editors / eval harnesses / training jobs
                        |  HTTP
                        v
   +-----------------------------------------------------------+
   |  app.py   (one Python process, default http://127.0.0.1:8090) |
   |   dashboard.html  ·  JSON API  ·  OpenAI gateway  ·  Service control |
   |   DeploymentManager (loads/unloads/evicts/admits)   ResourceSampler (2 s) |
   |   ChatService   RunStore   leases   training store   residency guard |
   +------------+---------------------------+------------------+
                |                           |                  |
         llama-server (lane r9700)   llama-server (lane 9070xt)   SQLite files
         port 8081, ROCm0            port 8082, ROCm1              in the data dir
```

- **Lane:** one GPU plus its port, as configured in `[[lanes]]`. Exactly two lanes are supported.
- **Server:** one `llama-server` process. It serves one lane, or both lanes for a split model.
- **Service:** `app.py`. It owns the servers it starts. A server on a lane port that it did not start is an
  *orphan* (see [5.10](#510-service-control-from-the-ui)).
- **Dashboard:** `dashboard.html`, served by the service. It is read when the service starts, so UI changes
  need a restart to appear.

| Module | Role |
|---|---|
| `app.py` | HTTP server, dashboard, deployment manager, chat service, gateway, service control, SQLite stores |
| `dual_gpu_setup/config.py` | Loads and validates the TOML config |
| `dual_gpu_setup/lmstudio.py` | Finds the `llama-server` binary, probes its version and flags, discovers devices and models, reads free VRAM |
| `dual_gpu_setup/server.py` | Builds the `llama-server` command line, starts, health-checks and stops it, verifies placement |
| `dual_gpu_setup/orchestrator.py` | The CLI scheduler, context sizing and the KV-cache estimate |
| `dual_gpu_setup/tasks.py` | Runs your task commands against a loaded model (CLI mode) |
| `dual_gpu_setup/gguf.py`, `reasoning.py` | Reads GGUF metadata (for KV-cache size and native reasoning levels) |
| `dual_gpu_setup/settings.py` | The settings editable from the UI, and saving them into the TOML |
| `dual_gpu_setup/hostos/` | All Windows-specific and Linux-specific code behind one interface |
| `dual_gpu_setup/scheduling/leases.py` | GPU leases |
| `dual_gpu_setup/training/` | Training telemetry, checkpoints, evaluation, benchmarks, forecasting |
| `launch_service.py` | One-shot launcher for the service |

## 4. Five ways to use it

| Surface | Use it when | Start with |
|---|---|---|
| **Dashboard** | You want to load models, chat, compare two models, see history and telemetry, or control the service | `py launch_service.py` then open http://127.0.0.1:8090 |
| **OpenAI gateway** | An editor or coding agent (Cline, OpenCode, Continue, Aider) or any OpenAI client should use the GPUs | Base URL `http://127.0.0.1:8090/v1`, any API key |
| **Client Integration API** | Your own code needs to plan, load and run models and get metrics back | [CLIENT_API.md](CLIENT_API.md) |
| **Batch CLI** | You want a one-off sweep: every configured model run once against your task command | `dual-gpu --config <toml> run` |
| **Training tracker** | A PyTorch training job should report metrics and be protected from inference | `from dual_gpu_setup.training import TrainingTracker` |

## 5. Features

### 5.1 GPU discovery and lanes

- Each `[[lanes]]` entry names a GPU by a **substring** of the device description (`match = "R9700"`).
  The app asks `llama-server --list-devices` and maps each lane to a device id (`ROCm0`, `ROCm1`).
- Devices that no lane matches (such as the integrated GPU) are never used.
- A lane can carry `pci_bus` for a stable identity in training telemetry, and `extra_args` passed to every
  single-lane server on that lane (not to a model split across both lanes).
- `dual-gpu inspect` prints the runtime, the device list and the lane-to-device mapping.

### 5.2 Choosing the `llama-server` build

- `discovery.backend_server_binaries` points at an explicit binary per backend (`rocm`, `vulkan`). Without
  an entry, the newest LM Studio runtime for that backend is used.
- Each binary is probed once for its version and supported flags. Flags are added only when the build has
  them; a renamed flag in a model's `extra_args` (`--no-mmap`) is translated for builds that renamed it.
- `discovery.multi_gpu_runtime = "lmstudio"` makes models that span both GPUs use LM Studio's runtime while
  single-GPU servers use the configured build. This exists because the official b11699 ROCm build produces
  corrupt output when a model is split across the R9700 and 9070 XT (see [section 11](#11-limits-non-goals-and-known-issues)).
- The build used is recorded on every run, in `/api/status`, in `load.json` and by `dual-gpu inspect`.

### 5.3 Deploying models

Three deployment modes, available from the dashboard, `POST /api/deploy` and the client API:

| Mode | What loads |
|---|---|
| `single_gpu` | One model on one named lane |
| `parallel_models` | Two models, one per lane, exactly two |
| `single_large_model` | One model across both lanes in a single process |

**Checks before a model starts:**

1. **Fits the budget.** Weights (plus a draft model, if configured) must fit `vram_gb × vram_safety_fraction`
   (0.90 by default) of the lane or lanes. For a split model the budget is the sum.
2. **KV cache.** When the GGUF metadata is readable, the estimated KV cache for the context window is added
   and checked too. The estimate assumes a 16-bit cache held in VRAM; it does not read a model's
   `extra_args`, so a cache placed in RAM or quantized that way is still counted at full size (see
   [9](#9-what-to-expect-measured-behaviour) and [11](#11-limits-non-goals-and-known-issues)).
   Architectures the estimator cannot read confidently (hybrid state-space, sliding-window without a
   per-layer pattern) fall back to the weights-only check.
3. **Pinning.** A model with `pin_lane` is refused on any other lane.
4. **Leases.** A lane held by a training lease returns `409`.
5. **Free VRAM (GPU-only admission).** The model must also fit the VRAM actually free on the GPU, counting
   every process, plus `vram_headroom_gb`. Otherwise the load fails with `Not enough free VRAM ...` instead
   of letting Windows spill into system RAM.
6. **Placement check after start.** A server that is healthy but holds less than half its weights in VRAM
   is stopped and reported as failed (it silently fell back to the CPU).

**Context window:**

- An explicit `context_window` wins. Otherwise it comes from the model's `ctx_size`, or from VRAM headroom:
  `ctx_min` (16,384), `ctx_mid` (32,768) or `ctx_max` (49,152) depending on how much room is left.
- A client can instead send `input_tokens` and `max_output_tokens`. The context is raised to
  `input + output + safety_tokens` (256), and a request above `ctx_max` is rejected with `400`.

**Reasoning:**

- *Evaluation* deployments use numeric budgets: off = 0, low = 1,024, medium = 4,096, high = 8,192, or any
  value, with `-1` meaning unlimited.
- *Coding-agent* deployments use `--reasoning auto` with no budget, and the model's **native effort levels**
  read from its chat template. Levels are never mapped to invented token budgets.

**Per-model options** (`[[models]]`): `ctx_size`, `parallel`, `pin_lane`, `multi_gpu`, `tensor_split`,
`reasoning_mode`, `reasoning_budget`, `reasoning_effort`, `gpu_layers`, `flash_attn`, `cache_reuse`,
`threads`, `ubatch_size`, `extra_args`, `size_gb`, and **speculative decoding** (`draft_model`, `draft_max`,
`draft_min`, `draft_p_min`, `draft_gpu_layers`). Speculative decoding runs a small draft model on the same
GPU. The output is identical to the target alone; only speed changes, and the acceptance rate is recorded.

**Other deployment rules:**

- `POST /api/deploy` and `POST /api/clients/deploy*` replace the whole deployment, except
  `POST /api/clients/deploy_lane`, which replaces one lane and leaves the other running.
- Servers start in parallel. If any fails, all started ones are stopped and the call returns `500` with each
  failure.
- A model whose weights are split across several files is found by its first shard.

### 5.4 The model catalog

A model can be requested by name if it is in the catalog: every `[[models]]` entry in the config, plus every
`.gguf` file found under the model roots (`discovery.models_dir`, LM Studio's downloads folder and its
default models folder). Files named `mmproj` and non-first shards are not listed. `POST /api/models/refresh`
rescans.

### 5.5 Chat, compare and run recording

- The **Chat** page loads a deployment, sends a prompt to one model, or sends the same prompt to both
  models at once and shows the answers side by side.
- Every call is stored as a **run**: the prompt, the output, the reasoning text if any, the finish reason,
  the server command line and runtime build, and:
  - **tokens:** input, cached, uncached, prefill, thinking, visible output, output, total
  - **timing:** time to first token, prefill and decode time and rate, end-to-end
  - **resources:** GPU utilization, dedicated and shared VRAM, process RAM (working set and committed), page
    faults, system RAM and page file, spill suspicion, and the individual samples taken during the call
- Resources are sampled in the background every 2 seconds (about two hours of history are kept in memory),
  so a very short request may have no fresh sample. These figures describe a sampling window, not an exact
  measurement at the request boundary.

### 5.6 Dashboard pages

| Page | What it shows or does |
|---|---|
| Overview | Totals, prompt-processing vs generation speed per model, token split, 14-day activity, resource peaks, per-GPU performance, tool reliability, recent runs |
| Chat | Deployment form, single or compare chat, live system and server telemetry |
| Run history | Searchable runs; each opens to its prompt, output, token accounting, timings, placement and raw samples |
| Analytics | Breakdown by `task_label` category and model (runs, success, input/output tokens, speeds, latency, VRAM); coding reasoning performance (native effort, thinking tokens, tool-call validity) |
| Projects | Project registry, usage by project, coding-agent activity |
| Training | Training runs, comparison, DDP scaling, evaluation queue, benchmarks, memory forecast, measured profiles |
| Hardware | Live per-GPU readings, lease holders, sampler health. Windows (AMD ADL): busy %, edge / junction / memory temperature, board power, clocks, fan, VRAM, shared memory and the PCIe link. Linux reads amdgpu sysfs. Throttle status, AMD SMI and the kernel-fault follower are Linux only and show as unavailable on Windows |
| Service | Process info, restart, stop, loaded models, unload, settings (see 5.10) |
| Setup wizard | Creates a project and the connection settings for Cline and OpenCode |

A banner on every page appears when a leftover model server is running, and a notice bar shows what an
unload is waiting for.

### 5.7 The OpenAI-compatible gateway

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, streaming and non-streaming, with tool calls, structured output and reasoning |
| `POST /v1/completions`, `POST /v1/infill` | Inline autocomplete (fill-in-the-middle); a thin pass-through |
| `GET /v1/models` | The loaded lanes, plus every requestable catalog model with a `loaded` flag |

Each also exists as `/v1/<lane>/<role>/...`, which pins the request to a lane and names the agent role.

**Choosing the model:** `model` set to a catalog name or id is served by that model (see 5.8). `model:
"dgpu:<lane>"`, a lane key, or the lane in the URL routes to whatever is loaded there. With only one model
loaded, any other name is routed to it.

**Attribution without setup:** a request is attributed to a project from its Bearer key, or the
`X-DGPU-Project` header, or the default project (`unassigned`). The project is created on first use and
nothing is left untracked. Optional headers: `X-DGPU-Agent-Role`, `X-DGPU-Session`, `X-DGPU-Task`,
`X-DGPU-Task-Label`, `X-DGPU-Connection-Type`, `X-DGPU-Target`.

**Privacy of recorded data:**

- Prompts and completions are **redacted** in the metrics database unless the request sends
  `X-DGPU-Capture-Content: true`. Token counts, timings and resource figures are always kept.
- A separate **conversation archive** (`chat_history.sqlite3`) keeps full content. It is off unless
  `project.store_chat_history = true`, is written off the request path, and drops records rather than slow
  a request. `X-DGPU-No-History: true` skips one turn. Read it with `GET /api/chat-history`.
- Autocomplete requests are not recorded as runs; only a request count and token totals go to the daily
  usage ledger (`GET /api/usage`).

**Quality features:** `grammar`, `response_format` and `json_schema` are validated and passed straight to
`llama-server`; tool-call validity is measured on every coding run; the streamed response carries a run id
in the `X-DGPU-Run-ID` header.

### 5.8 On-demand model manager

With `policy.auto_load_models` on (the default), a gateway request for a model that is not loaded loads it:

1. Candidates are the lanes where the model fits at its default context, smallest first. A model too big
   for either single GPU spans both.
2. A candidate is usable only if it fits the VRAM obtainable now: the free VRAM plus what stopping **our own
   idle model** on that lane would release.
3. A free GPU is preferred. Otherwise the **least recently used idle** model is replaced.
4. **A model that is answering a request is never stopped.** If every candidate stays busy for
   `drain_timeout_seconds`, the caller gets `503` and retries.
5. If the model fits nowhere, the caller gets `503` with the shortfall **before** anything is evicted.
6. Loads are serialized; a request waits up to `project.start_timeout_seconds` for another load to finish.
7. A stale llama-server on a lane port is stopped first (`kill_busy_ports`).

Models loaded this way run in coding-agent mode and are recorded that way, so a restart can reload them.

### 5.9 Host RAM and GPU residency

Defaults measured to keep each server's RAM near **0.6 GB** instead of one model's worth plus a growing
prompt cache (details in [section 9](#9-what-to-expect-measured-behaviour)):

- `load_mode = "none"` keeps the weights from staying resident in RAM after they reach the GPU.
- `cache_ram_mib = 0` turns off llama.cpp's prompt cache in RAM. A conversation's own follow-up turns still
  reuse their context from the GPU.

**Residency guard** (`residency_guard`, checked every `residency_check_seconds`, 15 by default). Windows can
move an idle model's VRAM into system RAM; its next request then waits while it is paged back in. The guard
finds idle servers that are no longer resident and pages them back with a 1-token request.

- Servers with a single slot start with `--parallel 2 --kv-unified` (when the build supports it and the
  model's `extra_args` do not set slots). The second slot is reserved for warm-ups; the KV pool is shared, so no
  extra KV VRAM is used.
- Real requests are pinned to slot 0, so they still queue one at a time and a warm-up can never overwrite a
  conversation's cached prompt.
- Each server reports `resident_gb`, `evictions` and `rewarmed_at` in `/api/status`.
- It has an effect on Windows only. On Linux the check is a no-op.

### 5.10 Service control from the UI

The **Service** page, and the matching endpoints, cover everything that used to need a terminal:

| Control | What it does |
|---|---|
| **Restart service** | Stops every model (and leftover servers), starts a replacement process that waits for this one to exit before binding the port, and optionally reloads the same models afterwards. The page reconnects by itself. A saved deployment older than 10 minutes is ignored. |
| **Stop service** | Stops every model and then the service. Start it again with `py launch_service.py`. |
| **Unload** (per GPU, or everything) | Frees GPU memory. A model that is answering gets up to `drain_timeout_seconds` (30 s) to finish. |
| **Stop now** | Appears while an unload is waiting. Ends the wait and stops the model at once, cutting the answer. |
| **Orphans** | A llama-server on a lane port that this service does not manage (for example left by an instance that was killed) is listed in `/api/status` and in a banner, and **Stop them** removes it. It holds VRAM and RAM until stopped. A load stops it automatically. Nothing is reported while a load, swap or unload is running (`transitioning` in `/api/status`), because the service's own server is then starting or draining and is not a leftover. |
| **Settings** | Edits `load_mode`, `cache_ram_mib`, `vram_admission`, `vram_headroom_gb`, `auto_load_models`, `residency_guard`, `residency_check_seconds`, `drain_timeout_seconds` and `kill_busy_ports`. Values are validated, applied to the running service and saved into the TOML with its comments kept (a `.bak` copy of the original is made once). Each setting says whether it applies **now** or on the **next model load**. |

The restart, stop and settings endpoints require `Content-Type: application/json`.

### 5.11 Client Integration API

For programs that drive the service over HTTP (full guide: [CLIENT_API.md](CLIENT_API.md)):

| Call | Purpose |
|---|---|
| `POST /api/clients/register` | Name a project once and get a `client_id` |
| `POST /api/clients/check_fit` | Ask whether a model fits a lane at a context size, loading nothing |
| `POST /api/clients/deploy` | Load one model, lane chosen automatically unless named |
| `POST /api/clients/deploy_parallel` | Load one model per lane |
| `POST /api/clients/deploy_lane` | Replace one lane and leave the other running, so each GPU can have its own queue |
| `POST /api/chat` | Run a prompt; the reply carries usage, timing and resources, and a JSON line is appended to the client's own metrics file |
| `GET /api/clients/{id}/runs` | The client's last 100 runs |
| `GET /api/clients/schema` | The metric field list |
| `POST /api/unload` | **Release the GPU when the task is done** (rules in [section 10](#10-rules-for-applications-that-use-the-service)) |

### 5.12 Projects, sessions, usage and archives

- **Projects, agent sessions, coding tasks and tool events** (`/api/projects`, `/api/agent-sessions`,
  `/api/coding-tasks`, `/api/tool-events`) attribute runs and tool results to a repository, an agent role and
  a task. The dashboard rolls them up.
- **Usage ledger** (`GET /api/usage`): daily request and token counts per project and connection type,
  including autocomplete.
- **Run pruning:** there is no automatic retention. `app.py --prune-older-than-days N` deletes older runs once.

### 5.13 GPU leases (training versus inference)

- A lease is an exclusive, expiring, token-protected claim on one or more lanes, taken all or nothing.
  Default lifetime 120 s unless renewed.
- A deploy onto a leased lane returns `409`; a lease cannot be taken on a lane that inference holds.
- Terminating a lease holder verifies the process start time first, so a reused PID is never killed.
- Endpoints: `GET/POST /api/leases`, `POST /api/leases/{id}/renew`, `POST /api/leases/{id}/preempt`,
  `DELETE /api/leases/{id}`.

### 5.14 Training telemetry and evaluation

This layer is designed for a PyTorch LoRA/SFT job (first target: the R9700) and is separate from inference.
Full design and status: [docs/TRAINING_TELEMETRY.md](docs/TRAINING_TELEMETRY.md).

| Capability | What it does |
|---|---|
| Tracker client | `TrainingTracker` (standard library only) and a Hugging Face `Trainer` callback stream loss, learning rate, gradient norm, token counts, step time, GPU phase timing and memory. Metrics are per window, never per micro-batch, and the client never blocks or fails training (spool to disk and replay if the dashboard is down). |
| Run reliability | `stalled` is judged on step progress, rank loss in DDP is detected, a checkpoint counts only once its manifest is committed, and a resume continuity report checks optimizer, scheduler and RNG state. |
| Hardware telemetry | A 1 Hz sampler reports utilization, VRAM/GTT, temperatures, power, clocks, PCIe link and AMD SMI throttle status; sensor source and unavailable reasons are persisted, and missing values are `null`, never `0`; amdgpu resets and page faults are captured from the kernel log. |
| Reproducibility | Every run has top-level host OS attribution even when tracker-side environment capture is disabled. Full snapshots distinguish the training-process OS from the dashboard host and include kernel, ROCm component versions, amdgpu, PyTorch/HIP, relevant ROCm environment variables and physical GPU mapping. |
| Evaluation pipeline | Checkpoints queue for evaluation on a spare lane (native, GGUF F16 and quantized GGUF), best checkpoint is chosen on the deployed score, regressions are flagged. Needs `[evaluation]` configured. |
| Experiments | Equal-token-budget comparison with seed-noise gating, benchmark sweeps (batch, context, LoRA, checkpointing, precision) that only recommend, a memory forecast, energy per million tokens, MFU estimate, DDP straggler and scaling analysis. |
| Privacy | Free-form payloads are scrubbed (content-bearing keys dropped, long strings truncated, e-mail and phone-like text redacted) before storage. |

Not built: preference-pair collection, FSDP/ZeRO and pipeline parallelism, and automatic restart or tuning.

### 5.15 LM Studio GPU selection

If LM Studio is also used on the machine: `GET/POST /api/lmstudio/gpus` read and set which GPUs LM Studio's
runtime may use (by editing `load.gpuSplitConfig.disabledGpus` in its `hardware-config.json`, preserving
other fields and backing up the original), and `POST /api/lmstudio/release` unloads LM Studio's models
(optionally stopping its server) so the GPUs are free first. This changes the saved setting only; models
already loaded keep their placement. These POSTs require `Content-Type: application/json`.

### 5.16 The batch CLI

`dual-gpu --config <toml> <command>`:

| Command | Purpose |
|---|---|
| `inspect` | Show runtime, build, devices and the lane mapping |
| `plan` | Show which models fit which lane and which are multi-GPU |
| `run` | Load each model, run your `[[tasks]]` command against it, stop it, move on |
| `run --dry-run` | Create the run folder and summary without loading anything |

**Scheduling:** models are sorted by size and drained from both ends at once. The large GPU takes the biggest
model that fits while the small GPU takes the smallest, so neither waits on the other's slow model.
Models with `multi_gpu = true` run last, spanning both lanes. A model that fits neither lane is skipped and
listed in `summary.json`.

**Tasks** receive placeholders in `command`, `cwd` and `env` (`{base_url}`, `{model_name}`, `{model_path}`,
`{model_size_gb}`, `{suite_id}`, `{run_dir}`, `{actual_ctx}`, `{lane_key}`, `{lane_display}`, `{backend}`,
`{device}`, `{host}`, `{port}`, `{project_name}`, `{config_dir}`) and the environment variables
`DGPU_SUITE_ID`, `DGPU_BASE_URL`, `DGPU_MODEL_NAME`, `DGPU_LANE_KEY`, `DGPU_LANE_DISPLAY`, `DGPU_BACKEND`,
`DGPU_DEVICE`, `DGPU_ACTUAL_CTX`, `DGPU_RUN_DIR`, `DGPU_HOST_OS`, `DGPU_HOST_OS_JSON`.

**Output** under `log_dir`: `summary.json`, `server_logs/`, one folder per model with `load.json` and task logs.
The summary and every plain-text launch/task log include host OS metadata.

## 6. HTTP API map

Everything is JSON on `http://127.0.0.1:8090`. Shapes and errors: [API_CONTRACT.md](API_CONTRACT.md). Errors are
`{"error": "..."}` with `400` bad input, `404`, `409` a change is in progress, `413` body over 10 MB, `415`
not JSON where required, `500`, `501` not available, `503` cannot serve now (retry).

| Group | Endpoints |
|---|---|
| UI and state | `GET /`, `/api/bootstrap`, `/api/status`, `/api/dashboard`, `/api/runs`, `/api/runs/{id}`, `/api/chat-history`, `/api/usage`, `/api/projects` |
| Deployment | `POST /api/deploy`, `/api/models/refresh`, `/api/unload`, `/api/chat` |
| Service control | `GET /api/service`, `POST /api/service/restart`, `POST /api/service/stop`, `GET/POST /api/settings` |
| OpenAI gateway | `GET /v1/models`, `POST /v1/chat/completions`, `/v1/completions`, `/v1/infill`, and the `/v1/<lane>/<role>/...` forms |
| Client integration | `POST /api/clients/register`, `check_fit`, `deploy`, `deploy_parallel`, `deploy_lane`; `GET /api/clients/{id}/runs`, `/api/clients/schema` |
| Projects and agents | `POST /api/projects`, `/api/agent-sessions`, `/api/coding-tasks`, `/api/tool-events` |
| Leases and hardware | `GET/POST /api/leases`, `.../{id}/renew`, `.../{id}/preempt`, `DELETE /api/leases/{id}`, `GET /api/hardware` |
| Training | `/api/training/runs` (and `/{id}`, `/metrics`, `/events`, `/checkpoints`, `/evals`, `/finish`, `/series`, `/analysis`, `/alerts`, `/ranks`, `/artifacts`, `/notes`, `/examples`, `/resume-report`), `/compare`, `/scaling`, `/alerts`, `/eval-queue`, `/benchmarks`, `/forecast`, `/profiles`, `/notes` |
| LM Studio | `GET/POST /api/lmstudio/gpus`, `POST /api/lmstudio/release` |

## 7. Configuration reference

One TOML file, passed with `--config` (default `example.dual_gpu.ubuntu.toml` on Linux and
`example.dual_gpu.toml` on Windows). Unknown keys are ignored. Relative discovery paths are resolved
from the config file's directory, and `~`/environment variables are expanded.

**`[project]`**

| Key | Default | Meaning |
|---|---|---|
| `name` | `dual-gpu` | Label in run summaries |
| `suite_prefix` | `dualgpu` | Prefix of generated run ids |
| `log_dir` | `./runs` | Where CLI run folders go |
| `host` | `127.0.0.1` | Bind host for `llama-server` |
| `cleanup_ports_on_start` | `true` | Batch CLI `run` only: stop leftover servers on lane ports before starting |
| `start_timeout_seconds` | `900` | How long a model may take to become healthy |
| `store_chat_history` | `false` | Keep full conversations in `chat_history.sqlite3` |
| `chat_history_path` | next to the metrics DB | Archive location |
| `default_project` | `unassigned` | Project for unattributed requests |

**`[discovery]`:** `lmstudio_home`, `models_dir`, `backend_server_binaries`, `backend_runtime_overrides`,
`backend_vendor_bin_overrides`, `multi_gpu_runtime` (`configured` or `lmstudio`).

**`[policy]`**

| Key | Default | Meaning |
|---|---|---|
| `vram_safety_fraction` | `0.90` | Share of a lane's VRAM usable for model files |
| `ctx_min` / `ctx_mid` / `ctx_max` | `16384` / `32768` / `49152` | Auto context sizes |
| `ctx_mid_headroom_gb` / `ctx_max_headroom_gb` | `5` / `10` | Headroom needed for mid and max |
| `safety_tokens` | `256` | Margin added to a client's token budget |
| `reasoning_mode` / `reasoning_budget` | `off` / `0` | Default reasoning |
| `cache_reuse` | `256` | llama.cpp prompt reuse chunk size |
| `flash_attn` | `auto` | Flash attention |
| `gpu_layers` | `999` | Layers to offload |
| `parallel` | `1` | Slots per server |
| `no_mmproj`, `jinja` | `true`, `true` | Skip multimodal projector; use the model's chat template |
| `kill_busy_ports` | `true` | Stop leftover llama-servers on lane ports when loading |
| `health_poll_seconds`, `settle_seconds_after_restart` | `1.0`, `2.0` | Start-up polling |
| `drain_timeout_seconds` | `30` | How long a swap or unload waits for running requests |
| `load_mode` | `none` | `none`, `mmap` or `auto` |
| `cache_ram_mib` | `0` | Prompt cache in RAM (0 = off) |
| `vram_admission`, `vram_headroom_gb` | `true`, `1.0` | Free-VRAM check before loading |
| `auto_load_models` | `true` | Gateway loads requested models on demand |
| `residency_guard`, `residency_check_seconds` | `true`, `15` | Keep idle models in VRAM |

**`[[lanes]]`:** `key`, `display`, `match`, `vram_gb`, `port`, `backend` (`rocm` or `vulkan`), `host`,
`extra_args`, `pci_bus`.

**`[[models]]`:** `name`, `path` (absolute, relative to the model root, or a glob), and the options in 5.3.

**`[[tasks]]`:** `name`, `command`, `cwd`, `shell`, `continue_on_error`, `env`.

**`[evaluation]`** (training checkpoint evaluation): `enabled`, `lane`, `base_model`, `base_f16_gguf`,
`base_quant_gguf`, `converter_script`, `native_command`, `gguf_command`, `parity_set`, `designate_every`,
`primary_metric`, `queue_capacity`, `ctx_size`, `timeout_s`, thresholds for accuracy and latency regression,
and `peak_tflops` per lane for the MFU estimate.

**`app.py` options:** `--config`, `--host` (default `127.0.0.1`), `--port` (default `8090`), `--data-dir`
(default `./chat_runs`), `--no-browser`, `--check` (validate and exit), `--prune-older-than-days N`. The
internal `--replace-pid` is used by the restart.

## 8. Files the app writes

| File | Contents |
|---|---|
| `<data-dir>/chatbot.sqlite3` | Runs (including host OS), deployments, resource samples, projects, sessions, coding tasks, tool events, clients |
| `<data-dir>/chat_history.sqlite3` | Full conversations (only if `store_chat_history`) |
| `<data-dir>/project_usage.sqlite3` | Daily usage per project |
| `<data-dir>/training.sqlite3` | Training runs, steps, checkpoints, hardware samples, evaluation queue, leases, benchmarks |
| `<data-dir>/deployments/<time>/server_logs/*.log` | Each `llama-server`'s output, with its exact command line and host OS at the top of each launch |
| `<data-dir>/launcher.log` | Output of the background service and of restarts, with host OS in each launch header |
| `<data-dir>/restart_state.json` | Present only between a restart with "reload models" and the new process starting |
| `<data-dir>/artifacts/` | Training artifacts |
| `<config>.bak` | A copy of the config, made once, before the first settings save |
| `<log_dir>/<suite>/` | CLI run output |

The data directory defaults to `./chat_runs` next to the config file.

## 9. What to expect: measured behaviour

Measured on the reference machine on **2026-10-17** with llama.cpp b11699 ROCm. These are observations, not
guarantees; they change with the model, quantization, context and driver.

### Speed (decode = generation tokens/s, prefill = prompt tokens/s)

| Model | GPU | Decode | Prefill | Source |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 | R9700 | 153 (140 through the app) | 6,382 | `llama-bench`; app runs |
| gpt-oss-20b MXFP4 | 9070 XT | 148 (137 through the app) | 6,355 | `llama-bench`; app runs |
| gemma-4-12B QAT Q4_0 | 9070 XT | about 63 | 570 to 2,500 (varies with prompt length) | app runs |
| Qwen3.8-27B Q4_K_M | R9700 | about 31 | 765 to 1,136 | app runs |
| REAP 58B MoE (38.5 GB), split across both GPUs | both | about 81 | not measured | app, LM Studio runtime |
| Qwen3-32B Q4_K_M, split 2:1 | both | about 25 to 27 | 1,025 to 1,250 | `llama-server`, LM Studio runtime |

- The official b11699 build beat LM Studio's runtime on single-GPU gpt-oss-20b by **4 to 8 % prefill and 4
  to 7 % decode**. Its split-model speed is not reported because its output was corrupt (section 11).
- Running two models in parallel did not slow the R9700's gpt-oss-20b (140 tok/s alone, 139 with a second
  model running). The second model on the 9070 XT was a different model, so its speed is not a like-for-like
  comparison.

### Time

| Action | Typical time |
|---|---|
| Load a 12 to 20 GB model onto one GPU | 8 to 16 s |
| Load a 38 GB model across both GPUs | about 20 to 25 s |
| Unload an idle model | about 2.6 s |
| Unload a model that is answering | waits up to 30 s unless **Stop now** is used (3.8 s when used after 3 s) |
| Restart the service | about 6 s with nothing loaded; 12 s in total when a 0.8 B model is reloaded; a larger model adds its own load time |
| Page an evicted 16.85 GB model back into VRAM | 10.3 s on the next request (0.2 s once resident) |

### Memory

| Setting | RAM held by one llama-server |
|---|---|
| llama.cpp defaults (Qwen3-8B, 4.7 GB) | 4.8 GB after load, growing to 8.9 GB after 20 distinct prompts |
| Defaults with a 20B model, 150 prompts | 11.6 GB growing to 18.4 GB, still rising |
| This app's defaults (`load_mode = "none"`, `cache_ram_mib = 0`) | 0.6 GB resident, flat; decode speed unchanged (96 tok/s) |

Every model unloaded cleanly in the tests: the process exited and GPU and system memory returned to their
starting levels.

### Where the context cache lives (KV cache)

`llama-server` reserves the whole KV cache for the configured context window when it loads. It does not grow as
the window fills (VRAM stayed flat while 16k tokens were filled), so where it lives is decided at load, by flags.
Measured on the R9700 with Qwen3-8B Q4_K_M at a 262,144-token context, which needs a 36.9 GB cache on a 32 GB
card (2026-10-17, one run each):

| llama-server flags | Cache lives | VRAM | Decode at 2k / 16k prompt (tok/s) | Prefill at 2k / 16k (tok/s) |
|---|---|---|---|---|
| none | GPU, but 11.4 GB was silently backed by system RAM (Windows) | 31.7 GB | 77 / 32 | 1,202 / 1,100 |
| `--no-kv-offload` | system RAM, 36.9 GB | 5.9 GB | 24 / 7 | 3,173 / 2,168 |
| `--cache-type-k q8_0 --cache-type-v q8_0` (needs `--flash-attn on`) | GPU, half size | 25.1 GB | 85 / 72 | 3,575 / 3,316 |
| `--cache-type-k q4_0 --cache-type-v q4_0` (needs `--flash-attn on`) | GPU, quarter size | 16.1 GB | 91 / 71 | 3,895 / 3,310 |
| `--fit on` (this build's default) | unchanged: it only adjusts arguments that are not set | 31.7 GB | 77 / 32 | 1,206 / 1,094 |

- A cache in RAM works but decode slows sharply with context depth. A quantized cache kept on the GPU was
  fastest. The quality cost of a quantized cache was **not measured**; q4_0 is the riskier choice.
- The context was far beyond the model's trained 32,768, which llama.cpp warns about. That does not change
  the memory behaviour, only the quality of long answers.
- Weights can be moved to RAM instead with `--n-cpu-moe N`, `--cpu-moe`, `-ncffn N` or fewer GPU layers; these
  were not measured here. `gpt-oss-120b` in the example config uses `--n-cpu-moe 20`.

## 10. Rules for applications that use the service

These apply to every program that loads a model through this service.

1. **Release the GPU when your task is done.** The service never unloads a model because a task ended, and
   has no idle timeout. After the last request of the task has finished, on success and on failure, send
   `POST /api/unload {"lane": "<lane>", "model": "<the model you deployed>"}` for each lane you used. The
   `model` value makes it safe: if another application has since loaded something else on that lane,
   nothing is stopped and the reply says `released: false`.
2. **Never send the bare `POST /api/unload {}` or `{"force": true}`** from an application. They stop every
   model, including other applications'.
3. **Do not release between requests of one task.** Reloading takes seconds to tens of seconds.
4. **Handle `503` and `409`.** `503` means the model cannot be served right now (all GPUs busy, or not
   enough free VRAM): retry later. `409` means a load or unload is in progress: retry in about 2 s.
5. **With per-lane queues** (`deploy_lane`), release each lane when its own queue is empty; do not wait for
   the other lane.
6. **Do not call a whole-deployment endpoint** (`/api/deploy`, `/api/clients/deploy`,
   `/api/clients/deploy_parallel`) while another application may be using the other lane: they replace
   everything.
7. **Tools that cannot call HTTP** (Cline, OpenCode, Continue): whoever starts the session releases the
   models at its end from the Service page, or with the call in rule 1.

Nothing enforces rule 1. The service does not record who loaded what; the `model` guard is what keeps it safe.
Full text and example: [API_CONTRACT.md](API_CONTRACT.md#releasing-the-gpus-when-a-task-is-done).

## 11. Limits, non-goals and known issues

**Scope limits**

- Exactly **two lanes**.
- **Ubuntu/Linux hardware is unverified.** The code path and fake `/proc`/DRM tests exist, but the
  reference two-GPU machine is Windows.
- **One machine**, one service.
- The deployment APIs keep **one deployment at a time**. A whole-deployment call replaces everything;
  `deploy_lane` and the gateway's on-demand loading manage lanes individually.
- Only models in the catalog (section 5.4) can be requested by name.
- The KV-cache check is an estimate; unreadable architectures fall back to a weights-only check.
- `task_label` has no fixed list; it is whatever string the caller sends, lower-cased.
- There is no automatic retention of run history.

**Known issues on the reference machine**

1. **Split models are corrupt on the official b11699 ROCm build.** Any model spanning the R9700 and 9070 XT
   produced garbage output on that build, whatever the split ratio, device order, split mode or flash
   attention setting. Each GPU alone is correct. LM Studio's runtime splits correctly, so
   `multi_gpu_runtime = "lmstudio"` routes split models to it. Re-test with each new llama.cpp build before
   changing that.
2. **Windows can move an idle model's VRAM into system RAM.** A 16.85 GB model read 0 GB resident while idle,
   and the next request took 10.3 s. The trigger is not identified (it did not happen from idle time alone,
   from listing devices, or from loading another model). The residency guard is the mitigation, and it has
   not yet been seen to trigger live.
3. **The capacity check ignores CPU-offloaded experts.** `gpt-oss-120b` (59 GB) is refused by `/api/deploy`
   because its configured `--n-cpu-moe 20` experts are not subtracted from the weights.
4. **No cancel for a single answer.** The only way to stop a running answer is to unload the model.
5. **Free-VRAM readings:** HIP's own free-memory figure is wrong on Windows (it reported an occupied card as
   empty), so the app uses the operating system's accounting instead. Anything else that reads the HIP figure
   will be misled.
6. **Vulkan** is configured as a backend but has not been benchmarked or verified here.
7. **Windows GPU sensor ids are inferred, not documented.** Temperatures, power, clocks and fan come from AMD's
   ADL library, whose sensor numbering was identified on the R9700 and 9070 XT by behaviour (edge matched Task
   Manager; junction reacted in about a second; memory reacted over tens of seconds; board power sat at each
   card's rated draw). A different GPU generation or driver may map differently. Two identical cards cannot be
   told apart by name, so their VRAM would show as unavailable.
8. **No parameter places the KV cache in RAM or sets its precision.** A deploy request accepts
   `context_window`, `input_tokens`, `max_output_tokens`, the reasoning fields and `tensor_split`, but nothing
   about the cache. An operator can add the flags in section 9 to a model's `extra_args`, but the capacity check
   still counts the cache at 16-bit size in VRAM, so a context that only fits with the cache in RAM or
   quantized is refused before `llama-server` starts (Qwen3-8B at 262,144 on the R9700: "needs 40.68 GB, safe
   budget 28.80 GB"; at 131,072 on the 9070 XT: "needs 22.68 GB, budget 14.40 GB"). Not built.

## 12. Security posture

- **There is no authentication.** Anyone who can reach the host and port can call any endpoint, including
  unload, deploy and (with a JSON content type) restart, stop and settings. The default bind is
  `127.0.0.1`. Binding to `0.0.0.0` (needed for LAN use, see [local_network_setup.md](local_network_setup.md))
  exposes full control and recorded data to the network; use network-level access control.
- **Only some endpoints require a JSON content type** (restart, stop, settings, and the two LM Studio
  POSTs). That forces a CORS preflight, so a web page cannot trigger them with a blind form post. Other
  state-changing endpoints, including `/api/deploy` and `/api/unload`, accept any content type, so a web
  page you visit could in principle post to them on a local service. This is a known gap.
- `llama-server` processes bind to `127.0.0.1` unless a lane or model `host` says otherwise, and run without
  an API key or web UI protection of their own.
- Coding prompts and completions are redacted from the metrics database by default; the conversation
  archive is opt-in and holds full content.
- Training payloads are scrubbed of personal data before storage.

## 13. Testing and verification status

- **390 automated tests** (`py -m pytest`) pass, 1 skipped (a DDP numerical check that needs `torch`). They
  cover the GGUF reader, KV estimator, config, run store, capacity checks, runtime and flag selection,
  the model manager (placement, eviction, admission, orphans, unload, Stop now, release), service control,
  settings saving, the gateway, leases and the training layer.
- The suite is guarded so it can never see or stop a real `llama-server` on the real lane ports.
- **Real-hardware checks** were run on spare ports (service 8095, lanes 8181/8182) so the live service
  was untouched: restart with reload, settings that survive a restart, orphan detection and removal, Stop
  now, per-lane unload, the release clause, and on-demand loading with eviction.
- **Not covered:** a real browser session of the dashboard after the latest changes; the residency guard
  firing live; anything on Linux.

## 14. Operating cheat-sheet

```powershell
# Start the service in the background and open the UI (use py: `python` is not on PATH here)
py launch_service.py
py launch_service.py --no-browser            # no browser, e.g. from a script

# Run in the foreground instead
py app.py --config .\example.dual_gpu.toml

# Check the setup without starting anything
dual-gpu --config .\example.dual_gpu.toml inspect
dual-gpu --config .\example.dual_gpu.toml plan
py app.py --check

# Tests
py -m pytest
```

Ubuntu equivalents use the edited Linux config and native llama.cpp binary:

```bash
python3 launch_service.py --config ./dual_gpu.toml
python3 app.py --config ./dual_gpu.toml --check
dual-gpu --config ./dual_gpu.toml inspect
dual-gpu --config ./dual_gpu.toml plan
python3 -m pytest
```

| I want to | Do this |
|---|---|
| Free the GPUs | Service page, **Unload everything** (or `POST /api/unload {}` if you are the only user) |
| Restart after changing code or the dashboard | Service page, **Restart service**, keep "Load the current models again" ticked |
| Change a memory or loading option | Service page, Settings, **Save** |
| See why a model failed to load | `<data-dir>/deployments/<latest>/server_logs/<model>.server.log`, and the `Not enough free VRAM` message |
| Find leftover model servers | The orange banner, or `orphans` in `GET /api/status` |
| Give an editor access | Base URL `http://127.0.0.1:8090/v1/<lane>/<role>`, any API key, `model` set to a name from `GET /v1/models` |

## 15. Where the other documents fit

| Document | Use it for |
|---|---|
| **app_overview.md** (this file) | What the app is, what it can do, what to expect, its limits |
| [README.md](README.md) | Setup, installation, CLI walkthrough, config guide, integration examples |
| [API_CONTRACT.md](API_CONTRACT.md) | Exact request and response shapes, errors, the release rule |
| [CLIENT_API.md](CLIENT_API.md) | Step-by-step guide for a program that drives the service |
| [CODING_AGENTS.md](CODING_AGENTS.md) | Cline, OpenCode and Continue setup, reasoning, speculative decoding, archive |
| [local_network_setup.md](local_network_setup.md) | Using the UI from another computer on the LAN |
| [docs/TRAINING_TELEMETRY.md](docs/TRAINING_TELEMETRY.md) | Training telemetry design, metrics catalog and status |
| [CHANGELOG.md](CHANGELOG.md) | What changed and why, with measurements |
| `docs/CHATBOT_LAYER.md`, `TELEMETRY_AND_STORAGE.md`, `UI_SPECIFICATION.md`, `IMPLEMENTATION_PLAN.md`, `optimised.md` | Earlier design notes and investigations; useful background, not current reference |

## 16. Keeping this document true

- **Change behaviour, change this file.** Any change that adds, removes or alters a feature, a default, a
  limit or an endpoint updates the matching section here in the same change, and the changelog.
- **Do not write a claim that was not checked.** State what was measured or tested, and mark the rest "not
  verified". Measured numbers carry their date and machine.
- **Move things between sections as they are fixed.** A known issue that is fixed leaves section 11 and goes
  into the changelog; a verification that is done updates the table in section 2.
- **Wire details live in API_CONTRACT.md.** This file describes scope and behaviour, and links there rather
  than copying request bodies.
