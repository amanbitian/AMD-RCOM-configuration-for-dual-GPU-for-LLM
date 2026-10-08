# Dual GPU Setup

`Dual_gpu_setup` is a reusable dual-GPU orchestration layer for local `llama-server` workloads on Windows. It was designed so eval projects and future local-model apps can share one GPU management system instead of each repo reimplementing:

- GPU discovery
- lane scheduling
- model placement
- `llama-server` startup and shutdown
- multi-GPU model routing
- VRAM spill checks
- per-model task execution

This repo was built by extracting the good dual-GPU ideas from:

- `F:\Projects\LLM_evaluator`
- `F:\Turing\eval_setup 2907`

and turning them into a cleaner, config-driven service layer.

## Screenshots

The web app ([app.py](app.py)) gives you a full local console for loading, chatting with, and comparing models across both GPUs — see [How It Works](#how-it-works) below for what is happening behind each view.

**Performance overview** — total runs, tokens processed, prompt-processing (prefill) vs. token-generation (decode) throughput per model, resource peaks, and 14-day activity, all aggregated live from every run's telemetry.

![Overview dashboard showing per-model prompt processing and token generation throughput](docs/screenshots/overview.png)

**Chat / compare workspace** — deploy one model split across both GPUs, two independent models (one per GPU), or a single model on one GPU, then send a prompt and watch both answers, token counts, and speeds side by side.

![Chat workspace running Qwen3.5-4B on the R9700 and gemma-3-270m on the 9070 XT at the same time](docs/screenshots/chat.png)

**Run history** — every prompt, output, token count, timing, and VRAM/RAM sample is persisted to SQLite and searchable afterward.

![Run history table with per-run prefill/decode throughput and resource columns](docs/screenshots/run-history.png)

## What This Project Does

This project does not evaluate models by itself.

It is an orchestrator that:

1. Reads a TOML config.
2. Discovers your LM Studio runtime and GPUs.
3. Creates one serving lane per GPU.
4. Loads models onto the correct GPU or across both GPUs.
5. Starts a local OpenAI-compatible `llama-server` endpoint.
6. Runs your configured task commands against that endpoint.
7. Stops the model and moves to the next one.

Because of that, you can plug many other applications into it:

- eval repos
- benchmark scripts
- Python apps
- Node.js apps
- FastAPI or Flask backends
- prompt testing tools
- internal automation scripts

## Important Behavior

The system is best understood as a run-time orchestrator, not a permanent API daemon.

- During a run, it starts one or more local `llama-server` instances.
- Those servers expose an OpenAI-compatible HTTP API on the configured ports.
- Your tasks or external apps can connect to those endpoints while the model is loaded.
- After the model finishes, the server is stopped and the next model may be loaded.

So if another application wants to use this service, it should either:

- be launched by this orchestrator as a configured task, or
- connect to the lane endpoint while the model is alive, or
- you extend this repo later with a persistent `serve-only` mode

The current version already supports the first two very well.

## How It Works

There are two ways to drive this repo. Both end up going through the same GPU-discovery and `llama-server` lifecycle code, they just expose it differently.

### 1. Batch orchestration (the `dual-gpu` CLI)

1. [config.py](dual_gpu_setup/config.py) loads and validates your TOML file — lanes, models, tasks, and policy defaults (context sizes, VRAM safety fraction, flash attention, etc).
2. [lmstudio.py](dual_gpu_setup/lmstudio.py) asks the LM Studio runtime for its device list and matches each configured lane (`match = "R9700"`, `match = "9070 XT"`) to a physical `ROCm` device id.
3. [orchestrator.py](dual_gpu_setup/orchestrator.py) sorts the model queue by size and drains it from both ends at once: the large GPU keeps pulling the biggest model that still fits while the small GPU keeps pulling the smallest, so neither card sits idle waiting on the other's long-running model (see [How Scheduling Works](#how-scheduling-works)).
4. For each model, [server.py](dual_gpu_setup/server.py) launches `llama-server.exe` pinned to that GPU's device id, polls its health endpoint until it is ready, and watches memory to flag VRAM spill into shared/system memory.
5. [tasks.py](dual_gpu_setup/tasks.py) runs your configured command(s) against that model's local OpenAI-compatible endpoint, substituting placeholders like `{base_url}` and `{model_name}` and injecting the `DGPU_*` environment variables.
6. The server is stopped and the next model in the queue loads. Models marked `multi_gpu = true` are held back and run last, spanning both lanes in a single `llama-server` process.
7. Everything lands in a run folder under `log_dir`: `summary.json`, per-model `load.json`, server logs, and task logs.

Use this path for unattended benchmark or eval sweeps where you just want every configured model exercised once and logged.

### 2. Interactive web app (`app.py`)

[app.py](app.py) is a dependency-free local web server (Python's standard-library `http.server` plus SQLite — no extra packages required) that wraps the same discovery and server-launch code in an interactive console, shown in the [screenshots](#screenshots) above:

1. On startup it reads your TOML config and scans LM Studio's settings and model directory for every installed GGUF quantization, then serves a single-page UI plus a small JSON API: `/api/bootstrap`, `/api/deploy`, `/api/chat`, `/api/unload`, `/api/status`, `/api/dashboard`, `/api/runs`.
2. In the **Chat** view you pick a deployment mode — one model split across both GPUs, two independent models (one per GPU), or one model on a single chosen GPU — and click **Load deployment**. That request reuses `server.py`/`orchestrator.py` to spawn the right `llama-server` process(es), exactly like the CLI would.
3. Prompts you send are routed to whichever server(s) are active. In "two models" mode the same prompt goes to both GPUs at once, so you can compare answers, speed, and resource use side by side in real time.
4. Every call records full token accounting (input/output/thinking/cached), timing (prefill tokens/sec, decode tokens/sec, end-to-end latency), and a resource snapshot (process CPU/RAM, dedicated VRAM, shared GPU memory, spill detection) into [chat_runs/chatbot.sqlite3](chat_runs/chatbot.sqlite3).
5. The **Overview** dashboard aggregates that data live: total runs, tokens processed, prompt-processing vs. token-generation throughput per model, latency percentiles, VRAM/RAM peaks, and a 14-day activity chart.
6. The **Run history** view lets you search past runs and open any one to inspect the exact prompt, output, native backend response, and every resource sample collected during that call.

Use this path when you want to interactively load models, compare them head to head, and watch GPU behavior as it happens.

## Architecture

The repo is split into small modules:

- [dual_gpu_setup/config.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/config.py): loads and validates TOML config
- [dual_gpu_setup/lmstudio.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/lmstudio.py): finds LM Studio runtimes, models, and GPU device ids
- [dual_gpu_setup/server.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/server.py): starts and stops `llama-server`, checks health, and warns on VRAM spill
- [dual_gpu_setup/orchestrator.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/orchestrator.py): manages the dual-lane queue and the multi-GPU phase
- [dual_gpu_setup/tasks.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/tasks.py): runs external tasks against each loaded model
- [dual_gpu_setup/gguf.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/gguf.py): dependency-free GGUF metadata reader, used to estimate KV-cache VRAM cost at deploy time
- [dual_gpu_setup/cli.py](F:/Projects/Dual_gpu_setup/dual_gpu_setup/cli.py): CLI entrypoint
- [launch_service.py](F:/Projects/Dual_gpu_setup/launch_service.py): one-shot, on-demand launcher for `app.py` (see [CLIENT_API.md](CLIENT_API.md))
- [example.dual_gpu.toml](F:/Projects/Dual_gpu_setup/example.dual_gpu.toml): sample config

## Testing

```powershell
python -m pip install -e ".[dev]"
pytest
```

37 tests cover the GGUF reader, the KV-cache estimator and context-window sizing logic, the
config loader, `RunStore` (migrations, client registration, dashboard aggregation, pruning),
and the deploy-time capacity check across all three deployment modes. The capacity-check tests
use synthetic GGUF fixtures ([tests/conftest.py](F:/Projects/Dual_gpu_setup/tests/conftest.py)),
not real model files, so the suite runs the same anywhere.

## Chatbot Layer Design

A detailed design is available for adding a persistent chatbot, live monitoring, run history, and UI above this orchestrator:

- [Chatbot architecture and three execution modes](F:/Projects/Dual_gpu_setup/docs/CHATBOT_LAYER.md)
- [Telemetry, token accounting, and storage schema](F:/Projects/Dual_gpu_setup/docs/TELEMETRY_AND_STORAGE.md)
- [Chat, deployment, monitoring, and history UI](F:/Projects/Dual_gpu_setup/docs/UI_SPECIFICATION.md)
- [Implementation phases and validation plan](F:/Projects/Dual_gpu_setup/docs/IMPLEMENTATION_PLAN.md)

These files were the original design proposal for that layer. [app.py](app.py) now implements most of it — the local API, browser UI, SQLite run database, and live resource telemetry described in these documents are working today; see [How It Works](#how-it-works) above for the current mechanics. Token-by-token streaming, cancellation, and richer per-physical-GPU charts remain future work (see [Current Limitations](#current-limitations)).

## Why This Setup Is Useful

In most local-model repos, GPU control gets mixed into task logic. That becomes painful fast:

- one repo picks GPUs by backend index
- another repo hardcodes ports
- a third repo duplicates model-loading logic
- multi-GPU handling is different everywhere
- clean shutdown gets forgotten

This project centralizes all of that.

Your future projects should only need to define:

- which GPUs exist
- which models to run
- which commands to execute once a model server is up

Everything else stays in this repo.

## Prerequisites

Before running this project, make sure you have:

1. Windows machine with 2 GPUs
2. LM Studio installed
3. At least one LM Studio `llama.cpp` runtime installed
4. Models downloaded locally through LM Studio or stored in a known model directory
5. Python 3.11 or newer

This version is optimized for:

- Windows
- LM Studio
- `llama-server.exe`
- local GGUF models

## Installation

From the repo root:

```powershell
python -m pip install -e .
```

If `python` is not available on PATH, use your full Python executable path:

```powershell
C:\path\to\python.exe -m pip install -e .
```

The editable install gives you the `dual-gpu` CLI command.

If you prefer not to install the script entrypoint, you can also run:

```powershell
python -m dual_gpu_setup.cli --config .\example.dual_gpu.toml plan
```

## Quick Start

### Run the chatbot web app

The repository now includes [app.py](F:/Projects/Dual_gpu_setup/app.py), a dependency-free local web application built on Python's standard library.

First validate the configuration and local run database:

```powershell
python .\app.py --config .\example.dual_gpu.toml --check
```

Then start the application:

```powershell
python .\app.py --config .\example.dual_gpu.toml
```

The application listens on localhost by default, which is the recommended mode for coding agents. The terminal prints the Windows-local URL:

```text
Dual GPU Studio is running
  Windows local: http://127.0.0.1:8090
```

The Windows browser opens at `http://127.0.0.1:8090`. Use `--no-browser` to prevent automatic browser launch, or choose another UI port with `--port`:

```powershell
python .\app.py --config .\example.dual_gpu.toml --port 8095 --no-browser
```

To deliberately allow LAN access, pass `--host 0.0.0.0` and follow [local_network_setup.md](local_network_setup.md).

The run database (`chat_runs/chatbot.sqlite3`) keeps every run and resource sample forever by
default — there's no automatic retention policy. To prune old data, pass
`--prune-older-than-days N`; this runs once at startup (before serving, or standalone with
`--check`) and deletes runs and resource samples older than `N` days:

```powershell
python .\app.py --config .\example.dual_gpu.toml --prune-older-than-days 30 --check
```

The app provides:

- one model across both GPUs;
- two independent models, one per GPU, with **Both models · Split view** selected by default so the same prompt runs in parallel and both responses appear side by side;
- one model on one selected GPU;
- automatic discovery of installed LM Studio GGUF models from LM Studio settings, the configured model root, and the default model directory;
- separate selectable entries for every installed quantization, with model search and fit warnings;
- editable context-window and Low/Medium/High reasoning-budget controls;
- persistent prompts, responses, token usage, timings, and resource summaries in `chat_runs/chatbot.sqlite3`;
- persisted deployment records and raw per-run resource samples for later auditing;
- live process CPU, process/system RAM, dedicated VRAM, and shared GPU-memory status;
- an Overview dashboard with token, throughput, latency, model, activity, and resource figures;
- searchable run history with a complete run inspector for prompts, outputs, settings, native backend data, and samples.

The configured model files and LM Studio runtime must exist before loading a deployment. Use the regular `inspect` command first if device mapping is uncertain.

The first runnable UI waits for the local model server to finish a response before displaying it. Token-by-token streaming, cancellation, temperature/power telemetry, and richer per-physical-GPU charts remain later implementation phases described in the design documents.

To access the application from a Mac or another device on the same trusted network, follow [local_network_setup.md](F:/Projects/Dual_gpu_setup/local_network_setup.md).

### 1. Review the example config

Open [example.dual_gpu.toml](F:/Projects/Dual_gpu_setup/example.dual_gpu.toml).

This file defines:

- project metadata
- LM Studio discovery paths
- global serving policy
- 2 GPU lanes
- model list
- tasks to run against each model

### 2. Update the config for your machine

You will usually need to adjust:

- `discovery.lmstudio_home`
- `discovery.models_dir`
- `lanes`
- `models`
- `tasks`

### 3. Check GPU discovery

Run:

```powershell
dual-gpu --config .\example.dual_gpu.toml inspect
```

This command shows:

- which runtime is being used
- which devices LM Studio exposes
- which physical GPU each lane resolves to

Use this first whenever you change drivers, runtimes, GPU hardware, or backend settings.

### 4. Preview scheduling

Run:

```powershell
dual-gpu --config .\example.dual_gpu.toml plan
```

This tells you:

- which models fit on one GPU
- which lane they can run on
- which models are pinned
- which models are marked as multi-GPU
- which models would be skipped

### 5. Dry run

Run:

```powershell
dual-gpu --config .\example.dual_gpu.toml run --dry-run
```

This does not load any models. It only:

- creates a run folder
- generates a suite id
- writes a summary
- shows the planned queue

Use this to confirm config shape before a real run.

### 6. Start a real run

Run:

```powershell
dual-gpu --config .\example.dual_gpu.toml run
```

This will:

1. clean up orphan `llama-server` processes on configured ports
2. start one worker lane per GPU
3. load models according to VRAM fit and pinning rules
4. run configured tasks for each model
5. stop the model server
6. move to the next model
7. run multi-GPU models last

## CLI Commands

### `inspect`

```powershell
dual-gpu --config .\your-config.toml inspect
```

Purpose:

- validate runtime discovery
- confirm device mapping
- verify lane-to-GPU resolution

### `plan`

```powershell
dual-gpu --config .\your-config.toml plan
```

Purpose:

- preview which models fit where
- confirm pinning and multi-GPU classification

### `run`

```powershell
dual-gpu --config .\your-config.toml run
```

Purpose:

- execute the full orchestration flow

### `run --dry-run`

```powershell
dual-gpu --config .\your-config.toml run --dry-run
```

Purpose:

- create only the run metadata without loading models

## How Scheduling Works

> **This section describes the `dual-gpu` CLI's own scheduler** — the one that drains a
> size-sorted queue from both ends when *this repo* runs your models for you (Integration
> Options 1-3). It is not available to a codebase driving the service over HTTP. An external
> client that wants both GPUs busy across a batch of models builds its own two-queue scheduler
> from `/api/clients/check_fit` and `/api/clients/deploy_lane` — see
> [CLIENT_API.md](CLIENT_API.md) step 2c for the sequence.

This setup assumes 2 lanes and asymmetric VRAM is allowed.

Example:

- lane A = 32 GB GPU
- lane B = 16 GB GPU

The queue is optimized so the large GPU takes the largest models it can fit, while the small GPU takes smaller models. That helps avoid a bad tail where the small card becomes idle and the big card is left with all the slow heavy models.

This is one of the most important upgrades carried over from the reference code.

## Architecture Example

Both GPUs are utilized simultaneously by design. This is not two models sharing one GPU. It is two independent `llama-server.exe` processes, each pinned to a different physical GPU, running different models in parallel.

```text
                   ┌─────────────────────────────────┐
                   │   One shared, size-sorted queue │
                   │   (models ordered small→large)  │
                   └─────────────────────────────────┘
                        ↙                        ↘
         pulls LARGEST that fits      pulls SMALLEST that fits
                   ↓                                ↓
        ┌─────────────────────┐          ┌─────────────────────┐
        │   R9700 (32GB)      │          │   9070XT (16GB)     │
        │   ROCm0             │          │   ROCm1             │
        │   Large model       │          │   Small model       │
        │   ex: 20B-35B       │          │   ex: 1B-14B        │
        └─────────────────────┘          └─────────────────────┘
```

Example mental model:

- the 32 GB card starts clearing the biggest models that only it can run
- the 16 GB card keeps draining smaller fast models
- both cards stay busy at the same time
- each lane works independently: `pull -> load -> run tasks -> unload -> pull again`

Why opposite ends instead of both-smallest-first:

- some models only fit on the large GPU
- the small GPU can never help with that backlog
- if both lanes start with the smallest models, they finish the shared pool first
- after that, the small GPU goes idle
- the large GPU is then forced to serialize every remaining big model alone

That is exactly the idle-tail problem this architecture avoids.

Illustrative run example:

- `R9700` loads a large model such as `Qwen3.6-35B-A3B`
- `9070XT` simultaneously runs a small model such as `MiniCPM5-1B-Agentic`
- while the large model is still in its long load/generate cycle, the smaller lane may already finish multiple fast models
- the total wall-clock time drops because both GPUs are continuously doing useful work

This is the key architectural idea behind the repo.

## Multi-GPU Models

Models marked with:

```toml
multi_gpu = true
tensor_split = "36,12"
```

are not run during the normal dual-lane phase.

They are held until both single-GPU lanes finish. Then the orchestrator:

- reserves both GPUs
- resolves both devices
- launches one `llama-server` process using both cards
- runs the configured tasks

Important:

- all lanes used by a multi-GPU model must share the same backend
- multi-GPU models run last
- `tensor_split` should be tuned to your hardware, not guessed randomly

## Config File Guide

The main config file is TOML.

See [example.dual_gpu.toml](F:/Projects/Dual_gpu_setup/example.dual_gpu.toml).

### `[project]`

Example:

```toml
[project]
name = "shared-eval-stack"
suite_prefix = "dualgpu"
log_dir = "./runs"
host = "127.0.0.1"
cleanup_ports_on_start = true
start_timeout_seconds = 900
```

Fields:

- `name`: project label used in run summaries
- `suite_prefix`: prefix for generated run ids
- `log_dir`: where run folders are created
- `host`: bind host for `llama-server`
- `cleanup_ports_on_start`: kill leftover servers on lane ports
- `start_timeout_seconds`: how long to wait for model startup

### `[discovery]`

Example:

```toml
[discovery]
lmstudio_home = "C:/Users/user/.lmstudio"
models_dir = "D:/LLM models"
```

Fields:

- `lmstudio_home`: LM Studio home directory
- `models_dir`: root folder where your GGUF models live

You can also extend this section with backend runtime overrides if needed.

### `[policy]`

Example:

```toml
[policy]
vram_safety_fraction = 0.90
ctx_min = 16384
ctx_mid = 32768
ctx_max = 49152
safety_tokens = 256
reasoning_mode = "off"
reasoning_budget = 0
cache_reuse = 256
flash_attn = "auto"
gpu_layers = 999
parallel = 1
```

Important fields:

- `vram_safety_fraction`: only use this fraction of VRAM for placement decisions
- `ctx_min`, `ctx_mid`, `ctx_max`: context sizes selected from available headroom
- `safety_tokens`: margin added on top of `input_tokens + max_output_tokens` when a deploy
  request reports its real token budget (see `resolve_context_window` in
  `dual_gpu_setup/orchestrator.py`); covers chat-template/tokenizer boundary differences
- `reasoning_mode`: default `--reasoning` mode
- `reasoning_budget`: default thinking-token budget
- `cache_reuse`: llama.cpp prompt cache reuse setting
- `flash_attn`: flash attention mode
- `gpu_layers`: default layers to offload
- `parallel`: default server slot count

### `[[lanes]]`

Example:

```toml
[[lanes]]
key = "r9700"
display = "AMD Radeon AI PRO R9700"
match = "R9700"
vram_gb = 32
port = 8081
backend = "rocm"
```

Fields:

- `key`: internal lane id
- `display`: human-friendly GPU name
- `match`: substring used to match `--list-devices` output
- `vram_gb`: VRAM size used for scheduling
- `port`: local HTTP port for this lane
- `backend`: serving backend, currently `rocm` or `vulkan`

### `[[models]]`

Example:

```toml
[[models]]
name = "gpt-oss-20b-GGUF"
path = "lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
size_gb = 12.11
pin_lane = "r9700"
parallel = 1
```

Common fields:

- `name`: model label
- `path`: absolute path, relative path, or glob under the model root
- `size_gb`: optional explicit size override
- `ctx_size`: optional per-model context override
- `parallel`: optional per-model slot override
- `pin_lane`: force model onto one lane
- `multi_gpu`: whether the model should span both GPUs
- `tensor_split`: split used for multi-GPU models
- `extra_args`: extra `llama-server` arguments
- `threads`: optional CPU thread override
- `ubatch_size`: optional prefill microbatch override
- `tags`: free-form labels for future use

### `[[tasks]]`

Example:

```toml
[[tasks]]
name = "eval"
cwd = "F:/Projects/LLM_evaluator"
command = ["python", "benchmark.py", "--base-url", "{base_url}", "--model", "{model_name}", "--suite-id", "{suite_id}"]
continue_on_error = false
env = { EVAL_RUN_DIR = "{run_dir}", EVAL_CTX = "{actual_ctx}" }
```

Fields:

- `name`: task label
- `cwd`: working directory for the command
- `command`: command string or list
- `shell`: whether to run through shell
- `continue_on_error`: whether orchestration continues if the task fails
- `env`: extra environment variables passed to the task

## Run Output

Each real run creates a folder under your configured `log_dir`.

Inside a run folder you will typically see:

- `summary.json`
- `server_logs/`
- one folder per model
- task logs per model
- `load.json` per model

This gives you:

- overall run tracking
- per-model startup metadata
- per-task output
- server startup logs
- basic VRAM diagnostics

## How Other Applications Can Connect

This is the most important integration concept in the repo.

For two coding agents running concurrently on separate GPU lanes, use the instrumented port-8090
gateway rather than connecting directly to llama-server. The complete Cline + OpenCode setup,
project attribution, worktree workflow, and privacy defaults are in
[CODING_AGENTS.md](CODING_AGENTS.md).

Other applications connect through the OpenAI-compatible HTTP endpoint started by `llama-server`.

During execution, each model is exposed at:

```text
http://127.0.0.1:<lane-port>
```

Examples:

- `http://127.0.0.1:8081`
- `http://127.0.0.1:8082`

### Integration Option 1: Use this repo to launch your app as a task

This is the cleanest pattern for eval projects.

Your external application is launched by this repo after the model is loaded.

Example:

```toml
[[tasks]]
name = "my-eval"
cwd = "F:/Projects/MyEvalProject"
command = ["python", "run_eval.py", "--base-url", "{base_url}", "--model", "{model_name}"]
continue_on_error = false
```

In this mode:

- this repo handles GPU lifecycle
- your app only consumes the endpoint
- no extra coordination layer is needed

### Integration Option 2: Read environment variables inside the task

The task runner injects these variables automatically:

- `DGPU_SUITE_ID`
- `DGPU_BASE_URL`
- `DGPU_MODEL_NAME`
- `DGPU_LANE_KEY`
- `DGPU_LANE_DISPLAY`
- `DGPU_BACKEND`
- `DGPU_DEVICE`
- `DGPU_ACTUAL_CTX`
- `DGPU_RUN_DIR`

That means your app can simply read them from the environment.

Python example:

```python
import os

base_url = os.environ["DGPU_BASE_URL"]
model_name = os.environ["DGPU_MODEL_NAME"]
suite_id = os.environ["DGPU_SUITE_ID"]
```

### Integration Option 3: Pass placeholders directly in the command

You can inject task values directly into the command line:

```toml
command = [
  "python",
  "run_eval.py",
  "--base-url", "{base_url}",
  "--model", "{model_name}",
  "--suite-id", "{suite_id}",
  "--lane", "{lane_key}"
]
```

Available placeholders:

- `{project_name}`
- `{suite_id}`
- `{base_url}`
- `{host}`
- `{port}`
- `{lane_key}`
- `{lane_display}`
- `{backend}`
- `{device}`
- `{model_name}`
- `{model_path}`
- `{model_size_gb}`
- `{actual_ctx}`
- `{run_dir}`
- `{config_dir}`

### Integration Option 4: Connect from Python directly

If your external application speaks the OpenAI-compatible API, it can connect directly.

Example using the official OpenAI Python client against local `llama-server`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8081/v1",
    api_key="not-needed-for-local"
)

response = client.chat.completions.create(
    model="your-loaded-model-name",
    messages=[
        {"role": "user", "content": "Say hello"}
    ],
    temperature=0
)

print(response.choices[0].message.content)
```

Important:

- the model must already be loaded by the orchestrator
- the correct port depends on which lane loaded the model
- for tasks launched by this repo, `{base_url}` is the safest source of truth

### Integration Option 5: Connect from Node.js

```javascript
const response = await fetch("http://127.0.0.1:8081/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer not-needed-for-local"
  },
  body: JSON.stringify({
    model: "your-loaded-model-name",
    messages: [
      { role: "user", content: "Hello" }
    ],
    temperature: 0
  })
});

const data = await response.json();
console.log(data);
```

### Integration Option 6: Client Integration API (register, get a feasibility verdict, and receive shared metrics)

While [app.py](app.py) is running, another local codebase can skip the dashboard entirely and
drive it over HTTP: register a project, ask "can you run model X on GPU Y with context Z" and
get an authoritative load/reject verdict, then run prompts tagged to that client and receive
full token/timing/resource metrics back — with its own copy appended to a file path it chooses.

See [CLIENT_API.md](CLIENT_API.md) for a walkthrough of that flow (`/api/clients/register`,
`/api/clients/check_fit` (read-only: which lane does this model fit, loading nothing),
`/api/clients/deploy`, `/api/clients/deploy_parallel` (load N models across N lanes at once,
auto-picking which model goes where), `/api/clients/deploy_lane` (swap one lane's model, leaving
the other GPU's running), `/api/chat` with `client_id`, `/api/clients/{id}/runs`,
`/api/clients/schema`), or [API_CONTRACT.md](API_CONTRACT.md) for the exhaustive
request/response reference — every field this service needs from a caller and every field it
shares back, for every endpoint it exposes, not just the client-facing ones.

**Running a whole batch of models this way?** Keeping both GPUs busy is the client's job over
HTTP — this service exposes the pieces but schedules nothing on its own. `check_fit` buckets your
model list by which lane each model fits, and `deploy_lane` lets each GPU advance through its own
queue without waiting for the other. [CLIENT_API.md](CLIENT_API.md) step 2c is the full working
sequence, including the ordering rules and the two mistakes that fail silently.

All of that requires `app.py` to already be running. If a client wants to trigger the service
on demand instead of assuming a human already started it, [launch_service.py](launch_service.py)
is a one-shot command (not an always-on daemon) that starts it if needed and exits once it's
healthy:

```powershell
python launch_service.py --config .\example.dual_gpu.toml
```

### Integration Option 7: Connect from curl or Postman

```powershell
curl http://127.0.0.1:8081/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer not-needed-for-local" `
  -d "{\"model\":\"your-loaded-model-name\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

This is useful for:

- smoke tests
- debugging
- validating that a lane is healthy before wiring in a larger app

## Example: Connecting Another Eval Repo

Suppose you have another repo:

```text
F:\Projects\my_new_eval_repo
```

and it already has:

```text
run_eval.py
```

which accepts:

- `--base-url`
- `--model`
- `--suite-id`

Then your config task becomes:

```toml
[[tasks]]
name = "my-new-eval"
cwd = "F:/Projects/my_new_eval_repo"
command = [
  "python",
  "run_eval.py",
  "--base-url", "{base_url}",
  "--model", "{model_name}",
  "--suite-id", "{suite_id}"
]
continue_on_error = false
```

That is enough to connect the repo.

You do not need to copy any GPU code into that application.

## Example: Connecting a Web Backend

If you have a FastAPI backend that should use whichever model this orchestrator loads, the recommended pattern is:

1. launch the backend as a task, or
2. launch a lightweight worker script as a task that calls your backend with the lane information

Example task:

```toml
[[tasks]]
name = "notify-backend"
cwd = "F:/Projects/my_service"
command = [
  "python",
  "notify_backend.py",
  "--base-url", "{base_url}",
  "--model", "{model_name}",
  "--lane", "{lane_key}"
]
continue_on_error = false
```

That lets your service become aware of the currently loaded model without owning the GPU scheduling itself.

## Recommended Integration Pattern

For most teams, the best pattern is:

1. keep this repo responsible for model serving and GPU scheduling
2. keep application repos responsible for business logic and eval logic
3. connect them through the task interface and the local OpenAI-compatible HTTP endpoint

That separation will keep future projects much easier to maintain.

## Troubleshooting

### `inspect` shows the wrong GPU mapping

Check:

- `match` values in `[[lanes]]`
- installed LM Studio runtime
- backend selection
- driver/runtime changes since the last successful run

### A model is very slow

Possible causes:

- model spilled into system RAM
- wrong GPU got selected
- multi-GPU split is suboptimal
- context size is too large for the available headroom

Look for the VRAM spill warning and the server log.

### Port already in use

If `cleanup_ports_on_start = true`, the orchestrator tries to clean orphan `llama-server` processes on lane ports.

If some other application owns that port, change the lane ports in config.

### Task command fails

Check:

- task `cwd`
- task command path
- whether the called script accepts the placeholders you pass
- per-model task log in the run folder

### Multi-GPU model does not start

Check:

- both lanes use the same backend
- `tensor_split` is valid
- model really needs `multi_gpu = true`
- extra args such as `--n-cpu-moe`

## Current Limitations

- expects exactly 2 lanes
- currently optimized for Windows
- current backend support is focused on LM Studio `rocm` and `vulkan`
- one active deployment at a time — a new deploy always replaces whatever was previously
  loaded, whether triggered from the dashboard or the [Client Integration API](CLIENT_API.md)
- no authentication — the HTTP API (`app.py`) is a local-trust-boundary service; don't bind
  `--host` beyond `127.0.0.1`/`localhost` without your own network-level access control
- the KV-cache VRAM feasibility check (see [CHANGELOG.md](CHANGELOG.md)) is best-effort: a
  hybrid state-space/attention architecture, or a sliding-window architecture whose GGUF
  doesn't expose a per-layer pattern, falls back to the older file-size-only check rather
  than a guess it isn't confident in
- `task_label` categories on the Analytics tab have no fixed enum — it's whatever string a
  caller passes, normalized to lowercase/stripped

## Future Extensions

Good next additions would be:

- model tags and filtering in CLI
- task groups by project type
- authentication/access control for the HTTP API, if it ever needs to bind beyond localhost
- automatic (not just opt-in) data retention once the run database's growth actually becomes
  a problem in practice

## Changelog

Notable changes, including the Client Integration API and the telemetry/performance fixes to
the GPU resource sampler, are tracked in [CHANGELOG.md](CHANGELOG.md). For the KV-cache VRAM
feasibility fix and the other gaps closed on 2026-09-27, [optimised.md](optimised.md) walks
through how each bug was actually found and confirmed before it was fixed.

## Summary

Use this repo when you want one shared dual-GPU serving layer for many local-model projects.

Use `inspect` to verify device discovery, `plan` to validate scheduling, and `run` to execute real tasks.

To connect another application, do not copy the GPU code. Point that application at the local `llama-server` endpoint through a configured task and pass `{base_url}` plus the provided environment variables.
