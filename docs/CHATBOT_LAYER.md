# Chatbot Layer for the Dual-GPU Orchestrator

## Status and scope

This document is the implementation specification for adding an interactive chatbot above the existing dual-GPU orchestration code.

The current repository is a batch-oriented orchestrator. It can start `llama-server`, place a model on one GPU or both GPUs, run configured commands, and then stop the server. It does **not** yet provide a permanent chat service, browser UI, conversation database, live metric stream, or request scheduler. Everything described below is a proposed addition unless explicitly identified as existing behavior.

The chatbot layer must support three execution modes:

1. **One large model across both GPUs**: one request is served by one `llama-server` process whose model tensors are split across GPU 0 and GPU 1.
2. **Two models in parallel**: one model is loaded on each GPU and both servers can answer concurrently. A user can chat with either model or send the same prompt to both for comparison.
3. **One GPU at a time**: a model is loaded on one selected GPU while the other GPU stays unused and available for another workload.

The design also records the prompt, response, generation settings, token counts, timing, GPU/CPU/RAM metrics, errors, and model placement for every model invocation.

## Goals

- Expose a persistent OpenAI-compatible chat service instead of only a finite batch run.
- Allow the user to select the execution mode before loading models.
- Make model, GPU, context-window, and reasoning controls visible in the UI.
- Stream generated text and live host/GPU metrics to the browser.
- Preserve every run so configurations and model performance can be compared later.
- Keep GPU scheduling in this repository; client applications should not need to understand devices or `llama-server` flags.
- Fail safely when a requested model, context size, or pair of models does not fit.

## Non-goals for the first version

- Training or fine-tuning models.
- Combining the logits of two independent models into one answer.
- Automatically judging which of two model answers is best.
- Hiding model/backend limitations. If a backend does not report thinking tokens or separate prefill timing, the stored value must be `null` with a reason, not an estimate presented as measured data.
- Exposing the control plane outside the local machine by default.

## Proposed architecture

```text
Browser UI
   |  HTTP + SSE/WebSocket
   v
Chat API / Control Plane
   |-- Conversation service
   |-- Run manager and request scheduler
   |-- Capability and validation service
   |-- Metrics collector
   |-- SQLite metadata store
   |-- JSONL or SQLite time-series samples
   |
   +--> Lane A adapter --> llama-server :8081 --> GPU A
   +--> Lane B adapter --> llama-server :8082 --> GPU B
   +--> Multi-GPU adapter --> one llama-server --> GPU A + GPU B
```

Recommended first implementation:

- **Backend**: FastAPI, because it supports typed REST endpoints, Server-Sent Events/WebSockets, background jobs, and Python integration with the current modules.
- **Frontend**: React with TypeScript, or a server-rendered UI if minimizing dependencies is more important than rich live charts.
- **Persistence**: SQLite in WAL mode for conversations, messages, runs, and summary metrics. Store dense metric samples in a separate `metric_samples` table initially; move to Parquet only if the database becomes too large.
- **Streaming**: Server-Sent Events for token output and metric updates. WebSockets are only required if the client must send interactive control messages over the same connection.
- **Model servers**: continue using the current `LlamaServerProcess` abstraction, extended with long-lived lifecycle and capability discovery.

## Core components

### 1. Control plane

The control plane owns configuration changes and model lifecycle. It must be the only component allowed to start or stop a model server.

Responsibilities:

- List GPUs, models, model metadata, and backend capabilities.
- Validate a requested execution profile before changing running servers.
- Serialize conflicting load/unload operations with a process-wide lock.
- Maintain a state machine for each lane: `empty`, `loading`, `ready`, `busy`, `unloading`, `error`.
- Maintain a system state: `idle`, `switching`, `single_large_model`, `parallel_models`, `single_gpu`.
- Reject new requests with HTTP `409` while an incompatible mode transition is in progress.
- Drain or cancel active requests according to an explicit user choice before unloading a model.
- Restore the last known configuration after an application restart only when `restore_last_profile` is enabled.

### 2. Chat service

The chat service accepts messages, assembles the prompt, applies the chosen generation controls, forwards the request to the correct local endpoint, and streams events back to the UI.

It must:

- Keep conversations and branches independent from server lifecycle.
- Record the exact message snapshot used for every request.
- Count or obtain tokens using the tokenizer associated with the loaded model.
- enforce the context limit before submitting the request.
- Record time to first token, inter-token latency, prefill time, decode time, and total latency.
- Preserve partial output and a terminal status if the user cancels generation or the server fails.
- Assign an immutable `run_id` to each model invocation. In compare mode, use a shared `comparison_id` plus a separate `run_id` for each model.

### 3. Scheduler

The scheduler operates on GPU reservations rather than sending requests directly to arbitrary ports.

Reservation rules:

- A multi-GPU server reserves both GPU IDs exclusively.
- A server pinned to one GPU reserves only that GPU.
- Two single-GPU servers may run simultaneously only if they reserve different GPUs and pass the memory-fit check.
- A one-GPU profile never moves to the other GPU unless the user changes the profile or explicitly enables failover.
- A mode change is atomic from the UI's point of view: either the requested profile becomes healthy, or the service reports failure and attempts to restore the previous healthy profile.
- A run records the actual devices used, not only the requested devices.

### 4. Backend adapter

Do not scatter `llama.cpp`-specific parsing throughout the application. Add an adapter interface with methods similar to:

```python
class InferenceBackend:
    def capabilities(self) -> BackendCapabilities: ...
    def start(self, deployment: DeploymentSpec) -> ServerHandle: ...
    def stop(self, handle: ServerHandle) -> None: ...
    def health(self, handle: ServerHandle) -> Health: ...
    def count_tokens(self, handle: ServerHandle, messages: list[dict]) -> int: ...
    def stream_chat(self, handle: ServerHandle, request: ChatRequest): ...
    def read_native_metrics(self, handle: ServerHandle) -> NativeMetrics: ...
```

The adapter should detect features at startup because LM Studio's bundled `llama-server` version and flags may change. The UI must use the reported capabilities to enable, disable, or label controls.

## Execution modes

### Mode A: one large model across both GPUs

Use this mode when a model cannot fit on either GPU by itself, or when the user explicitly wants a single model to use both devices.

```text
Prompt --> Chat API --> one llama-server --> GPU A + GPU B --> one response
```

Required configuration:

- Model ID/path.
- Ordered GPU IDs.
- Tensor split, either `auto` or an explicit ratio such as `36,12`.
- Context window.
- Reasoning mode and budget.
- GPU layers, batch size, micro-batch size, CPU thread count, cache type, and flash-attention setting when supported.

Behavior:

- Stop or drain both single-GPU lane servers before loading the large model.
- Verify that both lanes use a compatible backend.
- Reserve both GPUs until the server is unloaded.
- Start a single server, using the port of the primary lane or a dedicated multi-GPU port.
- Treat tensor split as placement, not model parallel request execution. One chat request still produces one response.
- Store per-GPU memory and utilization separately, plus aggregate values for the run.

Fit validation should include model weights, KV cache at the requested context, compute buffers, runtime overhead, and the configured safety margin. File size alone is not an adequate fit test.

### Mode B: two models, one per GPU, running in parallel

Use this mode for side-by-side comparison, independent conversations, throughput testing, or serving two specialized models.

```text
                                      +--> model A / GPU A --> response A
Prompt or independent requests --> API|
                                      +--> model B / GPU B --> response B
```

The UI must support two routing choices:

- **Direct chat**: the conversation selects model A or model B; only that server receives the message.
- **Compare**: the same immutable prompt snapshot and comparable generation settings are fanned out to both servers concurrently. Results appear side by side.

Required configuration per lane:

- GPU and model selection.
- Context window.
- Reasoning profile/budget.
- Generation controls.
- Optional lane-specific system prompt.

Behavior:

- Start one process per GPU and pin each process to exactly one physical device.
- Give every server a unique port.
- Do not merge conversation history between the two models unless the user explicitly copies or branches it.
- In compare mode, start both calls from the same prepared message snapshot and record the difference between their start timestamps.
- Record one run per model, not one combined run. Link them with `comparison_id`.
- If one request fails, allow the other to finish and show both terminal states.
- Display wall-clock latency for each model and comparison wall time (`max(finish) - min(start)`).

### Mode C: one GPU at a time

Use this mode to reserve the other GPU for graphics, training, another application, power savings, or controlled benchmarking.

```text
Prompt --> Chat API --> model --> selected GPU
Other GPU: unreserved by this service
```

Required configuration:

- Selected GPU.
- Model.
- Context window.
- Reasoning profile/budget.
- Generation controls.

Behavior:

- Start only the selected lane server.
- Show the non-selected lane as `unused`, not `offline`, unless it is actually unavailable.
- Never silently spill the model onto the second GPU.
- Reject a model that does not fit the selected GPU, with suggestions such as smaller context, smaller quantization, fewer GPU layers, or multi-GPU mode.
- Optional failover must be off by default because moving a model changes performance and invalidates controlled comparisons.

## Request and conversation lifecycle

Each interaction follows this sequence:

1. The user selects a conversation and model target, then submits a message.
2. The API writes the user message and creates a `run` with status `queued`.
3. The context builder creates an immutable prompt snapshot and calculates input tokens.
4. Validation checks model readiness, requested output allowance, context limit, reasoning budget, and cancellation state.
5. The scheduler reserves the target deployment and sets the run to `running`.
6. A resource sampler begins collecting process, CPU, RAM, and per-GPU samples.
7. The backend streams text and native timing/usage fields. The API emits normalized events to the UI.
8. The assistant message is updated in batches to avoid a database write for every token.
9. On completion, cancellation, or failure, the sampler takes a final sample and aggregates the run metrics.
10. The API commits the terminal run status and emits a final event containing authoritative totals.

Terminal statuses are `completed`, `cancelled`, `failed`, and `timed_out`. A process crash must never leave a run permanently marked `running`; startup recovery should mark abandoned runs as `interrupted`.

## Context-window controls

The UI must show:

- Model maximum context reported by metadata (`n_ctx_train` or equivalent).
- Server context allocated at load time.
- Tokens currently occupied by system prompt, history, tools, attachments, and the new user message.
- Reserved output tokens.
- Remaining usable tokens.
- KV-cache memory estimate and whether it fits.

Use this invariant before generation:

```text
input_tokens + max_output_tokens + safety_tokens <= effective_context_window
```

Where:

```text
effective_context_window = min(model_trained_limit, server_allocated_limit)
```

`safety_tokens` covers chat-template and tokenizer boundary differences. It should be configurable and visible in advanced settings.

Context presets may be `4K`, `8K`, `16K`, `32K`, `48K`, and `Custom`, but the UI must filter them using the model and deployment limits. Changing server-allocated context normally requires a model reload; the UI must warn the user and show the estimated KV-cache change before applying it.

When a conversation no longer fits, offer explicit policies:

- Reject and ask the user to reduce history.
- Drop oldest turns while retaining the system prompt.
- Use a stored summary plus recent turns.
- Branch into a new conversation.

The selected policy and any omitted message IDs must be stored with the run.

## Reasoning and thinking controls

Reasoning support varies by model and backend. A model profile should declare:

- whether reasoning is supported;
- supported reasoning modes or formats;
- whether a numeric budget is supported;
- minimum, default, and maximum thinking budget;
- whether thinking tokens are returned separately from visible output tokens;
- whether reasoning text may be displayed and stored.

The UI presents model-specific presets rather than assuming universal values:

| Preset | Meaning | Example mapping |
|---|---|---|
| Off | Disable reasoning when supported | mode `off`, budget `0` |
| Low | Fast, shallow reasoning | model-profile value, for example `1,024` |
| Medium | Balanced reasoning | model-profile value, for example `4,096` |
| High | Maximum configured reasoning | model-profile value, for example `8,192` |
| Custom | User-selected numeric budget | validated against the model profile |

The numbers above are illustrative and must not become global defaults. `Low`, `Medium`, and `High` map through the selected model's capability profile.

Budget accounting must be explicit:

```text
max_output_tokens = visible_output_allowance + thinking_budget
```

Use the backend/model's real accounting rule if it differs. The UI should show how the selected budget reduces the room available for visible output. If the backend reports only total completion tokens, store `thinking_tokens = null` and `thinking_tokens_reason = "not_reported_by_backend"`.

## Generation controls

The main composer should expose only frequently changed controls:

- Model target or compare target.
- Context preset.
- Reasoning preset.
- Maximum visible output tokens.
- Temperature.
- Stop/cancel generation.

An advanced drawer can expose:

- `top_p`, `top_k`, `min_p`, repeat penalty, repeat window, seed, and stop sequences.
- Streaming on/off.
- Chat template and system prompt profile.
- GPU layers, CPU threads, batch/ubatch size, cache type/quantization, flash attention, tensor split, and parallel slots.

Controls that require a server reload must be marked **Reload required**. Request-only controls should apply to the next message without restarting the model.

## Proposed REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/system` | Service version, state, active profile, and health |
| `GET` | `/api/v1/gpus` | GPU inventory and current samples |
| `GET` | `/api/v1/models` | Model catalog, metadata, capabilities, and fit estimates |
| `POST` | `/api/v1/deployments/validate` | Validate a proposed execution profile without changing state |
| `PUT` | `/api/v1/deployments/active` | Atomically activate an execution profile |
| `DELETE` | `/api/v1/deployments/active` | Drain and unload all managed model servers |
| `GET` | `/api/v1/conversations` | List conversations |
| `POST` | `/api/v1/conversations` | Create a conversation |
| `GET` | `/api/v1/conversations/{id}` | Read messages and branches |
| `POST` | `/api/v1/conversations/{id}/messages` | Start one direct or compare generation |
| `POST` | `/api/v1/runs/{id}/cancel` | Cancel an active generation |
| `GET` | `/api/v1/runs/{id}` | Read configuration, output, and summary metrics |
| `GET` | `/api/v1/runs/{id}/events` | Stream tokens, state, and metrics using SSE |
| `GET` | `/api/v1/runs` | Filter and compare historical runs |
| `GET` | `/api/v1/runs/{id}/metrics` | Read sampled time-series metrics |

Example deployment request for two parallel models:

```json
{
  "mode": "parallel_models",
  "lanes": [
    {
      "lane_key": "r9700",
      "model_id": "qwen3-32b-q4",
      "context_window": 32768,
      "reasoning_profile": "high"
    },
    {
      "lane_key": "9070xt",
      "model_id": "small-agent-model",
      "context_window": 16384,
      "reasoning_profile": "low"
    }
  ]
}
```

Example normalized SSE stream:

```text
event: run.state
data: {"run_id":"...","status":"running"}

event: message.delta
data: {"run_id":"...","text":"Hello"}

event: metric.sample
data: {"run_id":"...","gpu":[{"gpu_id":"...","utilization_pct":91.0,"vram_used_bytes":22162031232}],"process_rss_bytes":1453326336}

event: run.completed
data: {"run_id":"...","input_tokens":318,"output_tokens":692,"total_tokens":1010,"tokens_per_second":38.4}
```

## Configuration extension

Keep static defaults in TOML and store user-created profiles in the database. A future configuration can add sections similar to:

```toml
[chat]
host = "127.0.0.1"
port = 8090
database = "./data/chatbot.sqlite3"
metric_sample_interval_ms = 500
restore_last_profile = false
max_queue_depth = 32

[chat.context]
presets = [4096, 8192, 16384, 32768, 49152]
safety_tokens = 128
default_overflow_policy = "reject"

[[model_profiles]]
model = "gpt-oss-20b-GGUF"
reasoning_supported = true
reasoning_mode = "auto"
thinking_tokens_reported = true
thinking_low = 1024
thinking_medium = 4096
thinking_high = 8192
thinking_max = 16384
default_context = 16384
max_context = 32768
```

Model capability values must be verified for the exact model and backend version. The application should show their source: model metadata, probed backend, configuration override, or unknown.

## Security and local operation

- Bind to `127.0.0.1` by default.
- Require an explicit configuration change to listen on a LAN interface.
- If exposed beyond localhost, require authentication, CSRF protection, origin checks, TLS at the reverse proxy, and rate limits.
- Redact authorization headers, API keys, and secrets from logs.
- Treat prompts and model outputs as private data. Provide retention and delete controls.
- Validate model paths against configured model roots.
- Never accept arbitrary command-line flags from an unauthenticated API client.
- Record configuration changes in an audit table without storing secrets.

## Failure handling

| Failure | Expected behavior |
|---|---|
| Model does not fit | Reject before load; show estimated requirement and safe alternatives |
| Server fails during load | Mark deployment failed, capture log path/error, release reservation, restore previous healthy profile when possible |
| One compare target fails | Let the other target complete; display independent states |
| GPU metric source unavailable | Continue inference; store metric as `null` with source/error metadata |
| Client disconnects | Continue or cancel according to the request policy; never lose the run record |
| User cancels | Stop generation, preserve partial output, mark run `cancelled` |
| Application restarts | Mark orphaned active runs `interrupted`; reconcile or stop orphan model servers safely |
| Context overflow | Reject before inference with a token breakdown and available recovery policies |

## Acceptance criteria

- All three execution modes can be activated and clearly identified in the UI.
- A multi-GPU model exclusively reserves both configured GPUs.
- Two single-GPU models can stream answers at the same time without crossing device assignments.
- Single-GPU mode starts no process on the unselected GPU.
- The server rejects unsafe model/context combinations before attempting a load.
- Every invocation produces one persisted run with input, output, settings, status, timing, token accounting, placement, and resource summaries.
- Compare mode produces two linked runs and continues if only one fails.
- Live GPU, CPU, RAM, and token-rate data appears while a run is active.
- Unknown metrics are stored and displayed as unavailable, never as zero.
- Reload-required settings are visually distinct from per-request settings.
- Cancelling generation preserves the partial answer and final metric sample.

## Suggested implementation sequence

1. Extract the current process lifecycle into reusable deployment and lane managers.
2. Add SQLite migrations and repository classes for conversations, messages, deployments, and runs.
3. Add a metrics collector and normalized backend timing parser.
4. Add the FastAPI control, conversation, run, and event-stream endpoints.
5. Implement single-GPU mode end to end.
6. Implement two-model direct and compare routing.
7. Implement exclusive multi-GPU deployment and tensor-split validation.
8. Build the dashboard and chat UI described in [UI_SPECIFICATION.md](UI_SPECIFICATION.md).
9. Add crash recovery, retention, exports, and benchmark comparison views.
10. Run the validation plan in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

