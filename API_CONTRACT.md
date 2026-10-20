# API Contract

> For what the app is, its features, limits and what to expect, see [app_overview.md](app_overview.md), the
> source of truth for scope. This document is the authority for exact request and response shapes.

Formal request/response contract for every HTTP endpoint [app.py](app.py) exposes. For a
narrative walkthrough of the client-integration flow specifically, see
[CLIENT_API.md](CLIENT_API.md); this document is the exhaustive reference — every field this
service needs from a caller, and every field it will share back — for **all** endpoints,
not just the client ones.

## Starting the service

Every endpoint below requires `app.py` to already be running — there is no HTTP endpoint
that starts it, since a stopped service has nothing listening to receive that request. A
client that wants to trigger the service on demand (rather than assuming a human already
started it) runs [launch_service.py](launch_service.py) as a one-shot command, not an
always-on daemon: `python launch_service.py --config <path>`. It's idempotent (a no-op if
already running) and exits once the service is confirmed healthy or has failed to start —
it never itself stays resident. See [CLIENT_API.md](CLIENT_API.md#0-make-sure-the-service-is-actually-running)
for the full example. Run from a terminal it also opens the UI in your browser (a program
calling it gets no browser; `--open-browser` / `--no-browser` override). Once running, the service
can be restarted or stopped over HTTP (see [Service control](#service-control)).
When `--config` is omitted, Windows uses `example.dual_gpu.toml` and Linux uses
`example.dual_gpu.ubuntu.toml`; production clients should normally pass their edited config explicitly.

## Conventions

- Base URL: `http://127.0.0.1:<port>` (default port `8090`; see `--port`/`--host` in
  [README.md](README.md#quick-start)).
- All request/response bodies are JSON. `Content-Type: application/json` on requests with a
  body; request bodies are capped at 10 MB.
- **No authentication.** This is a local-trust-boundary service — anyone who can reach the
  bound host/port can call any endpoint. Do not bind `--host` beyond `127.0.0.1`/`localhost`
  without your own network-level access control.
- Error shape is the same everywhere: `{"error": "<message>"}` with an HTTP status code
  (`400` bad input, `404` not found, `409` a deployment change or service action is already in
  progress, `413` body too large, `415` a control endpoint needs `Content-Type: application/json`,
  `500` unexpected failure, `501` service control is not available in this process, `503` the model
  cannot be served right now: all GPUs busy or not enough free VRAM, so retry later).
- **Release what you load.** The service keeps a model on the GPU until something unloads it. An
  application that loaded a model for a task must release it when the task is done; see
  [Releasing the GPUs when a task is done](#releasing-the-gpus-when-a-task-is-done).
- Fields marked **required** cause a `400` with a descriptive message if missing or invalid.
  Fields marked **optional** have the stated default when omitted.
- New runs are attributed to the host OS. `host_os` always contains `system`, `release`,
  `version`, and `machine`; Linux also supplies `distribution`, `distribution_id`, and
  `distribution_version` when `/etc/os-release` is available.

---

## GET `/api/bootstrap`

Full snapshot used to render the dashboard on load. No input.

**Shares:**
```jsonc
{
  "app_version": "0.1.0",
  "config_path": "/path/to/dual_gpu.toml",
  "catalog": { /* see GET /api/models/refresh below for shape */ },
  "status": { /* see GET /api/status below for shape */ },
  "runs": [ /* up to 20 most recent runs, see GET /api/runs for row shape */ ]
}
```

## GET `/api/status`

Current deployment state and the latest resource sample. No input.

**Shares:**
```jsonc
{
  "state": "idle | switching | ready | error",
  "mode": "idle | single_large_model | parallel_models | single_gpu",
  "error": "string | null",
  "profile": { /* the deploy request that produced the current state, or null */ },
  "deployment_id": "uuid | null",
  // llama-server processes on a lane port that this service does not manage (e.g. left behind by an
  // instance that was killed). They still hold VRAM and RAM. Scanned at most every 10 s, and always
  // empty while "transitioning" is true (a server that is starting or draining is the service's own).
  "orphans": [{"lane": "<lane_key>", "port": "int", "pid": "int"}],
  // true while a load, swap or unload is running (an unload waits up to drain_timeout_seconds for requests)
  "transitioning": "bool",
  "servers": [
    {
      "target": "primary | <lane_key>",
      "model": "string",
      "lane_keys": ["<lane_key>", "..."],
      "device": "string (backend device id, e.g. 'rocm:0' or 'rocm:0,rocm:1')",
      "base_url": "http://127.0.0.1:<lane_port>",
      "pid": "int | null",
      "alive": "bool",
      "context_window": "int",
      "reasoning_budget": "int",
      // the llama-server build serving this target; also stored in each run's configuration
      "runtime": {"binary": "path", "source": "llama.cpp | lmstudio", "version": "string", "build": "int | null", "commit": "string"} | null,
      // residency guard (Windows): VRAM resident at the last check, times found paged out to RAM, last re-warm
      "inflight": "int (requests being answered now; Unload waits for them)",
      "resident_gb": "float | null",
      "evictions": "int",
      "rewarmed_at": "ISO-8601 timestamp | null"
    }
  ],
  "metrics": {
    "sampled_at": "ISO-8601 timestamp",
    "monotonic": "float (internal clock, not wall time)",
    "system": {
      "cpu_pct": "float | null",
      "ram_total_bytes": "int | null",
      "ram_available_bytes": "int | null",
      "ram_used_bytes": "int | null",
      "ram_used_pct": "float | null",
      "pagefile_used_bytes": "int | null",
      "pagefile_total_bytes": "int | null"
    },
    "servers": [
      {
        "target": "string", "model": "string", "lane_keys": ["..."],
        "pid": "int", "alive": "bool",
        "cpu_pct": "float | null", "rss_bytes": "int | null", "private_bytes": "int | null",
        "peak_rss_bytes": "int | null", "peak_private_bytes": "int | null",
        "page_fault_count": "int | null (cumulative since process start)",
        "gpu_utilization_pct": "float | null",
        "dedicated_vram_bytes": "int | null", "shared_gpu_memory_bytes": "int | null"
      }
    ]
  }
}
```

## GET `/api/dashboard`

Coding runs also expose `reasoning_levels`, grouped by model and effective native effort.
Each group reports run count, `thinking_tokens_reported_runs`, `avg_thinking_tokens`,
`thinking_tokens_estimated_runs`, `avg_thinking_tokens_estimated`, `avg_thinking_characters`,
`avg_reasoning_duration_ms`, decode throughput, latency, `avg_draft_acceptance_rate` (speculative
decoding), and tool-call validity (`tool_calls_total`, `tool_calls_malformed`,
`avg_tool_call_valid_rate`). Exact backend counts and the
character-based estimate are aggregated separately; missing thinking counts are excluded from
both averages, never treated as zero. `recent_runs` here contains lightweight metric summaries; use
`GET /api/runs/<id>` for full prompts, outputs, and configuration. Aggregation includes the
complete matching history while excluding large content bodies from its database read.

Coding run `configuration.reasoning` records `effort`, `effective_effort`,
`budget_policy: "model_controlled"`, `budget_tokens: null`, `server_budget_tokens: -1`,
and any explicit `output_limit_tokens`. Streamed coding timing additionally includes
`reasoning_duration_ms` (observed first-to-last reasoning chunk span), its source,
and `time_to_first_visible_token_ms`. Actual reasoning-token counts are only reported
when supplied by the backend; chunk counts are not token counts.

Aggregated analytics across the complete stored history. Optional query parameters:
`project_id=<uuid>`, `workload_kind=evaluation|coding_agent`, and a date range
`start_date=YYYY-MM-DD` / `end_date=YYYY-MM-DD` (both inclusive; either may be given alone;
a malformed date returns 400). When a date range is set, every summary, per-model, per-level,
per-project, tool-event, and `daily` aggregate is restricted to it, and `daily` returns every
day in range rather than the default last 14. The applied range echoes back as `filters`.

**Shares:**
```jsonc
{
  "generated_at": "ISO-8601 timestamp",
  "summary": {
    "total_runs": "int", "completed_runs": "int", "failed_runs": "int",
    "success_rate": "float | null (percent)",
    "models_used": "int",
    "input_tokens": "int", "output_tokens": "int", "thinking_tokens": "int (exact backend counts only)",
    "thinking_tokens_reported_runs": "int", "thinking_tokens_estimated": "int (~4 chars/token)",
    "thinking_tokens_estimated_runs": "int",
    "visible_output_tokens": "int", "total_tokens": "int",
    "avg_tokens_per_second": "float | null (per-request mean)", "aggregate_tokens_per_second": "float | null",
    "wall_clock_tokens_per_second": "float | null (output tokens / decode wall-clock, overlapping requests counted once)",
    "avg_concurrent_requests": "float | null (requests already in flight on the server when a run started)",
    "avg_prefill_tokens_per_second": "float | null",
    "avg_latency_ms": "float | null", "p95_latency_ms": "float | null",
    "avg_ttft_ms": "float | null",
    "peak_process_ram_bytes": "int", "peak_vram_bytes": "int",
    "peak_gpu_utilization_pct": "float", "avg_gpu_utilization_pct": "float | null",
    "peak_shared_gpu_memory_bytes": "int", "spill_runs": "int"
  },
  "models": [
    {
      "model": "string", "runs": "int", "success_rate": "float",
      "input_tokens": "int", "output_tokens": "int", "thinking_tokens": "int",
      "avg_tokens_per_second": "float | null", "avg_prefill_tokens_per_second": "float | null",
      "wall_clock_tokens_per_second": "float | null", "avg_concurrent_requests": "float | null",
      "avg_latency_ms": "float | null", "avg_gpu_utilization_pct": "float | null",
      "peak_vram_bytes": "int"
    }
  ],
  "gpus": [ /* same summary fields, plus gpu: lane key */ ],
  "projects": [ /* same summary fields, plus project_id and project name */ ],
  "agents": [ /* same summary fields, plus agent_role */ ],
  "backends": [ /* same summary fields, plus backend: "llama.cpp | LM Studio" (read-time label, see GET /api/runs) */ ],
  "operating_systems": [ /* same summary fields, plus os: "Windows | Linux" (read-time label, see GET /api/runs) */ ],
  "tool_summary": { "events": "int", "failed": "int", "success_rate": "float | null" },
  "categories": [
    {
      "category": "string (a task_label value some /api/chat call used, e.g. 'resume_extraction')",
      "runs": "int", "success_rate": "float", "input_tokens": "int", "output_tokens": "int", "thinking_tokens": "int",
      "avg_tokens_per_second": "float | null", "avg_prefill_tokens_per_second": "float | null",
      "wall_clock_tokens_per_second": "float | null", "avg_concurrent_requests": "float | null",
      "avg_latency_ms": "float | null", "avg_gpu_utilization_pct": "float | null",
      "peak_vram_bytes": "int"
    }
    // grouped by task_label across every model; runs that never set task_label are excluded
    // here (they're still counted in `models` and `summary`)
  ],
  "category_models": [
    {
      "category": "string", "model": "string",
      "runs": "int", "success_rate": "float", "input_tokens": "int", "output_tokens": "int", "thinking_tokens": "int",
      "avg_tokens_per_second": "float | null", "avg_prefill_tokens_per_second": "float | null",
      "wall_clock_tokens_per_second": "float | null", "avg_concurrent_requests": "float | null",
      "avg_latency_ms": "float | null", "avg_gpu_utilization_pct": "float | null",
      "peak_vram_bytes": "int"
    }
    // one row per (task_label, model_name) pair actually seen; this is what the dashboard's
    // Analytics tab filters by category and/or model
  ],
  "daily": [ { "date": "YYYY-MM-DD", "runs": "int", "tokens": "int" } /* last 14 days */ ],
  "recent_runs": [ /* 20 most recent, GET /api/runs row shape but slimmed: no request, output_text,
                      reasoning_text or backend_response; configuration holds only
                      {reasoning, runtime: {source, build}} */ ]
}
```

## GET `/api/runs`

**Needs:** optional query parameters `limit` (1-500), `offset`, `project_id`, and
`workload_kind`. Defaults to the 100 most recent runs.

**Shares:** `{"runs": [<run row>, ...], "total": int, "limit": int, "offset": int}` — each run row:
```jsonc
{
  "id": "uuid", "comparison_id": "uuid | null", "deployment_id": "uuid | null",
  "created_at": "ISO-8601", "finished_at": "ISO-8601 | null",
  "status": "running | completed | failed",
  "mode": "single_large_model | parallel_models | single_gpu",
  "target": "string", "model_name": "string", "model_path": "string | null",
  "lane_keys": ["..."], "device": "string | null",
  "context_window": "int | null", "reasoning_budget": "int | null",
  "request": { /* the exact chat-completions payload sent to llama-server */ },
  "configuration": { /* full server/runtime configuration snapshot; includes host_os */ },
  "output_text": "string | null", "reasoning_text": "string | null",
  "finish_reason": "string | null",
  "usage": { /* see GET /api/clients/schema */ },
  "timing": { /* see GET /api/clients/schema */ },
  "resources": { /* see GET /api/clients/schema */ },
  "error_text": "string | null",
  "backend_response": null,
  "client_id": "uuid | null (set only if this run was made via /api/chat with client_id)",
  "task_label": "string | null (set only if this run was made via /api/chat with task_label)",
  "project_id": "uuid | null", "agent_session_id": "uuid | null",
  "coding_task_id": "uuid | null", "agent_role": "string | null",
  "workload_kind": "evaluation | coding_agent",
  "backend": "llama.cpp | LM Studio (derived at read time from configuration.runtime.source; older runs without it fall back by created_at)",
  "os": "Windows | Linux (derived at read time from host_os.system; runs without host_os report Windows)",
  "host_os": {
    "system": "Windows | Linux", "release": "...", "version": "...", "machine": "...",
    "distribution": "Ubuntu 24.04 LTS (Linux only, when available)",
    "distribution_id": "ubuntu (Linux only, when available)",
    "distribution_version": "24.04 (Linux only, when available)"
  }
}
```

## GET `/api/runs/{run_id}`

**Needs:** `run_id` as a path segment.

**Shares:** the same run row as above, plus:
```jsonc
"metric_samples": [
  { "sampled_at": "ISO-8601", "offset_ms": "float (since run start)", "sample": { /* raw sampler tick, same shape as /api/status metrics */ } }
]
```
`404 {"error": "Run not found"}` if `run_id` doesn't exist.

---

## GET `/api/chat-history`

The full-conversation archive (a separate SQLite file, opt-in via `project.store_chat_history`).
Disabled by default; when off, returns `{"enabled": false, "turns": []}`.

**Query params:** `project_id`, `start_date`/`end_date` (YYYY-MM-DD, inclusive; 400 on bad
format), `limit` (default 100, capped 500), `offset`, and `content=1|true|yes` to include full
bodies (omitted by default).

**Shares:**
```jsonc
{
  "enabled": true,
  "turns": [
    {
      "id": "run-uuid", "created_at": "ISO-8601", "project_id": "…", "client_id": "…",
      "agent_role": "…", "model_name": "…", "target": "r9700", "workload_kind": "coding_agent",
      "input_tokens": "int|null", "cached_input_tokens": "int|null", "output_tokens": "int|null",
      "thinking_tokens": "int|null", "total_tokens": "int|null",
      "tokens_per_second": "float|null", "latency_ms": "float|null", "finish_reason": "…",
      // only when content=1:
      "messages": [ /* full input messages */ ], "tools": [ /* tool schemas */ ],
      "output_text": "…", "reasoning_text": "…|null", "tool_calls": [ /* parsed */ ]
    }
  ],
  "limit": 100, "offset": 0
}
```
Content is archived regardless of `X-DGPU-Capture-Content` (which only governs the metrics DB).
Per-request opt-out: `X-DGPU-No-History: true`, or `"store_history": false` in an `/api/chat` body.

---

## POST `/api/deploy`

Load a deployment. Reused internally by `/api/clients/deploy` (below) — this is the endpoint
the dashboard's "Load deployment" button calls, and it **stops whatever was previously
loaded first**: there is one active deployment at a time.

**Needs** (`mode` selects the shape of the rest of the body):

| mode | body |
|---|---|
| `single_large_model` | `{"mode": "single_large_model", "model": {"model": "<name or catalog id>", "context_window"?: int, "input_tokens"?: int, "max_output_tokens"?: int, "reasoning_budget"?: int, "tensor_split"?: "36,12"}}` — spans every configured lane in one process. |
| `single_gpu` | `{"mode": "single_gpu", "model": {"model": "<name>", "lane": "<lane_key>", "context_window"?: int, "input_tokens"?: int, "max_output_tokens"?: int, "reasoning_budget"?: int}}` |
| `parallel_models` | `{"mode": "parallel_models", "models": [{"model": "<name>", "lane": "<lane_key>", ...}, {...}]}` — exactly 2 entries, 2 distinct lanes. |

- `model` (the name/id field): **required**, must match a `catalog.models[].id` or `.name` from
  `/api/bootstrap`.
- `lane` (`single_gpu`/`parallel_models`): **required**, must match a `catalog.lanes[].key`.
- `context_window`: **optional**; defaults to the model's configured `ctx_size` or a
  policy-derived value based on available VRAM headroom. Minimum `256`.
- `input_tokens` / `max_output_tokens`: **optional, but must be given together**. When both are
  present, the resolved context window is bumped up to guarantee
  `input_tokens + max_output_tokens + policy.safety_tokens <= context_window`, even above whatever
  `context_window` was requested or the policy default would have picked. If that requirement
  exceeds `policy.ctx_max`, the deploy is rejected with `400` instead of silently loading a server
  that will fail requests mid-run with `exceed_context_size_error`.
- `reasoning_budget`: **optional**; defaults to the model's configured value or the policy
  default. Must be `-1` (unlimited) or `>= 0` for evaluation deployments.
- `workload_kind: "coding_agent"`: optional on the deployment or individual model entries.
  Uses `reasoning_mode: "auto"` and `reasoning_budget: -1`, ignoring legacy numeric presets.
  `reasoning_effort` selects a native level or `default`; available choices and the declared
  default are exposed in each catalog model's `coding_reasoning` object. Native efforts
  are not mapped to fixed token budgets.
- Server-side, for every mode including `single_large_model`: model file size vs. the safe
  VRAM budget of the lane(s) it's headed for, any `pin_lane` constraint, and — when the
  model's GGUF metadata is one the server can confidently read (see caveat below) — an
  estimated KV-cache cost for the resolved `context_window`, added to the file size and
  checked against the same budget. This is the "final verdict" step described in
  `/api/clients/deploy`.
  - The KV-cache estimate is best-effort: hybrid state-space/attention architectures, or a
    sliding-window architecture whose GGUF doesn't expose a per-layer pattern, can't be
    estimated confidently, and the deploy falls back to the file-size-only check for those
    (never blocked by a guess it isn't confident in).

**Shares:** the same shape as `GET /api/status` (the resulting `state`/`servers`/etc.).

**Errors:** `400` unsupported mode, unknown model/lane, model (or model + estimated KV cache)
doesn't fit the lane, model is pinned to a different lane, invalid context/budget; `409`
another deployment change is already in progress; `500` one or more servers failed to start
(message includes each failure), including `Not enough free VRAM to keep <model> on the GPU ...`
when the model would not fit the VRAM free on its GPU (`policy.vram_admission`).

## POST `/api/models/refresh`

Re-scans configured model paths and the LM Studio model directory for new/removed GGUF files.

**Needs:** nothing (body ignored).

**Shares:** `{"catalog": { /* same shape as bootstrap.catalog */ }}`
```jsonc
// catalog shape, for reference:
{
  "project": "string",
  "lanes": [{"key": "..", "display": "..", "match": "..", "vram_gb": "float", "capacity_gb": "float", "port": "int", "backend": "rocm | vulkan"}],
  "models": [{"id": "..", "name": "..", "display_name": "..", "family": "..", "quantization": "..", "publisher": "..", "source": "configured | lm_studio", "path": "..", "exists": "bool", "size_gb": "float", "multi_gpu": "bool", "pin_lane": "string | null", "default_context": "int | null", "default_reasoning_budget": "int", "tensor_split": "string | null"}],
  "model_count": "int",
  "model_roots": ["path", "..."],
  "context_presets": ["int", "..."],
  "reasoning_presets": {"off": 0, "low": 1024, "medium": 4096, "high": 8192}
}
```

## POST `/api/unload`

Frees GPU memory.

**Needs:** an optional JSON body.
- `{}` (nothing): stops every active server **and** any leftover llama-server holding a lane port,
  then returns to `idle`.
- `{"lane": "<lane_key>"}`: stops only what serves that lane (a model split across both GPUs also
  frees its other GPU, and a leftover server on that lane's port is stopped too). The other lane keeps
  running and the profile is updated; with nothing left the state is `idle`. The reply carries
  `"released": true` when a model was stopped and `false` when the lane had nothing loaded.
- `{"lane": "<lane_key>", "model": "<model>"}`: the same, but only if `<model>` is still what serves
  that lane. `<model>` is the string you deployed with (a catalog name or id) or the resolved model
  name. If the lane is empty or another model has replaced yours, nothing is stopped and the reply
  carries `"released": false`. This is how a client releases only its own model. `model` without `lane`
  is a `400`.
- `{"orphans_only": true}`: stops only the leftover servers and keeps every loaded model.

**Running requests.** A model that is answering a request gets `policy.drain_timeout_seconds` (30 s) to
finish before it is stopped, so one client's unload does not cut another client's answer. The status
shows `state: "switching"` meanwhile, and each server reports `inflight` (requests it is answering).
`{"force": true}` ends that wait: if an unload or load is already waiting, it is told to stop waiting and
the call returns at once; otherwise the models are stopped immediately, cutting any request in progress
(the client sees a closed connection). Combine with `lane` to force one GPU.

**Shares:** same shape as `GET /api/status`, plus `"released": bool` for a lane-scoped unload.
**Errors:** `400` unknown lane, or `model` without `lane`; `409` if another deployment change is
already in progress (for a second graceful unload the message says it is waiting for running requests
and to use `{"force": true}`).

## Releasing the GPUs when a task is done

**Requirement for every client application.** The service never unloads a model because a task ended
(there is no idle timeout). A model stays in GPU memory, with a host process holding some RAM, until
something unloads it, replaces it, or the service stops. An application that loaded a model for a task
(through `/api/deploy`, `/api/clients/deploy*`, or a gateway request that loaded it on demand) **must
release it when the task is complete.**

1. **When.** After the last request of the task has finished and any streamed response has ended, on
   success **and** on failure (use `finally`). Not between the requests of one task: loading a model
   again takes seconds to tens of seconds.
2. **What.** Only what you loaded. For each lane you used, send
   `POST /api/unload {"lane": "<lane>", "model": "<the model you deployed>"}`. The lane is in the deploy
   reply (`servers[].lane_keys`); for a model split across both GPUs either lane releases both.
3. **Never the bare form.** `POST /api/unload {}` stops every model, including other applications', and
   so does `{"force": true}`. Use them only if you are certain you are the service's only user, or when
   a person asks for it.
4. **Outcomes.** `200` with `released: true`: done. `200` with `released: false`: the lane was empty
   or another application's model is there now; nothing to do, do not retry. `409` (a load or unload
   is in progress): retry after about 2 s, a few times. `400`: a bug in the caller.
5. **Per-lane queues** (`/api/clients/deploy_lane`): release each lane when that lane's queue is empty.
   Do not wait for the other lane.
6. **Tools that cannot call HTTP** (Cline, OpenCode, Continue): whoever starts the session releases the
   models at its end, from the dashboard (Service, or the top-bar Unload) or with the call above.

A graceful unload waits up to `policy.drain_timeout_seconds` for requests still running on that lane (which
may be another application's) before it stops the model, so unloading right after your own last response
is safe. GPU memory is freed within seconds of the reply; a following load waits up to 20 s for it.

**What is not enforced.** The service does not record who loaded what, so this is a rule for clients, not
a lock; the `model` guard in step 2 is what keeps it safe. The service separately evicts the least recently
used **idle** model when it needs the GPU for another one, and never stops a model that is answering.

```python
import time, requests

def release(base, lane, model):
    for _ in range(5):
        r = requests.post(f"{base}/api/unload", json={"lane": lane, "model": model}, timeout=60)
        if r.status_code != 409:           # 409: another load/unload in progress, try again
            return r.ok and r.json().get("released")
        time.sleep(2)

lanes = deploy(model)                      # your deploy call; note which lanes it used
try:
    run_your_task()
finally:
    for lane in lanes:
        release(BASE, lane, model)
```

## Service control

Everything the dashboard's **Service** page does is available here. The state-changing calls
(`POST /api/service/*`, `POST /api/settings`) require `Content-Type: application/json`, which forces a
CORS preflight so a web page cannot trigger them with a blind form POST (`415` otherwise).

### GET `/api/service`

**Shares:** `{"pid", "started_at", "uptime_seconds", "host", "port", "config_path", "data_dir",
"app_version", "python", "host_os": {"system", "release", "version", "machine",
"distribution"?, "distribution_id"?, "distribution_version"?},
"can_restart": bool, "pending": "restart | stop | null", "restore_pending": bool}`.
`pid` changes when the service restarts, which is how a client knows the new process is the one
answering. `501` if this process has no service control (it was not started through `app.py`).

### POST `/api/service/restart`

**Needs:** `{"reload_models": bool}` (optional, default false).

Stops every model server and any leftover one on a lane port **before replying**, so a deployment
change in progress is reported to the caller. It then starts a replacement process (detached; it waits for
this one to exit before binding the port) and shuts this one down, a moment after the reply. With
`reload_models: true` the current deployment, including its coding-agent mode, is saved and the
replacement loads it again; a saved deployment older than 10 minutes is ignored. Poll `GET /api/service`
until `pid` changes. The replacement logs to `<data_dir>/launcher.log`.

**Shares:** `{"ok": true, "action": "restart", "reload_models": bool, "pid": <this process>}`.
**Errors:** `409` another deployment change or a service action is in progress, or this process cannot
restart itself; `415` not JSON.

### POST `/api/service/stop`

Stops every model server and then the service. Nothing answers afterwards; start it again with
`launch_service.py`. Same errors as restart.

### GET `/api/settings` and POST `/api/settings`

The `[policy]` settings that can be changed without editing the file:
`load_mode`, `cache_ram_mib`, `vram_admission`, `vram_headroom_gb`, `auto_load_models`,
`residency_guard`, `residency_check_seconds`, `drain_timeout_seconds` and `kill_busy_ports`.

`GET` shares `{"settings": [{"key", "label", "help", "type": "bool | int | float | choice", "value", "min",
"max", "choices", "apply": "live | next_load"}], "config_path"}`.

`POST` needs `{"values": {"<key>": <value>, ...}}` (the keys may also be sent at the top level). Every value
is validated first; nothing is applied unless all are valid. Changed values are applied to the running
service and saved into the `[policy]` table of the config file, keeping its comments and layout; the
first save leaves a copy of the original as `<config>.bak`. `apply: "live"` values take effect at once;
`next_load` values change the command line of the next model load, so a model that is already loaded
keeps its old behaviour until it is reloaded.

**Shares:** the `GET` payload plus `"changed": [keys]` and `"next_load": [changed keys that wait for the next load]`.
**Errors:** `400` unknown key or invalid value; `415` not JSON; `500` the file could not be written (nothing
changed).

## GET `/api/hardware`

Live GPU readings, independent of any inference or training process, plus who holds each lane.

**Shares:**
```jsonc
{
  "hardware": {
    "available": "bool",                       // false when no GPU source exists on this host
    "unavailable_reason": "string | null",
    "interval_s": "float (sampling period, 1.0)",
    "gpus": [{
      "pci_bus": "0000:03:00.0", "card": "string", "name": "string", "lane": "lane key | null",
      "vram_total_bytes": "int | null",
      "latest": {                              // null until the first sample
        "ts": "float", "gpu_busy_pct": "float | null", "vram_used_bytes": "int | null",
        "gtt_used_bytes": "int | null (system memory used by the GPU)",
        "temp_edge_c": "float | null", "temp_junction_c": "float | null", "temp_memory_c": "float | null",
        "power_w": "float | null", "sclk_mhz": "float | null", "mclk_mhz": "float | null", "fan_rpm": "int | null",
        "pcie_speed": "e.g. '32.0 GT/s PCIe' | null", "pcie_width": "int | null",
        "pcie_max_speed": "string | null", "pcie_max_width": "int | null",
        "unavailable": {"<field>": "why it is null"}   // a missing sensor is null with a reason, never 0
      },
      "history": [{"ts": "...", "gpu_busy_pct": "...", "vram_used_bytes": "...", "temp_*_c": "...", "power_w": "..."}]
    }],
    "system": {"ts": "float", "cpu_pct": "float | null", "ram_used_pct": "float | null", "swap_used_bytes": "int | null"},
    "amd_smi": {"available": "bool", "reason": "string | null"} | null,
    "sampler": {"ticks": "int", "dropped_ticks": "int", "last_sample_ms": "float", "avg_sample_ms": "float",
                "max_sample_ms": "float", "callback_errors": "int", "read_errors": "int", "last_error": "string | null"}
  },
  "lane_holders": {"<lane key>": {"holder": "string", "kind": "training | inference", "...": "lease details"}},
  "kernel_faults": {"active": "bool", "reason": "string | null"}   // amdgpu reset / page-fault follower (Linux only)
}
```

**Sources.** Linux reads amdgpu sysfs. Windows reads AMD's ADL library plus Windows performance counters and
device properties (VRAM and PCIe link refresh every 5 s and 60 s); see
[docs/TRAINING_TELEMETRY.md](docs/TRAINING_TELEMETRY.md#windows-gpu-sources-adl-and-windows-counters). Samples are
written to the database only while a training run is active. `unavailable_reason` is set and `gpus` is empty when
the host has neither source (for example Windows without the AMD driver).

## POST `/api/chat`

Run a prompt against the currently loaded deployment.

**Needs:**

| field | type | required? | notes |
|---|---|---|---|
| `messages` | `[{role, content}, ...]` | one of `messages`/`prompt` required | full conversation; takes priority over `prompt` if both are given |
| `prompt` | string | — | shorthand for a single user message |
| `target` | string | optional | a lane key/target from `status.servers[].target`, or `"compare"` to fan out to every active target; defaults to the first/only active target |
| `targets` | `["<target>", ...]` | optional | explicit list of targets to fan out to; overrides `target` |
| `temperature` | number | optional | default `0.7` |
| `top_p` | number | optional | default `0.95` |
| `max_tokens` | int | optional | default `1024` |
| `seed` | int | optional | passed through to the backend only if present |
| `timeout_seconds` | number | optional | default `900` |
| `client_id` | uuid string | optional | from `/api/clients/register`; tags this run and appends its metrics to that client's `output_path` |
| `task_label` | string | optional | freeform tag for what kind of work this call was (e.g. `"relevance"`, `"resume_extraction"`, `"fraud_d2_fusion"`). Stored on the run and rolled up into `/api/dashboard`'s `categories`/`category_models`, which back the dashboard's Analytics tab. No enum — any non-empty string works. |
| `project_id` | uuid string | optional | project from `POST /api/projects`; a registered `client_id` supplies this automatically |
| `workload_kind` | string | optional | `evaluation` (default) or `coding_agent` |
| `agent_role` | string | optional | e.g. `developer`, `qa`, or `reviewer` |
| `agent_session_id` | uuid string | optional | session created with `POST /api/agent-sessions` |
| `coding_task_id` | uuid string | optional | task created with `POST /api/coding-tasks` |

**Shares:**
```jsonc
{
  "comparison_id": "uuid | null (set only when fanning out to more than one target)",
  "results": {
    "<target>": {
      "run_id": "uuid",
      "model": "string",
      "status": "completed | failed",
      "output_text": "string | null",
      "reasoning_text": "string | null",
      "finish_reason": "string | null",
      "usage": {
        "input_tokens": "int | null", "cached_input_tokens": "int | null", "uncached_input_tokens": "int | null",
        "prefill_tokens": "int | null", "thinking_tokens": "int | null",
        "thinking_tokens_estimated": "int | null", "thinking_characters": "int | null",
        "thinking_tokens_source": "backend | estimated_from_characters | unavailable",
        "tool_calls_total": "int", "tool_calls_malformed": "int", "tool_call_valid_rate": "float | null",
        "visible_output_tokens": "int | null", "output_tokens": "int | null", "total_tokens": "int | null"
      },
      "timing": {
        "time_to_first_token_ms": "float | null", "prefill_duration_ms": "float | null", "decode_duration_ms": "float | null",
        "end_to_end_duration_ms": "float", "prefill_tokens_per_second": "float | null", "tokens_per_second": "float | null",
        "backend_prompt_tokens": "int | null", "backend_output_tokens": "int | null", "native_timing": "object (raw backend timings)"
      },
      "resources": {
        "sample_count": "int", "avg_process_cpu_pct": "float | null", "peak_process_cpu_pct": "float | null",
        "avg_gpu_utilization_pct": "float | null", "peak_gpu_utilization_pct": "float | null",
        "peak_process_rss_bytes": "int | null", "peak_process_private_bytes": "int | null",
        "process_page_fault_delta": "int | null",
        "peak_dedicated_vram_bytes": "int | null", "peak_shared_gpu_memory_bytes": "int | null",
        "spill_suspected": "bool | null",
        "avg_system_cpu_pct": "float | null", "peak_system_ram_used_bytes": "int | null", "peak_system_pagefile_used_bytes": "int | null",
        "samples_started_at": "ISO-8601 | null", "samples_finished_at": "ISO-8601 | null"
      },
      "error_text": "string | null (set only when status is failed)",
      "backend_response": null,
      "client_delivery": {
        "written": "bool",
        "path": "string (resolved output_path)",
        "error": "string (present only if written is false)"
      }
      // client_delivery is present only when the request included client_id
    }
  }
}
```

**Errors:** `409` no deployment is loaded; `400` unknown/inactive target(s), no
`messages`/`prompt` given, or unknown `client_id`.

When `client_id` is supplied, the JSONL record appended to the registered `output_path` contains
run/client/project identity, target/model/device/context, status, metrics, reasoning, error, and the
same `host_os` object described under `GET /api/runs`. The HTTP chat result itself does not duplicate
`host_os`; use its `run_id` with `GET /api/runs/{run_id}` when that attribution is needed.

---

## OpenAI-compatible coding-agent gateway

`GET /v1/models` and `POST /v1/chat/completions` provide the standard OpenAI-compatible
surface. Lane- and role-pinned variants are also supported:
`/v1/<lane>/<agent-role>/models` and `/v1/<lane>/<agent-role>/chat/completions`.

**On-demand models (`policy.auto_load_models`, default on).** When `model` names a catalog model
(a `catalog.models[].name` or `.id`), that model serves the request:
- Already loaded: routed to it.
- Not loaded: loaded in coding-agent mode onto a free GPU where it fits (smallest first), or in
  place of the least recently used **idle** model on a GPU where it fits. A model no single GPU
  can hold spans both GPUs after they are freed. A `/v1/<lane>/...` URL restricts placement to
  that lane.
- GPU only: the server starts only if its weights + estimated KV cache + `policy.vram_headroom_gb`
  fit the VRAM actually free on the GPU, counting every process (`policy.vram_admission`; on Windows
  from the OS's accounting, not HIP's per-process figure; on Linux from the native llama.cpp device
  listing). Placement prefers a GPU where it fits
  now, counting VRAM that evicting our own idle model there would release; when it fits nowhere
  the request gets **503** with the shortfall and nothing is evicted. The same check guards
  `/api/deploy` and the `/api/clients/deploy*` endpoints, which report it in their `500` message.
- A model that is serving a request is never stopped: if every candidate stays busy for
  `policy.drain_timeout_seconds`, the request gets **503** and should be retried. Loads are
  serialized; a request waits up to `project.start_timeout_seconds` for another load to finish.

`model: "dgpu:<lane>"`, a lane key, and model names that are not in the catalog keep plain lane
routing to whatever is loaded. With auto-load on, `GET /v1/models` lists the loaded lanes followed
by every requestable catalog model (`{"id", "object": "model", "owned_by", "loaded": bool}`).

**Zero-config, watertight attribution (no API key, no pre-registration).** Every request is
attributed to a real project, resolved in this order and **created on first use**:
1. a Bearer value that is a registered `client_id` → that client's project;
2. otherwise a project name/id from the `X-DGPU-Project` header, else the **Bearer value itself**
   (so a tool whose only field is an "API key" just puts a project name there);
3. nothing identifying → the configured `project.default_project` (default `"unassigned"`).

An unknown Bearer value is **not** rejected — it becomes a project name. So a client connects by
pointing at the base URL and naming itself once; nothing is ever left untracked. (Registering via
`POST /api/clients/register` is still available and adds the metrics-file sink, but is optional.)

Streaming requests are relayed as SSE and receive an `X-DGPU-Run-ID` response header; non-streaming
responses contain `dgpu_run_id`. The optional `X-DGPU-Agent-Role`, `X-DGPU-Session`, `X-DGPU-Task`,
`X-DGPU-Task-Label`, and `X-DGPU-Connection-Type` headers add finer attribution. Coding prompt and
completion content is redacted from storage unless `X-DGPU-Capture-Content: true` is supplied.

### Inline autocomplete (FIM) endpoints

`POST /v1/completions` and `POST /v1/infill` (and their `/v1/<lane>/<role>/...` variants) proxy
the raw fill-in-middle endpoints of the selected lane's llama-server, for IDE inline
autocomplete (Continue, Tabby, Twinny, llama.vscode, …). Lane selection is identical to chat
(path, `X-DGPU-Target`, or `model`). These are a **thin passthrough**: because autocomplete
fires on nearly every typing pause, they do **no** run recording, GPU sampling, or history
logging — so autocomplete adds no inference overhead and never bloats the metrics DB. They are
still drain-safe (counted in-flight during a deployment change). Point a small, fast FIM coder
model (e.g. Qwen2.5-Coder-1.5B/7B) at the smaller lane for autocomplete and the agent model at
the larger lane.

## GET / POST `/api/lmstudio/gpus`

Choose which GPUs **LM Studio's own** llama.cpp runtime may use (this is unrelated to the lanes
this service launches itself). It reads and edits `<lmstudio_home>/.internal/hardware-config.json` —
only `load.gpuSplitConfig.disabledGpus` for the backend; every other field is preserved, the first
original is kept as `hardware-config.json.dgpu.bak`, and writes are atomic. A config that cannot be
parsed is never overwritten (`409`).

Device indexes are the order `llama-server --list-devices` prints (the same order LM Studio uses),
e.g. `0` = R9700, `1` = 9070 XT, `2` = integrated graphics.

**GET** — optional `?backend=rocm|vulkan` (default `rocm`).

**POST needs:**

| field | type | required? |
|---|---|---|
| `gpus` | `"all"`, or non-empty list of device indexes, device ids (`"ROCm0"`) or unique name fragments (`"R9700"`) | **required** |
| `backend` | `"rocm"` \| `"vulkan"` | optional, default `"rocm"` |

**Shares:**
```jsonc
{
  "backend": "rocm", "config_path": "C:/Users/.../.lmstudio/.internal/hardware-config.json",
  "devices": [{ "index": 0, "device": "ROCm0", "name": "AMD Radeon AI PRO R9700",
                "memory": "32624 MiB, 32472 MiB free", "enabled": true }],
  "enabled": [0], "disabled_gpus": [1, 2],
  "lm_studio": { "reachable": true, "models": [{ "identifier": "qwen/qwen3.8-27b", "status": "idle",
                 "queued": 0, "size_bytes": 17742040464, "context_length": 32768 }] },
  // POST only:
  "previous_disabled_gpus": [], "backup_path": "…/hardware-config.json.dgpu.bak", "note": "string"
}
```
**Errors:** `400` unknown backend, empty/invalid `gpus`, or a selector matching no / several GPUs;
`415` the POST is not `Content-Type: application/json` (a web page cannot trigger it with a blind
form post); `409` the runtime or config file cannot be read or written.

This changes LM Studio's saved setting, not its running process. A model that is already loaded keeps
its placement, so apply it in this order: `POST /api/lmstudio/release` (with `stop_server`), then
`POST /api/lmstudio/gpus`, then start LM Studio's server and load the model. That sequence was
checked on the R9700 + 9070 XT machine: the reloaded 27B model sat entirely on the R9700. It was not
isolated whether the server restart is required, and editing the file while models stay loaded was
not seen to move them.

## POST `/api/lmstudio/release`

Free the GPU memory and RAM LM Studio holds: unload every model, optionally stop its server. Use it
before deploying onto this service's lanes, which share the same GPUs.

**Needs:** `{ "force": false, "stop_server": false }` — both optional, and only the JSON value `true`
counts. Without `force` it refuses (`409`) while any model is generating or has queued requests.

**Shares:** `{ "unloaded": ["<identifier>"], "server_stopped": bool, "lm_studio": { …as above… } }`

**Errors:** `409` LM Studio is busy, the `lms` CLI is missing, or an `lms` command failed; `415` as above.

## GET `/api/usage`

Per-project usage across **every** connection — coding agents, model inference, inline
autocomplete, and any client that self-tags. It is a **daily rollup** (one row per project ×
connection-type × day), accumulated in memory and flushed on a background timer, so recording
adds no per-request DB write and the table never bloats.

The `connection_type` defaults per entry point — `coding_agent` (chat gateway), the request's
`workload_kind` (`/api/chat`), `autocomplete` (`/v1/completions` and `/v1/infill`) — and any
caller can override it with the `X-DGPU-Connection-Type` header (e.g. `training`, `batch`,
`rag`), so a training/data pipeline is tracked under its own type. Project attribution comes
from the Bearer `client_id` or the `X-DGPU-Project` header; unattributed usage is recorded
under `project_id: ""`.

**Query params:** `project_id`, `connection_type`, `start_date`/`end_date` (YYYY-MM-DD,
inclusive; 400 on bad format).

**Shares:**
```jsonc
{
  "rows": [
    { "project_id": "…", "connection_type": "coding_agent", "usage_date": "2026-10-11",
      "requests": 42, "input_tokens": 183000, "output_tokens": 15200, "updated_at": "ISO-8601" }
  ],
  "by_connection_type": {
    "coding_agent": { "requests": 42, "input_tokens": 183000, "output_tokens": 15200 },
    "autocomplete": { "requests": 1900, "input_tokens": 240000, "output_tokens": 38000 }
  }
}
```
Token counts are exact for chat/inference and best-effort for autocomplete (counted for
non-streamed completions; streamed autocomplete records the request without token totals).

## Training and ROCm telemetry

Training runs are independent of inference runs and live in `training.sqlite3`. The training
process sends metrics through `TrainingTracker`; `app.py` samples host/GPU telemetry and owns lane
leases. A missing sensor is returned as `null` with an unavailable reason, never as zero.

### GET `/api/hardware`

Shares `{"hardware", "lane_holders", "kernel_faults"}`. `hardware` contains availability,
sampling interval, AMD GPU identities, latest/history samples, system load, AMD SMI status and
sampler overhead/errors. Linux samples may include VRAM/GTT, temperatures, power, clocks, PCIe,
`throttle_status`, `amd_smi_fields`, and `unavailable` reasons.

### POST and GET `/api/training/runs`

`POST` requires `name`; optional fields include `id`, `lanes`, `config`, `dataset`, `env`, `seed`,
`max_steps`, `parent_run_id`, `tags`, `pid`, `world_size`, `unload_inference`, and `lease_ttl_s`.
Configured lanes are leased atomically; inference conflicts return `409` unless explicitly unloaded.

The created run and every row returned by `GET` include top-level:

```jsonc
{
  "id": "string", "name": "string", "status": "running | stalled | lost | completed | failed | cancelled",
  "host_os": {"system": "Linux", "distribution_id": "ubuntu", "release": "...", "machine": "..."},
  "lanes": ["lane-key"], "world_size": 1, "last_step": 0, "progress": null,
  "tokens": {}, "throughput": {}, "latest": null, "peak_mem_bytes": null
}
```

The detail response `GET /api/training/runs/{id}` adds `env`, `config`, `dataset`, `summary`, events,
checkpoints, hardware summary, artifacts, notes and alerts. `env.os` identifies the training process;
`env.server_host_os` identifies the dashboard host; `env.server_hardware` contains physical GPU/PCIe,
kernel, amdgpu and ROCm details. Top-level `host_os` prefers the training-process identity and always
falls back to the server identity when tracker-side capture is disabled.

### Training run subresources

- `POST /api/training/runs/{id}/metrics`, `/events`, `/checkpoints`, `/finish`, `/evals`, `/notes`,
  `/examples`, `/select`, and `/artifacts` ingest the corresponding tracker or experiment records.
- `GET /api/training/runs/{id}/series` returns downsampled step and persisted GPU series. GPU points
  include `throttle_status`, `telemetry_sources`, and `unavailable` metadata.
- `GET /api/training/runs/{id}/analysis`, `/alerts`, `/evals`, `/resume-report`, `/hard-examples`,
  `/ranks`, and `/artifacts` expose derived analysis and saved state.
- `GET /api/training/alerts`, `/compare`, `/scaling`, `/eval-queue`, `/benchmarks`, `/profiles`, and
  `/notes` expose cross-run views. Matching POST endpoints submit evaluations, forecasts, benchmarks
  and notes; benchmark cancellation and checkpoint designate/promote/demote routes are also supported.

### GPU leases: `/api/leases`

A lease is an exclusive, expiring claim on one or more lanes, used mainly by training so an
inference deploy can't land on a GPU that's training. Deploys onto a leased lane return `409`.

- `GET /api/leases` → `{"leases": [<lease>], "lane_holders": {"<lane>": {"holder", "kind", "lease_id", "expires_at"}}}`.
  Loaded inference servers also appear as derived pseudo-leases (`id: "inference:<target>"`,
  `kind: "inference"`, `model`, `expires_at: null`); they can't be renewed or released here.
- `POST /api/leases` with `{"holder": "training:<run_id>", "lanes": ["r9700"], "kind": "training", "ttl_s": 120, "pid": 1234}`.
  `holder` and `lanes` are required; `ttl_s` defaults to 120 (max 7 days). Unknown lanes → `400`;
  a lane with inference loaded or already leased → `409`. Re-acquiring with the same holder and
  lanes refreshes the existing lease. The response is the lease **including `token`**, which is
  needed to renew or release it.
- `POST /api/leases/{id}/renew` with `{"token", "ttl_s"?}` extends it (an expired lease is
  revived only if its lanes are still free).
- `DELETE /api/leases/{id}?token=...` releases it.
- `POST /api/leases/{id}/preempt` stops the holder's process (only if the PID still matches the
  start token recorded at acquire) and releases the lease → `{"process": "terminated | pid_mismatch_or_gone | no_process", "lease": <lease>}`.

The token can be passed as body `token`, query `?token=`, or an `X-Lease-Token` header; a wrong
token → `403`. Lease shape:

```jsonc
{
  "id": "hex", "holder": "string", "kind": "training | ...", "state": "active | expired | released",
  "lanes": ["lane-key"], "ttl_s": "float", "remaining_s": "float (0 unless active)",
  "acquired_at": "unix seconds", "renewed_at": "unix seconds", "expires_at": "unix seconds",
  "released_at": "unix seconds | null", "pid": "int | null", "exe": "string | null",
  "token": "string (acquire response only)"
}
```

## Project and agent telemetry endpoints

- `GET /api/projects` lists registered projects with run/token totals.
- `POST /api/projects` needs `name`; `repo_path` and `git_remote` are optional. It creates or
  updates by case-insensitive name.
- `POST /api/agent-sessions` needs `project_id`; `agent_role` and `runtime` are optional.
- `POST /api/coding-tasks` needs `project_id` and `title`; `branch` is optional.
- `POST /api/tool-events` needs `project_id`; accepts `session_id`, `task_id`, `run_id`,
  `tool_type`, `duration_ms`, `exit_code`, `status`, and arbitrary JSON `detail`.

## POST `/api/clients/register`

Register a consuming codebase once, up front. **Optional** — the gateway already auto-provisions
a project from the Bearer key or `X-DGPU-Project` header (see "Zero-config attribution" above).
Register only when you also want run metrics appended to a file in your repo (`output_path`).

**Needs:**

| field | type | required? |
|---|---|---|
| `project_name` | string, non-empty | **required** |
| `github_repo` | string | optional |
| `output_path` | string, non-empty | **required** — a file path *in the caller's own repo* where run metrics get appended as JSON lines by `/api/chat` |

**Shares:**
```jsonc
{
  "client_id": "uuid",
  "project_name": "string",
  "github_repo": "string | null",
  "output_path": "string",
  "created_at": "ISO-8601"
}
```
**Errors:** `400` if `project_name` or `output_path` is missing/empty.

## POST `/api/clients/deploy`

Ask "can you run this model, and if so, load it" in one call — a thin wrapper around
`/api/deploy` in `single_gpu` mode that resolves a lane automatically.

**Needs:**

| field | type | required? | notes |
|---|---|---|---|
| `model` | string | **required** | name or catalog id from `/api/bootstrap` → `catalog.models` |
| `gpu` | string | optional | a lane key from `catalog.lanes[].key`; if omitted, the service picks the smallest lane the model fits (or its pinned lane) |
| `context_window` | int | optional | same defaulting/validation as `/api/deploy` |
| `input_tokens` | int | optional, with `max_output_tokens` | same sizing guarantee as `/api/deploy`: pass the real size of the largest prompt you'll send and the context window is grown to fit it |
| `max_output_tokens` | int | optional, with `input_tokens` | expected completion length; required together with `input_tokens` |

**Shares:** same shape as `GET /api/status` on success (the resolved `device`, `base_url`,
`context_window`, etc. are in `servers[]`).

**Errors:** same as `/api/deploy` — `400` unknown model/lane, doesn't fit, pinned elsewhere,
invalid context, required context exceeds `policy.ctx_max`, or `input_tokens`/`max_output_tokens`
given without the other; `409` another deployment change in progress. **Note:** like
`/api/deploy`, this replaces whatever was previously loaded.

## POST `/api/clients/deploy_parallel`

Like `/api/clients/deploy`, but for N models where N is the lane count (today: 2) — auto-assigns
each model a distinct lane instead of requiring the caller to already know which model goes where.
A thin wrapper around `/api/deploy` in `parallel_models` mode that resolves lanes automatically.

**Needs:**

| field | type | required? | notes |
|---|---|---|---|
| `models` | `["name1", "name2"]` | **required** | exactly one model per configured lane; each a name or catalog id from `/api/bootstrap` → `catalog.models` |
| `context_window` | int | optional | same defaulting/validation as `/api/deploy`, applied identically to every model in the request |
| `input_tokens` | int | optional, with `max_output_tokens` | same sizing guarantee as `/api/deploy`/`/api/clients/deploy`, applied identically to every model |
| `max_output_tokens` | int | optional, with `input_tokens` | expected completion length; required together with `input_tokens` |

Lane assignment is strict smallest-fit at the requested context: a model that passes the full
model-file + KV-cache check on the smallest lane is assigned there. Only a model that fails that
check may use the larger lane. Therefore two models that both fit the small lane, or two models
that both require the large lane, return `400` and must be run sequentially on their common lane;
the service never promotes a small model merely to keep both GPUs occupied. Every capacity,
context, and pin-lane check is the same check used by `/api/deploy`.

**Shares:** same shape as `GET /api/status` on success (the resolved `device`, `base_url`,
`context_window`, etc. are in `servers[]`, one entry per model/lane pair).

**Finding which lane a given requested model landed on:** use `profile.models[]`
(`[{"model": <exact string you sent>, "lane": <key>}, ...]`), **not** `servers[]`.
`servers[].model` is the orchestrator's *resolved* short display name (e.g.
`"gemma-3-270m-it-Q8_0"`), which will not equal a catalog id you requested with (e.g.
`"lmstudio:lmstudio-community/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q8_0.gguf"`) — matching against
`servers[].model` silently fails to find every model. `profile.models[].model` echoes your input
strings back verbatim, making it the only reliable join key. (Same caveat applies to
`/api/deploy`'s `single_large_model`/`single_gpu` modes and `/api/clients/deploy`'s response,
for the same reason — `servers[].model` is always the resolved short name there too.)

**Errors:** `400` if `models` isn't exactly one entry per lane, plus everything `/api/deploy`
can return for `parallel_models` mode (unknown model/lane, doesn't fit, pinned elsewhere, invalid
context, required context exceeds `policy.ctx_max`); `409` another deployment change in progress.
**Note:** like `/api/deploy`, this replaces whatever was previously loaded.

## POST `/api/clients/deploy_lane`

Replace the model on one lane while preserving every other active lane. This supports
work-conserving schedulers: when one GPU finishes earlier, it can load its next assigned model
without waiting for or interrupting the other GPU.

**Needs:** `model` (catalog name/id), `lane` (lane key), and optional `context_window`,
`input_tokens`, and `max_output_tokens` with the same validation rules as
`/api/clients/deploy`. The caller must have no outstanding chat request on the lane being
replaced. Simultaneous lane replacements are serialized server-side rather than returning 409.

**Works from an idle service — you do not need `/api/clients/deploy_parallel` first.** With
nothing loaded on the named lane there is simply nothing to stop, so a scheduler can bring both
lanes up by calling this endpoint once per lane: after the first call `mode` is `single_gpu`, and
it becomes `parallel_models` when the second lane lands. "Replace" describes the effect on the
one lane named, not a precondition that something is already running there.

**Shares:** the normal status response containing all active servers, including preserved lanes.
Only the requested lane's process/model changes.

**`profile.models[]` stays a valid join key across a lane replacement.** The entry for the
replaced lane carries the exact `model` string this call sent, and entries for preserved lanes
carry the exact strings their own deploy call sent, forwarded unchanged. A lane loaded by some
path that recorded no request string (a direct `/api/deploy` that omitted `lane`, for example)
falls back to the resolved short display name for that one entry — the same value `servers[]`
reports. `servers[].model` remains the resolved short name for every lane and is still not
comparable against a catalog id you requested with.

## POST `/api/clients/check_fit`

Read-only feasibility check: "would this model fit this lane at this context window" — without
loading anything. Runs the exact same resolution/validation `/api/deploy` would (model+lane
resolution, then the capacity/KV-cache check every deploy mode uses), but never calls `deploy()`
and never starts a process. Built for a client that wants to **plan a schedule ahead of time** —
e.g. partition a whole model list into "fits the small lane" / "needs the big lane" buckets
before deploying anything — instead of discovering fit only by trial deploy, which would
actually load a model just to find out it doesn't fit.

**Needs:**

| field | type | required? | notes |
|---|---|---|---|
| `model` | string | **required** | name or catalog id from `/api/bootstrap` → `catalog.models` |
| `lane` | string | **required** | a lane key from `catalog.lanes[].key` |
| `context_window` | int | optional | same defaulting as `/api/deploy`; omit to use the model's configured/policy default |
| `input_tokens` | int | optional, with `max_output_tokens` | same sizing guarantee as `/api/deploy` — the context is grown to fit this before the capacity check runs |
| `max_output_tokens` | int | optional, with `input_tokens` | expected completion length; required together with `input_tokens` |

**Shares:**
```jsonc
{
  "fits": "bool",
  "reason": "string | null — the exact AppError message a real /api/deploy would have raised, present only when fits is false",
  "resolved_context_window": "int | null — the context window actually checked against (after any input_tokens/max_output_tokens growth), present only when fits is true"
}
```

**Errors:** this endpoint answers `200` with `fits: false` for essentially everything that can
go wrong, including cases you may expect to be a `400`. An unknown `model`, an unknown `lane`, a
model file missing from disk, `input_tokens` without `max_output_tokens`, and a `context_window`
below 256 all come back as `{"fits": false, "reason": "<message>"}` rather than an HTTP error,
because the check reports every validation failure the same way a capacity failure is reported.

**So validate lane keys yourself before bucketing a model list.** A typo in `lane` returns
`fits: false` with `reason: "Unknown lane: <key>"` for *every* model you test, which looks
exactly like a real "nothing fits this GPU" verdict and will silently push your whole list onto
the other lane. Take lane keys from `/api/bootstrap` → `catalog.lanes[].key`, and treat a
`reason` that does not describe capacity as a bug in your request, not a placement answer.

It does **not** return `409` while a deployment change is in flight — it takes no deployment
lock at all, which is what makes it safe to call in a loop (see below). Only a request that
cannot be parsed at the HTTP layer (a non-integer `context_window`, say) fails with a status
code rather than a verdict.

**Never mutates state and never conflicts with an active deployment** — safe to call in a loop,
once per candidate model, without affecting whatever is currently loaded.

## GET `/api/clients/{client_id}/runs`

**Needs:** `client_id` as a path segment.

**Shares:** `{"runs": [<run row>, ...]}` — up to 100 most recent runs made with that
`client_id`, same row shape as `GET /api/runs`.

## GET `/api/clients/schema`

**Needs:** nothing. **Shares:** the field-by-field documentation of every `usage`/`timing`/
`resources` key described under `/api/chat` above, as `{"usage": {...}, "timing": {...},
"resources": {...}}` with one description string per field — this is the machine-readable
version of the tables in this document, kept in sync in [app.py](app.py)
(`CLIENT_METRICS_SCHEMA`).
