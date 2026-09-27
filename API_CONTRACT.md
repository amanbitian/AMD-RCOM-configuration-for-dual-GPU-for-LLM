# API Contract

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
for the full example.

## Conventions

- Base URL: `http://127.0.0.1:<port>` (default port `8090`; see `--port`/`--host` in
  [README.md](README.md#quick-start)).
- All request/response bodies are JSON. `Content-Type: application/json` on requests with a
  body; request bodies are capped at 10 MB.
- **No authentication.** This is a local-trust-boundary service — anyone who can reach the
  bound host/port can call any endpoint. Do not bind `--host` beyond `127.0.0.1`/`localhost`
  without your own network-level access control.
- Error shape is the same everywhere: `{"error": "<message>"}` with an HTTP status code
  (`400` bad input, `404` not found, `409` a deployment change is already in progress, `413`
  body too large, `500` unexpected failure).
- Fields marked **required** cause a `400` with a descriptive message if missing or invalid.
  Fields marked **optional** have the stated default when omitted.

---

## GET `/api/bootstrap`

Full snapshot used to render the dashboard on load. No input.

**Shares:**
```jsonc
{
  "app_version": "0.1.0",
  "config_path": "F:/Projects/Dual_gpu_setup/example.dual_gpu.toml",
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
      "reasoning_budget": "int"
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

Aggregated analytics across all stored runs (not client-scoped). No input.

**Shares:**
```jsonc
{
  "generated_at": "ISO-8601 timestamp",
  "summary": {
    "total_runs": "int", "completed_runs": "int", "failed_runs": "int",
    "success_rate": "float | null (percent)",
    "models_used": "int",
    "input_tokens": "int", "output_tokens": "int", "thinking_tokens": "int",
    "visible_output_tokens": "int", "total_tokens": "int",
    "avg_tokens_per_second": "float | null", "avg_prefill_tokens_per_second": "float | null",
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
      "avg_latency_ms": "float | null", "avg_gpu_utilization_pct": "float | null",
      "peak_vram_bytes": "int"
    }
  ],
  "categories": [
    {
      "category": "string (a task_label value some /api/chat call used, e.g. 'resume_extraction')",
      "runs": "int", "success_rate": "float", "input_tokens": "int", "output_tokens": "int", "thinking_tokens": "int",
      "avg_tokens_per_second": "float | null", "avg_prefill_tokens_per_second": "float | null",
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
      "avg_latency_ms": "float | null", "avg_gpu_utilization_pct": "float | null",
      "peak_vram_bytes": "int"
    }
    // one row per (task_label, model_name) pair actually seen; this is what the dashboard's
    // Analytics tab filters by category and/or model
  ],
  "daily": [ { "date": "YYYY-MM-DD", "runs": "int", "tokens": "int" } /* last 14 days */ ],
  "recent_runs": [ /* 20 most recent, see GET /api/runs for row shape */ ]
}
```

## GET `/api/runs`

**Needs:** nothing (always returns the 100 most recent runs across all clients/callers).

**Shares:** `{"runs": [<run row>, ...]}` — each run row:
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
  "configuration": { /* full server/runtime configuration snapshot for this run */ },
  "output_text": "string | null", "reasoning_text": "string | null",
  "finish_reason": "string | null",
  "usage": { /* see GET /api/clients/schema */ },
  "timing": { /* see GET /api/clients/schema */ },
  "resources": { /* see GET /api/clients/schema */ },
  "error_text": "string | null",
  "backend_response": null,
  "client_id": "uuid | null (set only if this run was made via /api/chat with client_id)",
  "task_label": "string | null (set only if this run was made via /api/chat with task_label)"
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
  default. Must be `>= 0`.
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
(message includes each failure).

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

Stops every active server and returns to `idle`.

**Needs:** nothing (body ignored). **Shares:** same shape as `GET /api/status`.
**Errors:** `409` if another deployment change is already in progress.

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

---

## POST `/api/clients/register`

Register a consuming codebase once, up front.

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
