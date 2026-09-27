# Client Integration API

Any other codebase running on this machine can drive the dual-GPU orchestrator over
plain HTTP while `app.py` is running (default `http://127.0.0.1:8090`). This is
additive to the existing dashboard UI — the UI keeps using `/api/deploy`, `/api/chat`,
etc. exactly as before. Nothing below changes how metrics are computed.

This is a walkthrough of the flow. For the exhaustive field-by-field request/response
reference of every endpoint (this one and the rest of the app's API), see
[API_CONTRACT.md](API_CONTRACT.md).

## 0. Make sure the service is actually running

Every endpoint below needs `app.py` already running — there's no HTTP "start" endpoint,
because if the service is down there's nothing listening to receive that request. If your
codebase wants to trigger the service on demand rather than assuming a human already started
it, run [launch_service.py](launch_service.py) as a command before calling anything else:

```
python launch_service.py --config F:/Projects/Dual_gpu_setup/example.dual_gpu.toml
```

This is a one-shot command, not an always-on daemon: it checks whether the service is already
healthy at the given host/port and exits immediately if so; otherwise it starts `app.py` as a
detached background process, waits until it reports healthy, prints its `base_url`, and exits.
The service itself keeps running after the launcher exits — the launcher does not stay
resident. Safe to call before every session; it's idempotent.

```python
# from any Python client, before using the API below
import subprocess, sys, json
result = subprocess.run(
    [sys.executable, "F:/Projects/Dual_gpu_setup/launch_service.py",
     "--config", "F:/Projects/Dual_gpu_setup/example.dual_gpu.toml"],
    capture_output=True, text=True, check=True,
)
base_url = json.loads(result.stdout)["base_url"]  # e.g. http://127.0.0.1:8090
```

## 1. Register your project once

```
POST /api/clients/register
{
  "project_name": "LLM_evaluator",
  "github_repo": "github.com/you/LLM_evaluator",   // optional
  "output_path": "F:/Projects/LLM_evaluator/metrics/dual_gpu_runs.jsonl"
}
```

`output_path` is a file path in *your* repo. Every run you attribute to this
client (step 3) gets appended there as one JSON line — so your codebase ends up
with its own durable copy of the performance data, independent of this repo's
own SQLite store.

Response:
```
{ "client_id": "...", "project_name": "...", "github_repo": "...", "output_path": "...", "created_at": "..." }
```
Save `client_id` — you pass it on every later call.

## 2. Ask to run a model — this service gives the final verdict

```
POST /api/clients/deploy
{
  "model": "Qwen3-32B-GGUF",   // model name or catalog id from GET /api/bootstrap -> catalog.models
  "gpu": "r9700",              // optional lane key; omit to let the service pick automatically
  "context_window": 32768,     // optional; a policy default is used if omitted
  "input_tokens": 12000,       // optional; largest prompt size you plan to send
  "max_output_tokens": 1024    // optional; must be passed together with input_tokens
}
```

The service resolves a GPU lane (or uses the one you named), checks the model's file size
*and* an estimated KV-cache cost for the resolved `context_window` against that lane's safe
VRAM budget, and either:

- **loads it and returns `200`** with the resolved device/context (same shape as
  `/api/status`), or
- **returns an error** (e.g. 400) explaining exactly why it doesn't fit — same
  validation used by the dashboard's "Load deployment" button, so the answer is
  authoritative, not a guess made on the client side.

The KV-cache estimate is read straight from the model's own GGUF metadata (layer count, head
counts, sliding-window pattern where the file exposes one) — no config beyond the model file
itself is needed. For a handful of architectures the estimate isn't confident (hybrid state-
space/attention designs, or a sliding-window scheme whose GGUF doesn't record a per-layer
pattern) and the check quietly falls back to file-size-only, same as before — it will never
block a deploy on a guess it isn't sure about.

**Tell it your real token budget.** If you know how large your prompts and
expected completions actually are, pass `input_tokens` and `max_output_tokens`
(both required together). The service enforces:

```
input_tokens + max_output_tokens + safety_tokens <= effective_context_window
```

and bumps the allocated `context_window` up to cover it — even above whatever
`context_window` you requested or the policy default would have picked. If the
required size is larger than `policy.ctx_max`, the deploy is **rejected with a
400** telling you the exact numbers, instead of silently loading a server that
will reject your requests mid-run with `exceed_context_size_error`. This is the
failure mode you hit evaluating resumes at `n_ctx=8192` against ~19k-token
prompts — pass `input_tokens`/`max_output_tokens` and it can't happen again.

Note: like the dashboard, this replaces whatever was previously loaded — there is
one active deployment at a time. If you need two models loaded concurrently, use
the existing `/api/deploy` with `mode: "parallel_models"` (see the dashboard's
"Two models" tab for the shape).

## 3. Run a prompt and get metrics back — tagged to your client

Use the existing chat endpoint, adding your `client_id`:

```
POST /api/chat
{
  "client_id": "...",
  "target": "r9700",              // or omit for the only/first active model
  "task_label": "resume_extraction",   // optional: tag what kind of call this was
  "prompt": "..."
}
```

`task_label` is a freeform string — no fixed enum, use whatever names your pipeline's task
types (`"relevance"`, `"resume_extraction"`, `"fraud_precheck"`, `"fraud_d2_fusion"`, ...). It
gets stored on the run and rolled up per (category, model) pair in `/api/dashboard`, which is
what feeds the dashboard's **Analytics** tab (see step 6).

The response is unchanged from today's `/api/chat` (per-target `usage`, `timing`,
`resources`, etc. — see the schema below) plus one new field:

```
"client_delivery": { "written": true, "path": "F:/Projects/LLM_evaluator/metrics/dual_gpu_runs.jsonl" }
```

confirming whether the JSON line was appended to your `output_path`. A write
failure there (e.g. bad path) never fails the chat call itself — you still get
the metrics in the response either way.

## 4. What data you get

`GET /api/clients/schema` returns the exact field list below at runtime.

| Group | Field | Meaning |
|---|---|---|
| usage | `input_tokens` | Prompt tokens sent |
| usage | `cached_input_tokens` | Prompt tokens served from KV cache reuse |
| usage | `uncached_input_tokens` | `input_tokens - cached_input_tokens` |
| usage | `prefill_tokens` | Tokens processed during prefill |
| usage | `thinking_tokens` | Reasoning tokens, if any |
| usage | `visible_output_tokens` | `output_tokens - thinking_tokens` |
| usage | `output_tokens` | Total completion tokens |
| usage | `total_tokens` | `input_tokens + output_tokens` |
| timing | `time_to_first_token_ms` | Latency to first output token |
| timing | `prefill_duration_ms` | Wall time for prefill |
| timing | `decode_duration_ms` | Wall time for generation |
| timing | `end_to_end_duration_ms` | Total request wall time |
| timing | `prefill_tokens_per_second` | Prefill rate |
| timing | `tokens_per_second` | Decode rate |
| resources | `avg_gpu_utilization_pct` / `peak_gpu_utilization_pct` | GPU load during the run |
| resources | `peak_dedicated_vram_bytes` | Peak VRAM used |
| resources | `peak_shared_gpu_memory_bytes` / `spill_suspected` | VRAM-spill indicators |
| resources | `peak_process_rss_bytes` / `peak_process_private_bytes` | Peak process RAM / committed memory |
| resources | `process_page_fault_delta` | Page faults incurred by the model process during this run |
| resources | `peak_system_ram_used_bytes` / `peak_system_pagefile_used_bytes` | Whole-system RAM / commit-charge pressure during the run |

## 5. Fetch your own run history any time

```
GET /api/clients/{client_id}/runs
```

Returns your last 100 runs from this service's own store (in addition to what
was appended to your `output_path`), in case you need to re-fetch or you skipped
writing a local copy.

## 6. See it broken out on the dashboard, by category and model

Once some runs carry a `task_label`, open the dashboard's **Analytics** tab (a nav item
alongside Overview / Chat / Run history). Two dropdowns — Category and Model — filter a
breakdown table (runs, success rate, decode tok/s, prefill tok/s, avg latency, peak VRAM) so
you can compare, say, `resume_extraction` on the 4B model against the 26B model side by side,
exactly like a hand-built benchmark table but always current. The same data is available
programmatically as `categories`/`category_models` in `/api/dashboard` — see
[API_CONTRACT.md](API_CONTRACT.md#get-apidashboard) for the exact shape.

Runs made without a `task_label` are unaffected — they still count in the dashboard's
Overview tab and per-model figures, just not in this category breakdown.
