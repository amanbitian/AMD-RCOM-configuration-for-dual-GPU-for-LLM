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

## 1b. Plan first: ask which lane each model fits, without loading anything

`POST /api/clients/check_fit` answers "would this model fit this lane at this context window"
as a read-only verdict. It starts no process, takes no deployment lock, and never disturbs
whatever is currently loaded, so it is safe to call in a loop — once per (model, lane) pair —
before you deploy anything at all:

```
POST /api/clients/check_fit
{
  "model": "gemma-4-26B-A4B-it-Q4_K_M",
  "lane": "9070xt",
  "context_window": 51200,
  "input_tokens": 19000,       // optional, same pairing rule as /api/clients/deploy
  "max_output_tokens": 12000
}
```

```jsonc
{ "fits": false, "reason": "gemma-4-26B... needs an estimated 27.03 GB ...", "resolved_context_window": null }
```

Do this rather than learning about fit by trying a deploy — a trial deploy actually loads
several GB of model onto a GPU just to tell you it does not belong there.

**Bucket your model list with it.** Test every model against your *smallest* lane. The ones
that fit there belong there; only the ones that fail belong on a larger lane. That single rule
is the whole placement policy this service uses internally (`deploy_parallel` applies exactly
the same test), and it is what keeps a small model from being promoted to the big GPU just to
fill a slot.

**One trap worth knowing:** `check_fit` reports *every* validation failure as `fits: false`,
not just capacity ones — including an unknown `lane` key. A typo there returns `fits: false`
for every model you test, which is indistinguishable from a genuine "nothing fits this GPU"
answer and will quietly push your entire list onto the other lane. Read lane keys from
`GET /api/bootstrap` -> `catalog.lanes[].key` instead of hardcoding them, and treat any `reason`
that is not about capacity as a bad request on your side. See
[API_CONTRACT.md](API_CONTRACT.md) for the full list.

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
one active deployment at a time.

## 2b. Running two models at once — `deploy_parallel`

If you need two models loaded concurrently (one per GPU lane) instead of one at a time, use
`POST /api/clients/deploy_parallel` instead of `/api/clients/deploy` — same idea, but for a list
of models instead of one, with the service auto-assigning each the best-fitting lane:

```
POST /api/clients/deploy_parallel
{
  "models": ["gemma-3-4b-it-Q4_K_M", "Qwen3.8-27B-Q6_K"],  // one per configured lane (today: 2)
  "context_window": 32768,      // optional, applied identically to every model in the list
  "input_tokens": 12000,        // optional, same sizing guarantee as /api/clients/deploy
  "max_output_tokens": 1024
}
```

Lane assignment follows a strict smallest-fit rule at the requested context. Every model that
passes the full capacity/KV-cache check on the smaller lane stays on that lane; only a model that
fails there may use the larger lane. Two models belonging to the same lane must be run
sequentially; the endpoint returns `400` instead of moving a small model to the larger GPU just
to fill both lanes. Response shape matches `GET /api/status`.

**Finding which lane each of your models landed on — read `profile.models[]`, not
`servers[]`.** `servers[].model` is the service's own *resolved short display name* (e.g.
`"gemma-3-270m-it-Q8_0"`), which will not match a catalog id you requested with (e.g.
`"lmstudio:lmstudio-community/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q8_0.gguf"`). A real client
integration hit this directly: matching against `servers[].model` silently failed for every
model, with no error — it just looked like the deploy worked while nothing was actually
distinguishable per-model. `profile.models[]` (`[{"model": <exact string you sent>, "lane":
<key>}, ...]`) echoes your input back verbatim and is the only reliable join key. See
[API_CONTRACT.md](API_CONTRACT.md)'s `/api/clients/deploy_parallel` entry for the full field
reference.

**After this call, talk to each lane directly — don't deploy again.** Both models are now
reachable by their own `target` (the lane key from `profile.models[]`). Send every subsequent
`/api/chat` call with your `client_id` + that fixed `target`, exactly like step 3 below. Note
that a deployment is not owned by a `client_id` — there is one deployment shared by the whole
service, and `client_id` only controls run attribution at chat time. Register once and reuse
that single id for both lanes; registering twice splits your own metrics across two client
records and two `output_path` files.

Do **not** call `/api/clients/deploy` or `/api/clients/deploy_parallel` again for either model
while you're still using the pair — either endpoint **replaces the whole deployment** (there's
still only one active deployment at a time, now spanning both lanes), so a second deploy call
would tear down the *other* lane's model too, not just refresh the one you're targeting. To
advance one lane without touching the other, use `deploy_lane` (step 2c).

## 2c. More than two models — one independent queue per GPU

`deploy_parallel` loads a *fixed pair*, so a batch of many models run that way makes both lanes
wait at a barrier: the fast lane sits idle until the slow lane finishes before the next pair can
load. For a real batch, give each GPU its own queue and let it advance on its own.

`POST /api/clients/deploy_lane` is what makes that possible — it replaces the model on the lane
you name and leaves every other lane's model running untouched:

```
POST /api/clients/deploy_lane
{
  "model": "gemma-4-26B-A4B-it-Q4_K_M",
  "lane": "r9700",
  "context_window": 51200,
  "input_tokens": 19000,
  "max_output_tokens": 12000
}
```

The full working sequence, in order:

1. **Register once** (step 1) and keep the one `client_id` for every lane.
2. **Bucket your model list with `check_fit`** (step 1b) into one queue per lane.
3. **Start each lane with `deploy_lane`.** You do *not* need a `deploy_parallel` first — on an
   idle service there is simply nothing to replace, so calling `deploy_lane` once per lane is
   how both GPUs come up. `mode` reads `single_gpu` after the first call and `parallel_models`
   once the second lane lands.
4. **Run one worker thread per lane**, each pinned to its lane's `target` for the whole life of
   that model: every `/api/chat` carries `client_id` + that fixed `target` and nothing else.
   The worker must never call a deploy endpoint other than `deploy_lane` for its own lane —
   `/api/clients/deploy` and `/api/clients/deploy_parallel` both replace the *entire* deployment
   and would tear down the other lane's live model mid-run. Enforcing this in your client (a
   chat wrapper that physically cannot deploy) is worth the few lines; the failure is silent
   otherwise.
5. **When a lane's model is done, that worker calls `deploy_lane` again** for its next model and
   carries on. It never waits for the other lane.

```python
import threading

client_id = register_once()          # step 1
queues = bucket_with_check_fit(...)  # step 1b -> {"9070xt": [...], "r9700": [...]}

def worker(lane, models):
    for model in models:
        deploy_lane(model=model, lane=lane, context_window=51200,
                    input_tokens=19000, max_output_tokens=12000)
        # Pinned to (client_id, lane) — cannot redeploy, so it cannot evict the other GPU.
        run_your_whole_pipeline(PinnedChatClient(client_id, target=lane), model)

threads = [threading.Thread(target=worker, args=(lane, models))
           for lane, models in queues.items()]
for t in threads: t.start()
for t in threads: t.join()
```

**Ordering rules the service enforces for you:** two lane workers that finish at the same moment
both get served — simultaneous `deploy_lane` calls are serialized server-side and the second
*blocks* rather than failing with `409`. Chat traffic on a preserved lane keeps working
throughout the other lane's model swap. The one rule you must hold yourself: don't replace a
lane while your own chat request on *that* lane is still outstanding.

**Finding your models in the response still works the same way.** `profile.models[]` carries the
exact `model` string you sent for the lane you just replaced, and forwards the strings the other
lanes were deployed with unchanged — so it stays the join key here too. `servers[].model` is
still the resolved short display name on every lane and still will not match a catalog id.

**One caveat if you are benchmarking.** GPU figures (`avg_gpu_utilization_pct`,
`peak_dedicated_vram_bytes`) are measured per model process, so concurrent lanes don't
contaminate each other's numbers. The whole-system fields — `peak_system_ram_used_bytes`,
`peak_system_pagefile_used_bytes` — are machine-wide by definition, so a chat measured while the
other lane happens to be loading a model reads high through no fault of the model being timed.
Compare those two fields only within the same execution mode, not between a parallel run and a
sequential one.

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
