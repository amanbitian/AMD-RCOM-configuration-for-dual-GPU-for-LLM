# Dual coding-agent setup

Dual GPU Studio can serve two coding agents at the same time while keeping the existing
evaluation and dashboard metrics intact. The recommended pairing is **Cline** as the developer
and **OpenCode** as the independent QA/reviewer. Put them on separate GPU lanes and separate Git
worktrees so they do not edit the same files concurrently.

## 1. Start the service and register the project

```powershell
py app.py --config .\example.dual_gpu.toml --no-browser
```

The service now binds to `127.0.0.1` by default. Register each repository once:

You can do the complete setup without these API commands: open the dashboard and use **Setup
wizard**. It creates the project, registers the shared metrics key, assigns models to both GPUs,
loads the parallel deployment, creates the developer/QA sessions, and displays copy-ready Cline and
OpenCode settings. The API example below is useful for automation or repeatable scripts.

```powershell
$body = @{
  project_name = "Cutline"
  github_repo = "github.com/you/cutline"
  output_path = "F:/Projects/Cutline/metrics/dual_gpu_runs.jsonl"
} | ConvertTo-Json

$client = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8090/api/clients/register `
  -ContentType application/json -Body $body

$client
```

Keep the returned `client_id`. Use the same value as the API key in both agents. This is how
requests from both tools are attributed to the same project. The Projects page can also register
repositories, but client registration additionally creates the API key and per-project JSONL
delivery path.

## 2. Load one coding model per GPU

Use the dashboard's **Two models** deployment, or call `POST /api/clients/deploy_parallel` as
documented in [CLIENT_API.md](CLIENT_API.md). A practical split is:

- larger GPU (`r9700`): stronger developer model, larger context;
- smaller GPU (`9070xt`): faster reviewer/test/debug model.

Do not point either agent directly at ports 8081 or 8082. Those are backend model-server ports
and bypass project attribution. Point both tools at port 8090, the instrumented gateway.

## 3. Configure Cline as the developer

Choose Cline's OpenAI-compatible provider and use:

```text
Base URL: http://127.0.0.1:8090/v1/r9700/developer
API key:  <the client_id returned above>
Model:    r9700
```

The URL pins Cline to the `r9700` lane and records its role as `developer`. Streaming is relayed
without waiting for the whole completion. Coding prompts and completions are redacted from the
telemetry database by default; token counts, timing, model identity, project, lane, and GPU samples
are still stored.

## 4. Configure OpenCode as QA/reviewer

Configure an OpenAI-compatible provider with:

```text
Base URL: http://127.0.0.1:8090/v1/9070xt/qa
API key:  <the same client_id>
Model:    9070xt
```

Give OpenCode a reviewer task: inspect the developer branch, run build/tests, report failures, and
only make repair commits when explicitly requested. Keeping the roles asymmetric avoids two agents
independently rewriting the same implementation.

## 5. Run them safely in parallel

Use two worktrees or two clones:

```text
F:/Projects/Cutline-dev   -> Cline, branch agent/developer
F:/Projects/Cutline-qa    -> OpenCode, branch agent/qa
```

Recommended loop:

1. Cline implements one bounded task and commits it.
2. OpenCode reviews that commit from its own worktree and runs the build/tests.
3. Record build and test results with `POST /api/tool-events` if the tool does not expose events
   automatically.
4. Cline repairs failures; OpenCode re-validates.
5. A human approves the final merge.

Avoid giving both agents write access to the same working tree. GPU parallelism is safe; concurrent
filesystem edits to one checkout are not.

## Gateway metadata and routes

The gateway supports:

```text
POST /v1/chat/completions
POST /v1/<lane>/chat/completions
POST /v1/<lane>/<agent-role>/chat/completions
GET  /v1/models
GET  /v1/<lane>/<agent-role>/models
```

Optional headers for clients that support custom headers:

```text
X-DGPU-Project: <project id or name>
X-DGPU-Agent-Role: developer | qa | reviewer
X-DGPU-Session: <session id>
X-DGPU-Task: <coding task id>
X-DGPU-Task-Label: <free-form category>
X-DGPU-Capture-Content: true
```

`X-DGPU-Capture-Content` is opt-in. Without it, coding-agent prompt and completion contents are not
persisted.

## What the dashboard tracks

Existing evaluation metrics remain available: input, cached, uncached, prefill, reasoning, visible
output and total tokens; TTFT, prefill/decode time and rates, end-to-end latency; CPU, RAM, page
faults, dedicated/shared VRAM, GPU utilization and spill suspicion. Coding runs add project, role,
session, task, workload and GPU-lane attribution. Overview totals use the complete matching history,
while Run history remains paginated/capped for responsive browsing.

The Projects page and global project selector filter token, model, agent, latency and GPU summaries.
Per-GPU throughput uses generated tokens divided by measured decode time when native decode timing is
available; the dashboard also retains the original average per-request rate for comparison.

## Coding-agent latency and reasoning

Coding gateway requests use the background resource sampler. They do not run synchronous
PowerShell GPU-counter queries before and after every completion. Very short requests may
have no fresh resource sample; resource metrics describe the available sampling window,
not an exact measurement at each request boundary.

Choose **Coding agent** before loading a deployment, or use the coding setup wizard.
Coding mode uses the model's **native reasoning effort**, with `--reasoning auto` and
`--reasoning-budget -1` (no separate thinking cap). The old numeric presets remain available
for evaluation mode. Changing workload in the UI does not silently restart a loaded server;
reload a capped evaluation deployment before using it for coding. The gateway rejects
such a deployment with a reload instruction because llama-server cannot override a
server-side cap by sending a request budget of -1.

The installed Qwen3.8-27B template declares **low**, **medium**, and **xhigh**, with **xhigh**
as its native default. It does not declare `high`. Choices are read from the GGUF template
and cached using file size and modification time. Unknown templates expose Model default
until their native choices are identifiable; we do not invent effort-to-token mappings.

## Speculative decoding (faster responses, same output)

To raise tokens/sec without changing accuracy, pair a model with a small **draft model** on
the same GPU lane. The draft proposes tokens and the target model verifies each one, so the
generated text is identical to running the target alone — only throughput changes. The draft
must share the target's tokenizer family (e.g. a Qwen3 0.6B draft for a Qwen3 target).

Configure it per model in the TOML (not the UI, so incompatible pairings can't be selected by
accident):

```toml
[[models]]
name = "Qwen3-Coder-30B"
path = ".../Qwen3-Coder-30B-Q4_K_M.gguf"
draft_model = ".../Qwen3-0.6B-Q4_K_M.gguf"  # path or glob, resolved like `path`
draft_max = 16        # optional; max tokens drafted per step (backend default if 0)
draft_min = 0         # optional
draft_p_min = 0.0     # optional; min draft-token probability to keep speculating
draft_gpu_layers = 999  # optional; draft layers on GPU (defaults to policy.gpu_layers)
```

These map to llama-server `--model-draft`, `--gpu-layers-draft`, `--device-draft` (pinned to
the target's lane), `--draft-max`, `--draft-min`, and `--draft-p-min`. The draft's footprint
is added to the lane's VRAM budget when the context window is auto-sized (an explicit
`context_window` still wins — size it yourself if you set one). Deploy fails fast if the draft
file is missing.

**Monitoring:** each run records `timing.draft_tokens`, `timing.draft_accepted_tokens`, and
`timing.draft_acceptance_rate` (when the backend reports them); the run detail shows the
acceptance percentage, and the analytics summary exposes `avg_draft_acceptance_rate`. Higher
acceptance means a better draft match and a larger speedup; a low rate means the draft is
costing more than it saves — pick a closer-matched or smaller draft, or disable it. Lossless:
accuracy is unchanged whatever the acceptance rate.

## Structured output and grammar-constrained tool calls (quality)

The dominant tool-use failure for local models is emitting tool-call JSON that does not parse.
To prevent it, the gateway passes three OpenAI/llama.cpp fields straight through to
llama-server, unchanged, on both the `/api/chat` coding path and the OpenAI gateway:

- `grammar` — a GBNF grammar string; the backend constrains decoding to it.
- `response_format` — `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {…}}`.
- `json_schema` — a bare JSON schema object.

These are validated before the request is sent: a non-string `grammar`, a `response_format`
with an unknown `type` or a json_schema form missing its schema, or setting both `grammar` and
`response_format` at once, each fail fast with a clear error instead of a cryptic backend
failure. Most harnesses (Cline, Aider, Roo) set these for you when a model is marked as
tool-capable; you can also send them directly from your own client.

**Monitoring:** every coding run records `usage.tool_calls_total`, `usage.tool_calls_malformed`
(no function name, or arguments that are not valid JSON), and `usage.tool_call_valid_rate`.
The run detail shows the validity percentage, and the *Coding reasoning performance* table has a
**Tool-call validity** column and exposes `avg_tool_call_valid_rate` per model and effort level.
Compare validity with and without a grammar/schema constraint to confirm it is lifting quality;
a rate below ~100% on a tool-heavy workload is the signal to constrain output.

## Conversation archive (full history, separate store)

Two databases, by design:

- **Metrics DB** (`chatbot.sqlite3`) — always on. Token counts, timing, resources, and
  configuration for every run. Prompts and responses are **redacted** here unless a caller
  sends `X-DGPU-Capture-Content: true`. This keeps it small and the dashboard fast.
- **Conversation archive** (`chat_history.sqlite3`, opt-in) — the full history of every app
  that uses this repo: complete input messages, tool schemas, output text, reasoning, tool
  calls, plus the same token counts, in their own file.

Enable it in the TOML:

```toml
[project]
store_chat_history = true
# chat_history_path = "G:/llm-archive/chat_history.sqlite3"  # optional; default is next to the metrics DB
```

**It never slows inference.** Writes go through an in-memory queue to a background worker that
batches the commits; the request path only does an O(1) enqueue, and JSON serialization plus
the DB write happen off-thread. If the queue ever fills, records are dropped rather than
applying backpressure to a model request. A per-request `X-DGPU-No-History: true` header (or
`"store_history": false` in an `/api/chat` body) skips archiving a sensitive turn.

**Read it back** with `GET /api/chat-history` — query params `project_id`, `start_date` /
`end_date` (YYYY-MM-DD, inclusive), `limit`, `offset`, and `content=1` to include the full
message/response bodies (omitted by default so listings stay light). Token columns
(`input_tokens`, `output_tokens`, `thinking_tokens`, `total_tokens`, …) are stored as real
columns, so future aggregation and export don't need to parse JSON. Because it's a plain
SQLite file in its own location, it can be copied, backed up, or queried by other tools
independently of the live server.

| Native stage | Thinking allowance | Monitored per run |
| --- | --- | --- |
| Model default | Model controlled; no fixed cap | Resolved default when declared by the template |
| Low / medium / high / xhigh, when supported | Model controlled; no fixed cap | Actual backend thinking tokens, observed streaming thinking span, decode rate, latency |

Run history shows the selected/effective effort and model-controlled budget. Analytics has
a **Coding reasoning performance** table grouped by model and effective effort, including
how many runs actually reported thinking-token counts. Missing counts stay unknown rather
than being estimated from streamed chunks. The observed thinking span measures time between
the first and last reasoning chunks; it is not native GPU decode time and excludes the
wait before the first chunk. Counts are not inferred by extra tokenizer requests.

## Inline autocomplete + agent (Continue)

The gateway serves both an agent surface (`/v1/.../chat/completions`) and an inline-autocomplete
surface (`/v1/.../completions` and `/v1/.../infill`, raw FIM). Autocomplete is a thin
passthrough — no recording or sampling — so it never slows inference. The dual-GPU split maps
cleanly onto the two needs: **agent/chat on the big lane, autocomplete on the small lane.**

For autocomplete, load a small, fast **FIM coder** model (e.g. Qwen2.5-Coder-1.5B or 7B) on the
smaller lane — a large reasoning model makes ghost-text unusable. Put a coder-instruct model on
the big lane for the agent (instruct coders are faster and more reliable at tool calls than
reasoning models).

[Continue](https://continue.dev) ties both together — paste into `~/.continue/config.json`
(replace lane keys/models with yours):

```jsonc
{
  "models": [
    { "title": "DualGPU Agent", "provider": "openai", "model": "dgpu-agent",
      "apiBase": "http://127.0.0.1:8090/v1/r9700/developer", "apiKey": "local" }
  ],
  "tabAutocompleteModel": {
    "title": "DualGPU Autocomplete", "provider": "openai", "model": "dgpu-autocomplete",
    "apiBase": "http://127.0.0.1:8090/v1/9070xt/autocomplete", "apiKey": "local",
    "useLegacyCompletionsEndpoint": true
  }
}
```

Continue posts chat to `.../developer/chat/completions` (agent, big lane) and autocomplete to
`.../autocomplete/completions` (FIM, small lane). Aider is a strong terminal alternative for the
agent half (`--openai-api-base http://127.0.0.1:8090/v1/<lane>/<role> --openai-api-key local`);
it has no autocomplete, so pair it with Continue's autocomplete if you want both.

The dashboard leaves **Max output** blank in coding mode. An explicit client output limit
is still respected and recorded, and the context window remains finite. Thinking consumes
part of the completion/context allowance. Compare task correctness and elapsed time as well
as decode throughput. A larger effort is not a guaranteed accuracy improvement.

API deployment example (use your model's catalog id):

```json
{
  "mode": "single_gpu",
  "workload_kind": "coding_agent",
  "model": {
    "model": "<catalog id>",
    "lane": "r9700",
    "context_window": 32768,
    "reasoning_effort": "default"
  }
}
```

The coding gateway accepts a per-request `reasoning_effort` override from the model's
available choices. Omit it to use the deployment selection. `default` uses the native
template default, not a literal effort named "default". Numeric request thinking caps are
rejected in coding mode; explicit client `none`/`enable_thinking: false` remains honored.

Keep an agent's configured context limit aligned with the loaded server, including room
for its output and any reasoning. Last night's Qwen3.8-27B Q6_K log contained a 35,510-token
request rejected by a 32,768-token server. Use the agent's history compaction before that
limit, or deploy a larger context after checking memory capacity. The gateway forwards
messages and tool schemas intact; it does not silently remove repository context.

For performance comparisons, keep model quantization, context length, prompt, sampling,
reasoning effort, and GPU placement the same. Compare warm follow-up requests separately
from cold prompts. `timing.tokens_per_second` is native or decode-duration-derived throughput;
it is null when decode timing is unavailable. `timing.end_to_end_tokens_per_second` includes
prefill and request overhead. Streaming requests ask for usage by default, while preserving
an explicit `stream_options.include_usage: false`. Prompt caching was already working in
the observed Qwen deployment and is still controlled by the caller/backend.

Validation notes for the October 9 change:

- The existing Qwen logs show roughly 23–25 decode tokens/sec, with cache reuse.
- Three local GPU-counter probes took 774, 789, and 792 ms. Removing two synchronous
  captures avoids that work on each request; this is a counter-cost measurement, not a
  measured increase in GPU decode throughput.
- Tests cover streamed tool-call preservation, tool-call TTFT, usage counts, missing-lane
  rejection, capture opt-in, failure handling, and the generated reasoning command.
- Dashboard aggregation now reads only metrics and selected reasoning metadata through a
  WAL reader without holding the inference writer lock. A local single-pass comparison over
  2,898 stored runs took 6,530 ms for full-row loading/decoding and 151 ms for the metrics
  projection (157 MB versus 5.4 MB serialized). This measures the database read/decode path,
  not total dashboard rendering or model throughput; cache state can affect the comparison.
- Both coding request paths avoid synchronous counters. SSE forwarding finishes at `[DONE]`
  rather than waiting for connection closure, and interrupted streams are marked failed.
- Aggregate decode speed excludes output tokens from runs missing decode timing, avoiding
  inflated rates when telemetry coverage differs across runs.
- Restart the gateway to use the updated code. No new GPU decode-speed or coding-accuracy
  result is claimed: the model servers were stopped during verification.
