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
