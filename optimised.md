# Optimization & Bug-Fix Report — 2026-09-27

This is the narrative version of what's in [CHANGELOG.md](CHANGELOG.md): not just what
changed, but how each issue was actually found and confirmed before anything was fixed.
Every claim below was checked against real evidence (real GGUF files, real `llama-server`
output, real measured VRAM) rather than assumed — including one claim that turned out to be
wrong and was dropped rather than "fixed."

## Background

After building the Client Integration API and telemetry pipeline earlier the same day, the
question was asked directly: **"are there gaps in this codebase?"** Rather than answer from
general impression, each candidate gap was checked against the actual code before being
reported. This document covers what happened next, when the answer was "yes, fix them."

---

## Bug 1: deploy could say "fits" and still OOM

### How it was found

The dashboard's "final verdict" feature (does a model fit on a GPU?) was traced through the
code path that actually approves a deploy: `DeploymentManager._validate_single_fit()` in
`app.py`. It checked exactly one thing:

```python
if size > lane_capacity_gb(self.config, lane):
    raise AppError(...)
```

`size` is the model **file** size in GB. Nowhere in that function, or anywhere it's called
from, is the requested **context window** considered. A separate feature added earlier the
same day (`resolve_context_window`, which grows the context to fit `input_tokens +
max_output_tokens + safety_tokens`) only guarantees the context is big enough to hold the
prompt — it says nothing about whether that context's KV cache actually fits in VRAM.

Grepping `orchestrator.py` for `parallel` and for any KV-cache-related term confirmed there
was no VRAM math anywhere beyond the crude file-size check and a rough headroom heuristic
(`ctx_for`) that only applies when no context is explicitly requested. The moment a caller
requested a large context explicitly, or `resolve_context_window` bumped it up to satisfy a
prompt-size guarantee, nothing re-checked feasibility.

A second gap was found in the same pass: `_validate_profile`'s `single_large_model` branch
called `_configured_model(...)` but never called `_validate_single_fit` at all — multi-GPU
deploys had **no capacity check whatsoever**, not even the existing file-size one.

### How it was fixed

A new module, `dual_gpu_setup/gguf.py`, reads just the metadata header of a GGUF model file
(architecture name, layer count, head counts, embedding size — never the multi-gigabyte
tensor data) using nothing but Python's standard library. `estimate_kv_cache_gb()` in
`orchestrator.py` uses that metadata to compute the standard transformer KV-cache formula:
`2 (K+V) × kv_heads × head_dim × bytes_per_element × context_size`, summed per layer.

`_validate_capacity()` (replacing `_validate_single_fit`) now checks
`model_file_gb + kv_cache_gb` against the lane's VRAM budget, and is called from **all three**
deploy modes, closing both gaps in the same change.

### How it was verified

Real GGUF files already on disk were used to test this, not synthetic ones (synthetic
fixtures came later, for the automated test suite). This surfaced two more bugs before the
fix was trusted:

**Bug 1a — wrong head dimension.** The formula initially derived `head_dim` as
`embedding_length / head_count`. On a real Qwen3.5 model file, that division came out to
`5120 / 24 = 213.33` — not an integer, which is architecturally impossible (head dimension is
always a whole number). Dumping every scalar field in that file's GGUF metadata found the
real answer: the file explicitly states `attention.key_length = 256` and
`attention.value_length = 256`, which do **not** match the derived value. Many modern
architectures store this explicitly precisely because it doesn't always divide evenly. Fixed
by preferring the explicit fields, falling back to the derived value only when they're absent
and the division is exact.

**Bug 1b — hybrid architectures don't have a normal KV cache.** The same Qwen3.5 file (which
happens to be the model live-deployed on this machine at the time) turned out to have
`qwen35.ssm.*` metadata keys — it's a hybrid state-space/attention architecture, where most
layers use a fundamentally different, roughly context-independent memory mechanism, not a
per-token KV cache at all. Treating every layer as standard attention would have wildly
over-estimated its VRAM cost. The estimator now detects any `*.ssm.*` key and returns "not
confidently estimable" rather than a number — callers fall back to the pre-existing
file-size-only check for these models, unchanged from before this fix.

**Sliding-window handling.** Two real Gemma files were checked (`gemma3`, `gemma4`).
`gemma4`'s metadata included an explicit per-layer `attention.sliding_window_pattern` array
(`[True, True, True, True, True, False]`, repeating — 5 local layers per 1 global layer, which
matches Gemma's known public architecture). The estimator uses this to cap those layers' KV
cost at the sliding window (1024 tokens) instead of the full requested context, and the
resulting real-file estimate (1.37 GB at a 51,200-token context) was confirmed smaller than
the naive full-context estimate for the same file — the pattern-aware code path is doing real
work, not a no-op. `gemma3` exposes only a **scalar** `sliding_window` with no per-layer
pattern, so the local/global ratio for that file is unknown; the estimator falls back to
treating every layer as full context there, which over-estimates (the safe direction for a
feasibility gate — it can produce an overly cautious rejection, never a false "fits").

**End-to-end integration test.** A throwaway two-lane config was built pointing at a real
32B model file, with the lane VRAM deliberately sized so the model file alone fit but
file+KV-cache did not at a large context. Three cases were run directly against
`DeploymentManager._validate_profile()`:

```
lane capacity: 22.50 GB (25GB * 0.9)
CASE 1 (ctx=32768): correctly REJECTED -> deepseek-32b-test needs an estimated 27.03 GB
  (19.03 GB model file + ~8.00 GB KV cache at 32768 context tokens) but testlane has a
  safe budget of 22.50 GB...
CASE 2 (ctx=4096): correctly PASSED
CASE 3 (bumped by input_tokens): correctly REJECTED -> ...needs an estimated 26.91 GB
  (19.03 GB model file + ~7.88 GB KV cache at 32256 context tokens)...
```

The `single_large_model` gap and the SSM-hybrid fallback were verified the same way before
either was called fixed.

---

## Retraction: a claimed gap that wasn't confirmed

An earlier report of "gaps" in this codebase included: *"`parallel` (concurrent request
slots) is invisible to the VRAM math."* Before writing any code for it, the actual
`llama-server --help` output was captured on this machine (not assumed from memory):

```
-c,    --ctx-size N     size of the prompt context (default: 0, 0 = loaded from model)
-kvu,  --kv-unified, -no-kvu, --no-kv-unified
                         use single unified KV buffer shared across all sequences
                         (default: enabled if number of slots is auto)
```

This describes `--ctx-size` as the shared/total KV pool, not a per-slot allocation multiplied
by `--parallel`. Since this repo always passes an explicit `parallel` value (default `1`,
never `-1`/auto) and no model in the current config uses `parallel > 1`, there was no way to
confirm an actual bug from the evidence available. Rather than build a fix for something
unverified, the claim was retracted and no code was written for it. This is documented here
specifically as a marker of the standard applied: a claimed gap without confirming evidence
doesn't get "fixed," it gets dropped.

---

## Bug 2 (found while testing Bug 1's fix): config loader crashed on incomplete TOML

### How it was found

Writing an integration test for the capacity-check fix required a minimal test config. The
first attempt omitted most of `[policy]` (relying on documented defaults) and crashed:

```
TypeError: int() argument must be a string, a bytes-like object or a real number,
  not 'member_descriptor'
```

`load_config()` used patterns like `policy_raw.get("reasoning_budget", PolicyConfig.reasoning_budget)`
— accessing the default off the **class**. `PolicyConfig` is a `@dataclass(slots=True)`, and
for a slotted dataclass, `ClassName.field` returns the slot descriptor object, not the actual
default value; only `ClassName().field` (an instance) does. Confirmed with a two-line repro:

```python
@dataclass(slots=True)
class C:
    x: int = 5

C.x        # <member 'x' of 'C' objects>   <- not 5!
C().x      # 5
```

This bug was silent in every real usage of this repo up to now purely because the example
config specifies every single `[policy]`/`[project]` field explicitly — the buggy fallback
path was simply never exercised. It would bite anyone who wrote a TOML config that omitted
even one optional field.

### How it was fixed

`load_config` now builds one default instance per config section
(`ProjectConfig()`, `DiscoveryConfig()`, `PolicyConfig()`) up front and reads fallback values
from those instances instead of from the classes.

### How it was verified

A minimal TOML omitting `[project]` entirely and all of `[policy]` was loaded successfully
after the fix, returning the correct documented defaults (`reasoning_budget=0`,
`ctx_max=49152`, `health_poll_seconds=1.0`, etc.), and the real production
`example.dual_gpu.toml` was re-loaded afterward to confirm no regression. Both are now
regression tests in `tests/test_config.py`.

---

## Supporting fixes verified the same way

- **`task_label` normalization** — confirmed `"  Resume_Extraction  "` and `"resume_extraction"`
  now normalize to the identical stored value before this could fragment the dashboard's
  Analytics tab into duplicate categories.
- **Optional data retention** (`RunStore.prune_older_than`) — tested that an old run's
  `metric_samples` rows are removed via the existing `ON DELETE CASCADE` foreign key when the
  parent run is pruned, and that a non-positive `days` value is rejected.
- **Automated test suite** — 37 tests, all passing, using synthetic GGUF fixtures
  (`tests/conftest.py`) built from scratch (a small GGUF binary writer) so the suite never
  depends on any real model file being present on disk. Confirmed the suite still passes
  after every fix above, not just once at the end.

---

## Optimization: batch-evaluating many models stopped wasting a GPU

**Different session, same day, separate from the KV-cache/VRAM bugs above** — this is the
narrative for the client-integration work that made a *batch* of models (a talent-eval project
running its whole pipeline once per model, `talent-llm-eval`'s `model_constants.MODELS_TO_RUN`)
actually use both GPUs, instead of one lane sitting idle while the other worked through the
whole list alone.

### The problem, found by asking directly

Aman asked "aren't we using dual gpu in parallel?" The honest answer was no: the client's own
transport (`DualGpuClient`) only ever called `POST /api/clients/deploy` (`deploy_for_client`,
`single_gpu` mode) — one model, one lane, on demand. The only time both lanes had run
concurrently was a **manual** one-off `POST /api/deploy` with `mode: "parallel_models"`, done by
hand outside any client's code, days earlier. Nothing in this service made parallel usage the
*default* for a batch — a caller had to already know which model belongs on which lane to use
`parallel_models` mode at all, since `deploy_for_client`'s auto-lane-pick only existed for one
model at a time.

### The fix: three additive, generic endpoints — nothing client-specific baked into the service

1. **`POST /api/clients/deploy_parallel`** — the 2-model equivalent of `deploy_for_client`. Give
   it a plain list of model names (one per configured lane), it auto-assigns each the
   best-fitting lane (largest lane first, each time taking the largest remaining model that
   still fits it) and deploys via the *existing* `deploy()`/`_validate_profile("parallel_models",
   ...)` path. Zero duplicated capacity/KV-cache/pin-lane validation — every check the bugs above
   added applies here automatically, for free.
2. **`POST /api/clients/check_fit`** — read-only: "would model X fit lane Y at context Z,"
   without loading anything. Reuses `_configured_model` + `_validate_capacity` (same code the
   deploy path uses) but catches the resulting `AppError` and returns `{"fits": false, "reason":
   ...}` instead of raising. Exists so a client can *plan* a schedule ahead of time — e.g.
   partition a whole model list into "fits the small lane" vs. "needs the big lane" buckets
   before deploying anything — rather than discovering fit only by trial deploy, which would
   actually load a model just to find out it doesn't fit.
3. **`launch_service.py`** already existed for auto-starting the service on demand — it just
   wasn't documented as the answer to "the client can't reach the orchestrator." Nothing new
   here; the gap was that no client was actually calling it yet.

**A real integration bug this approach caught, not just avoided:** the first live test of
`deploy_parallel` matched requested model names against the response's `servers[].model` — which
turned out to be the service's own *resolved short display name* (e.g. `"gemma-3-270m-it-Q8_0"`),
never equal to a full catalog id a caller requests with. Every match silently failed; the client
would have fallen back to sequential every time, with **no error at all** telling anyone it never
actually ran in parallel. `profile.models[]` (echoes the caller's exact input strings back,
paired with the resolved lane) is the reliable join key — this was true of `/api/deploy` and
`/api/clients/deploy` too, just newly exposed by having two models to disambiguate between at
once. Documented as an explicit callout in both `API_CONTRACT.md` and `CLIENT_API.md` now, so the
next integration doesn't rediscover it the same way.

### A second, unrelated bug this work surfaced: console windows flashing every 2 seconds

Once a client actually started driving `app.py` via `launch_service.py` (which runs it detached,
with no console, so it can survive as a background service), `ResourceSampler`'s GPU-metrics poll
— PowerShell `Get-Counter`, unflagged, called every `interval_seconds` (2s) for the service's
entire lifetime — popped a brand-new console window on Windows every single tick, forever. Fixed
with `creationflags=subprocess.CREATE_NO_WINDOW` on every helper-process call in
`dual_gpu_setup/server.py` and `dual_gpu_setup/lmstudio.py` (not `tasks.py` — that's the separate
CLI `run`/`plan`/`inspect` path, always run in a user's own foreground terminal, so it never
lacked a console to begin with). Verified by stopping the pre-fix service, confirming silence
with nothing running, restarting with the fix, and having the integrating project's user confirm
directly that the flashing stopped.

### How the pieces above were verified live, not just at the code level

Deployed two real models (`gemma-3-270m-it-Q8_0` + `SmolLM2-360M.Q8_0`) via `deploy_parallel`:
best-fit correctly put the larger one on the larger lane. Ran real concurrent chat completions
against both — both threads started and finished together (not one-then-the-other), each
returned distinct real `timing`/`resources` telemetry (different tokens/sec, different GPU
utilization%, different peak VRAM), confirming two genuinely separate backends answered, not one
lane silently double-serving both requests under two different labels.

### Update: `check_fit` became the foundation for two further pieces, not left as a standalone check

Two more changes landed directly in this repo after the above (see `CHANGELOG.md`'s "Independent
per-lane model replacement" and "Context-aware smallest-fit GPU placement" entries — not written
by the same pass that wrote this section, but building straight on `check_fit`):

- **`deploy_parallel_for_client` was rewritten to use `check_fit` for real placement, not a size
  heuristic.** Every requested model is tested against the small lane first; if it fits there, it
  belongs there — only a model that fails that check is allowed onto the large lane. Two models
  that both fit the small lane (or both need the large one) are now a **rejected** pairing
  (`400`, explicit reason), not silently mis-assigned — a caller has to pair one small-fitting
  model with one large-only model, which is the actual constraint this was missing before.
- **`POST /api/clients/deploy_lane`** replaces just one lane's model, leaving every other active
  lane's model running untouched. This is what makes a true work-conserving scheduler possible:
  a caller with a whole list of models (not just 2) can run two independent per-lane queues — the
  small lane keeps advancing through every model that fits it, the large lane independently keeps
  advancing through the rest — instead of both lanes waiting at a fixed-pair barrier for
  `deploy_parallel`'s next call.

**Now closed (2026-09-28):** `talent-llm-eval`'s batch runner wired up that two-independent-queue
scheduler — `main.py`'s `run_parallel_lane_queues` buckets its model list with `check_fit`, starts
each lane with `deploy_lane`, and runs one worker thread per GPU, each pinned to its lane so it
cannot evict the other. Both GPUs now advance through their own queue with no fixed-pair barrier.

Being the first real consumer of these endpoints immediately exposed two gaps in *this* repo,
both now fixed (see `CHANGELOG.md`, 2026-09-28):

- **`deploy_lane` broke the join key this repo documents in bold.** `profile.models[]` is
  documented as the only reliable way to match requested model strings back to lanes, because
  `servers[].model` is a resolved short display name. `deploy()` honors that by storing the
  caller's raw profile — but `deploy_lane_for_client` rebuilt the profile from live handles using
  the resolved name, so the documented join key was wrong for precisely the endpoint the
  documented scheduler is built on. The client only worked because it had independently decided to
  match on `servers[].target` instead, which no document suggested. Now a lane replace carries the
  caller's own string for the replaced lane and forwards the recorded strings for preserved lanes.
- **The sequence itself was never written down here.** `check_fit` appeared nowhere in
  `CLIENT_API.md` and `deploy_lane` had one trailing sentence, so the only description of the
  working recipe lived in the *consumer's* repo, where the next integration would never find it.
  `CLIENT_API.md` now has it as steps 1b and 2c, and `README.md` no longer implies the CLI's
  internal scheduler is what an HTTP client gets.

The lesson worth keeping: the endpoints were generic and correct, and the docs were thorough
enough to look finished — but the first real integration still had to discover the join key the
hard way, because nobody had checked the bolded claim against the one code path added last.

---

## What this means going forward

The methodology applied throughout, worth calling out explicitly since it's the actual point
of this document: every fix above was checked against real files, real tool output, or a
real repro *before* being called correct — including the one candidate gap that got dropped
because it couldn't be confirmed. See [CHANGELOG.md](CHANGELOG.md) for the terse version of
each change, and `tests/` for the regression coverage that now guards all of it.
