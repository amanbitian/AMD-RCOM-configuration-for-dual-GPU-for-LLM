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

## What this means going forward

The methodology applied throughout, worth calling out explicitly since it's the actual point
of this document: every fix above was checked against real files, real tool output, or a
real repro *before* being called correct — including the one candidate gap that got dropped
because it couldn't be confirmed. See [CHANGELOG.md](CHANGELOG.md) for the terse version of
each change, and `tests/` for the regression coverage that now guards all of it.
