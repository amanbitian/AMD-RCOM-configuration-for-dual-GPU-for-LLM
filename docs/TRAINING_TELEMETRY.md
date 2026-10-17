# Training Telemetry, Reliability, and Checkpoint Evaluation

## Purpose

Extend the inference-focused orchestrator so that a PyTorch training job (first target: LoRA/SFT of a small model on the R9700) is measurable, recoverable, and evaluated continuously — without routing training through `llama-server` and without disturbing the existing inference path.

Every training experiment must answer four questions at once:

1. Did the model improve?
2. How fast did it train?
3. What hardware did it consume?
4. What limited it?

[TELEMETRY_AND_STORAGE.md](TELEMETRY_AND_STORAGE.md) principles apply unchanged: raw counters plus derived rates, monotonic durations, base units, a recorded source for every field, and `null` (never `0`) for anything unmeasured.

## Hardware facts that shape the design

| Item | Value | Consequence |
|---|---|---|
| GPU 0 | Radeon AI PRO R9700, 32 GB, Navi 48, 64 CU, 640 GB/s, blower cooler | Primary trainer. Sustained junction/memory temperature must be watched. |
| GPU 1 | Radeon RX 9070 XT, 16 GB, Navi 48, 64 CU, 640 GB/s | Same compute class as GPU 0; differs mainly in **memory capacity**. Speed must still be measured, not assumed equal. |
| Interconnect | PCIe 5.0, x8/x8 expected | Verify per run from sysfs; risers/BIOS can silently negotiate down. |
| OS for training | Ubuntu + ROCm | ROCm PyTorch on RDNA4 is a Linux-first stack. Inference remains supported on both Windows and Linux. |

Memory limits per strategy are **upper bounds on model/optimizer state**, not usable capacity. Activations, all-gather buffers, workspace, and allocator overhead must also fit on each card individually:

| Strategy | Hard ceiling | Practical note |
|---|---|---|
| Single GPU | 32 GB or 16 GB | Baseline; always measure first. |
| DDP | Full replica must fit on the 16 GB card | Balanced compute makes DDP attractive for 1–3B LoRA. |
| FSDP / ZeRO | < 2 × 16 GB of sharded state | Even sharding means the 16 GB card sets the limit. |
| Pipeline | < 48 GB physical | Each stage must fit its card; pipeline bubbles cost throughput. |
| Independent jobs | 32 GB and 16 GB separately | No communication; this is the first dual-GPU mode to adopt (train + live eval). |

Use the memory forecaster and measured peaks (see [Memory forecasting](#memory-forecasting)) before trusting any of these.

## Scope boundary

| Responsibility | Lives in | Notes |
|---|---|---|
| Platform probes (process, system, GPU) for Windows and Linux | this repo | Shared by inference and training. |
| Stable GPU identity, GPU leases | this repo | Prevents inference deploys from landing on a training GPU. |
| Tracker client, ingestion API, `training.sqlite3`, dashboard | this repo | The tracker client is the contract between projects. |
| Checkpoint → GGUF → `llama-server` evaluation orchestration | this repo | It already owns `llama-server` lifecycle. |
| Training loop, dataset build, splits, task metric computation | the training project (e.g. TalentTensor) | Logged into this repo through the tracker. |

Reuse what the training framework already provides instead of reimplementing it here:

- Hugging Face `Trainer` already saves optimizer, scheduler, RNG, and sampler state and resumes via `resume_from_checkpoint`.
- `EarlyStoppingCallback` and `load_best_model_at_end` already cover best-checkpoint selection.
- Recent `Trainer` versions normalize loss by token count across gradient accumulation (`num_items_in_batch`).

This repo **adds verification** around those features (manifests, integrity checks, resume tests, equivalence tests); it does not replace them.

## Phase 0: minimum for the first R9700 LoRA run

Build only this before the first real run, then let measured data drive the rest:

1. Linux platform probes using sysfs and `/proc` (no subprocess per tick).
2. `pci_bus` lane identity and GPU leases.
3. Tracker client with the P0 speed, loss, and memory metrics plus an environment snapshot.
4. `training.sqlite3` and the ingestion endpoints.
5. Progress-aware heartbeat and the `stalled` run state.
6. Checkpoint manifests (written last; no manifest = incomplete checkpoint).
7. A minimal Training tab: run list, loss/LR, tokens/sec, VRAM, temperatures, power.

## 1. Platform layer

At design time, the platform work was to move the existing Windows code behind an interface and add
a Linux implementation. This now lives under `dual_gpu_setup/hostos/`:

```text
dual_gpu_setup/hostos/
  base.py        # ProcessProbe, SystemProbe, GpuProbe, ProcessControl protocols
  windows.py     # current ctypes + PDH code moved from app.py / server.py
  linux.py       # /proc, sysfs, optional amdsmi
```

The originally Windows-only code paths and their implemented Linux equivalents were:

| Location | Original Windows path | Linux implementation |
|---|---|---|
| `app.py` `ResourceSampler._process_metrics` / `_system_metrics` | `ctypes.windll`; returns `{}` elsewhere | `/proc/<pid>/stat`, `/proc/<pid>/status`, `/proc/stat`, `/proc/meminfo` (or `psutil`) |
| `server.py` `gpu_process_metrics_batch` | PowerShell `Get-Counter` | sysfs device counters + `/proc/<pid>/fdinfo` (`drm-memory-vram`, `drm-engine-gfx`) |
| `server.py` `_pid_listening_on`, `_is_llama_server` | `netstat`, `tasklist` | `/proc/net/tcp` + `/proc/<pid>/exe` (or `psutil`) |
| `server.py` process kill | `taskkill /F /T` | process-group `SIGTERM` then `SIGKILL` (`start_new_session=True` at spawn) |
| `lmstudio.py` runtime discovery | `llama.cpp-win-x86_64-*`, `llama-server.exe` | also match `llama.cpp-linux-*`, binary without `.exe`, or an explicit `server_binary` path in config |

### Linux GPU sources (amdgpu)

| Field | Source |
|---|---|
| GPU busy % | `/sys/class/drm/cardN/device/gpu_busy_percent` |
| Memory busy % | `mem_busy_percent` (where exposed) |
| VRAM used / total | `mem_info_vram_used`, `mem_info_vram_total` |
| GTT used (Linux analog of Windows shared memory; spill signal) | `mem_info_gtt_used` |
| Edge / junction / memory temperature | `hwmon/hwmon*/temp{1,2,3}_input` (check `temp*_label`) |
| Board power | `hwmon/hwmon*/power1_average` or `power1_input` |
| Core / memory clock | `hwmon/hwmon*/freq{1,2}_input`, `pp_dpm_sclk`, `pp_dpm_mclk` |
| Fan | `hwmon/hwmon*/fan1_input` |
| PCIe link | `current_link_speed`, `current_link_width`, `max_link_*` |
| Throttle status, energy counter | `gpu_metrics` blob or `amdsmi` (best effort) |
| Per-process VRAM / GFX time | `/proc/<pid>/fdinfo/*` |

AMD SMI coverage on consumer Radeon parts is narrower than on Instinct; fields that return N/A must be stored as `null` with `unavailable_reason`. Verify coverage on the actual install before designing panels around a field.

Sampler rules:

- Runs in the dashboard process (or a sidecar), never inside the training process.
- 1 s cadence for device metrics; sysfs reads are cheap enough for this on Linux.
- Records its own sample duration and dropped-sample count (monitoring overhead).

## 2. GPU identity and leases

### Stable identity

Add `pci_bus` to each lane:

```toml
[[lanes]]
key = "r9700"
display = "AMD Radeon AI PRO R9700"
match = "R9700"
pci_bus = "0000:03:00.0"
vram_gb = 32
port = 8081
backend = "rocm"
```

The PCI bus ID joins sysfs card paths, AMD SMI devices, `torch.cuda` device indices (`get_device_properties(i)` exposes PCI fields in recent PyTorch; otherwise map through HIP ordering), `HIP_VISIBLE_DEVICES`, and llama.cpp's `ROCm0`/`ROCm1` names. Name matching stays as a fallback; on Linux, marketing names are not reliable.

### Leases

A lease is an exclusive claim on one or more lanes:

```text
POST   /api/leases           {holder: "training:<run_id>", lanes: ["r9700"], ttl_s: 120}
POST   /api/leases/{id}/renew
DELETE /api/leases/{id}
```

- `DeploymentManager.deploy*` rejects a deploy onto a leased lane with HTTP 409 (or queues it, if requested).
- Leases expire without renewal, so a crashed trainer does not lock a GPU forever. The tracker's heartbeat renews it.
- The sampler adds lease-holder PIDs to its watch list so training processes are sampled like `llama-server` ones.

## 3. Tracker client

A stdlib-only module imported by the training script:

```python
from dual_gpu_setup.training import TrainingTracker, HFTrackerCallback

tracker = TrainingTracker(
    server="http://127.0.0.1:8090",
    name="tt-lora-r16-seed1",
    lanes=["r9700"],
    config=full_config_dict,             # hyperparameters, LoRA rank/alpha/targets, precision
    dataset={"manifest": "data/v3/manifest.json"},
    log_every=20,                        # optimizer steps per scalar window
)
trainer = Trainer(..., callbacks=[HFTrackerCallback(tracker)])
# or, in a custom loop:
tracker.microbatch(batch)                # on-device token counting, no sync
tracker.step_completed(loss, lr=..., grad_norm=...)
tracker.event("eval_start")
tracker.checkpoint(step, path, kind="resume" | "adapter")
tracker.finish(status="completed")
```

Rules:

- **Never block or fail training.** A background thread batches records to the API; if the API is unreachable, records append to a local JSONL spool and replay later.
- **No per-microbatch sync.** Token counts accumulate as device tensors and are read once per logging window.
- **Global totals in DDP:** one `all_reduce` of the token/sample counters at each window boundary; rank 0 logs global values, every rank logs its own timing.
- **Phase tagging:** each window is labelled `warmup`, `train`, `eval`, `checkpoint`, or `paused`. Windows containing eval or checkpoint time are excluded from steady-state throughput.
- **Run-level peaks:** if per-window peak memory is reset, a run-level maximum is kept separately.
- **Phase timing** (forward/backward/optimizer/data) uses `torch.cuda.Event` pairs read at window boundaries, or short profiler windows. Never wrap every op in `synchronize()`.

### Token accounting

Per optimizer step, log all three:

| Count | Definition |
|---|---|
| `processed_tokens` | `input_ids.numel()`; the padded tensor footprint |
| `nonpad_tokens` | `attention_mask.sum()` |
| `supervised_tokens` | `(labels[:, 1:] != -100).sum()`; accounts for the causal shift |

Derived: padding ratio = `1 - nonpad/processed`, supervision ratio = `supervised/nonpad`. Padded tokens approximate cost; padding-aware kernels may skip some work, so always display padding ratio next to measured step time.

## 4. Metrics catalog

Source key: **T** tracker (training process) · **S** sampler (sysfs/procfs/amdsmi) · **D** derived server-side · **P** profiler window · **K** kernel journal.

### 01 Training speed

| Metric | Pri | Src | Notes |
|---|---|---|---|
| Training tokens/sec (non-padding, global) | P0 | T | Primary throughput metric. |
| Supervised tokens/sec | P0 | T | |
| Optimizer steps/sec, optimizer step time | P0 | T | Step includes all accumulation microbatches. |
| Padding ratio, packing efficiency | P0 | T | Largest hidden waste in SFT. |
| Tokens/sec per GPU | P0 | T | |
| Effective batch size (examples and tokens per update) | P0 | T | |
| Microbatches/sec, samples/sec | P1 | T | Samples/sec is secondary: a 300-token and a 6,000-token resume are both one sample. |
| Forward / backward / optimizer latency | P1 | T/P | GPU events or profiler. |
| Epoch duration, ETA | P1 | D | ETA from rolling steady-state windows only. |
| Warm-up / compilation time | P1 | T | MIOpen tuning and `torch.compile` can take minutes on ROCm; separate from steady state. |

### 02 GPU and VRAM

| Metric | Pri | Src | Notes |
|---|---|---|---|
| GPU busy % per card | P0 | S | Activity, not efficiency. |
| Device VRAM used / peak per card | P0 | S | |
| PyTorch allocated / reserved / peak | P0 | T | `reserved − allocated` = allocator fragmentation. |
| VRAM headroom | P0 | D | Total − device peak. |
| Junction and memory temperature | P0 | S | Alert thresholds and time-above-threshold per run. |
| Board power | P1 | S | Integrated into energy. |
| Allocator retries (`num_alloc_retries`) | P1 | T | Rises before an OOM. |
| GTT used | P1 | S | Spill indicator. |
| Core / memory clock, fan | P1 | S | |
| Host RAM, swap | P1 | S | |
| MFU (estimate) | P1 | D | `~6 × N × tokens/sec / peak`; use trainable N for LoRA; always labelled estimate. |
| Throttle status, memory-controller %, bandwidth | P2 | S | Best effort; often N/A on Radeon. |

### 03 Bottlenecks

| Metric | Pri | Src | Notes |
|---|---|---|---|
| DataLoader wait per step | P0 | T | Most common bottleneck. |
| GPU idle share of step | P0 | D | Step time − GPU-busy time. |
| Sequence-length histogram (per window and per dataset) | P0 | T | Long resumes dominate step time. |
| Checkpoint write time and whether it blocks | P1 | T | |
| Eval time share of wall clock | P1 | D | |
| Tokenization time (if not pre-tokenized) | P1 | T | Prefer pre-tokenized datasets. |
| DataLoader worker CPU %, disk read MB/s, page faults | P1 | S | |
| Host→device copy time, kernel launch count, `.item()` stalls | P2 | P | Profiler windows only. |

### 04 Multi-GPU

| Metric | Pri | Src | Notes |
|---|---|---|---|
| Per-rank step time and straggler wait | P0 | T | |
| All-reduce time per step | P0 | T/P | |
| Scaling efficiency | P0 | D | `T_combined / (T_R9700_alone + T_9070XT_alone)`, using matched standalone baselines. |
| PCIe link speed/width per card | P1 | S | Recorded at run start and on change. |
| Per-rank microbatch size and VRAM headroom | P1 | T/S | The 16 GB card limits first. |
| Gradient communication bytes, comm/compute ratio | P2 | P | `rocprofv3 --rccl-trace` windows. |

### 05 Learning quality

| Metric | Pri | Src | Notes |
|---|---|---|---|
| Train loss per supervised token (raw + smoothed) | P0 | T | |
| Validation loss / perplexity | P0 | T | Fixed held-out sets (see [Data quality](#7-data-quality-and-leakage)). |
| Learning rate, grad norm (pre-clip), clip rate | P0 | T | |
| Trainable parameter count | P0 | T | Confirms LoRA config. |
| Task metrics per checkpoint | P1 | T/D | Logged by the training project; see the TalentTensor list below. |
| Train/validation gap | P1 | D | |
| Loss by data source / task bucket | P1 | T | |
| Quality at matched token budgets | P1 | D | Replaces a single "gain per 1M tokens" ratio, which always favours short runs. Step-0 base-model eval is the baseline. |
| Seed variance | P1 | D | ≥3 seeds before treating a difference as real; display mean and spread. |
| Per-layer grad/weight norms, update/weight ratio | P2 | T | Every 100–500 steps. |

TalentTensor task metrics (computed in that project, logged through `tracker.eval(...)`):

| Task | Metrics |
|---|---|
| JD ↔ resume relevance | macro F1, per-class precision/recall, confusion matrix |
| Ranking | NDCG@K, MRR, pairwise accuracy |
| Fraud signals | precision, recall, PR-AUC, false-positive rate |
| Structured extraction | field-level P/R/F1, exact match, schema-valid rate |
| Output format | JSON validity, label compliance |
| Calibration | ECE, Brier (requires label-token log-probs) |
| Generalization | metrics by source, document length, and subgroup where lawful and appropriate |

### 06 Stability and cost

| Metric | Pri | Src | Notes |
|---|---|---|---|
| NaN/Inf count, loss spikes, skipped steps | P0 | T | |
| OOM events with config | P0 | T | |
| Last good checkpoint, resume count | P0 | D | |
| Exit code, traceback | P0 | T/D | |
| GPU reset / page fault / ring timeout | P0 | K | See [Fault detection](#6-fault-detection-and-recovery). |
| Device disappearance | P0 | S | Enumeration change between samples. |
| Energy per run (GPU), Wh per 1M tokens | P1 | D | Integrated board power. |
| Telemetry overhead | P1 | D | Sampler duration, dropped samples, tracker enqueue time; periodic A/B with telemetry minimized. |
| Loss scale | P1 | T | FP16 only; hidden for BF16 runs. |
| Whole-system energy | P2 | S | CPU package via RAPL (root-readable on recent kernels) or an external meter. |
| Electricity cost | P2 | D | kWh × configured rate. |

### Reproducibility snapshot (P0, once per run)

Git commit and dirty-diff hash, full config, seed, dataset manifest hash, tokenizer hash, base-model revision, precision, optimizer, attention implementation, gradient checkpointing, `torch`/`transformers`/`peft`/`trl` versions, training-process and dashboard-host OS, ROCm installation path and component version files, PyTorch HIP version, amdgpu and kernel versions, and relevant `HSA_*`, `HIP_*`, `MIOPEN_*`, `PYTORCH_*`, `RCCL_*`, `ROCR_*`, and `TORCH_*` environment. Pin the ROCm stack and run a short training smoke test before any upgrade; compatibility claims in release notes must be checked against the exact installed version.

## 5. Checkpoints

Two artifacts per save:

| Kind | Contents | Use |
|---|---|---|
| `adapter` | LoRA weights + adapter config + tokenizer files | Evaluation and deployment |
| `resume` | adapter + optimizer + scheduler + RNG + sampler/dataloader state + step/token counters | Exact resumption |

Integrity rules:

1. The framework writes the checkpoint directory.
2. The tracker then writes `manifest.json` **last**: file list, sizes, SHA-256 (adapter files synchronously; large optimizer state hashed in a background thread), step, tokens seen, config hash, base-model revision, tokenizer hash.
3. Written via temp file + `fsync` + rename.
4. **A checkpoint without a valid manifest is incomplete** and is never used for resume or eval.
5. Retention: keep the last K resume checkpoints, plus every adapter flagged `best` or `designated`.

Resume selects the newest checkpoint whose manifest validates, and logs a `resumed` event with the step, the checkpoint, and the reason.

## 6. Fault detection and recovery

| Signal | Detection |
|---|---|
| Training process exit | Exit code and state via lease-holder PID |
| Hang (including RCCL deadlock) | Heartbeat carries `last_completed_step`; run becomes `stalled` when the step has not advanced for `max(N × rolling step time, floor)` |
| GPU reset / page fault / ring timeout | Long-lived `journalctl -k -f -o json` follower matching `amdgpu` patterns (user must be in `systemd-journal` or `adm`); raw lines stored with the event |
| Device disappearance | sysfs/AMD SMI enumeration change |
| Collective timeout | `init_process_group(timeout=...)` with async error handling enabled, so failures exit instead of hanging |

A background heartbeat thread alone cannot detect a hang: it keeps beating while the main thread is stuck in a collective. Staleness is therefore judged on step progress, not on heartbeat arrival.

On fault: mark the run `failed` or `stalled`, record the last valid checkpoint, release or hold the lease according to policy, and alert. Automatic restart is opt-in and capped.

## 7. Data quality and leakage

Computed in the training project; summaries logged through `tracker.dataset(...)`.

- **Splits:** splitting by candidate alone or by JD alone still leaks the other entity. Use two validation sets, **unseen-JD** and **unseen-candidate**, plus a time-based split where dates exist. Use connected-component splitting only if the components stay reasonably sized. Keep a human-verified test set that is never used for selection.
- **Near-duplicates:** MinHash or embedding similarity across splits; report the leakage rate.
- **Truncation:** percentage of examples truncated and tokens lost, by document length bucket.
- **Provenance:** synthetic vs human-labelled ratio, teacher-model disagreement, per-source validation metrics.
- **Personal data:** resumes contain PII. Per-example records, probe outputs, and archived chat history need redaction and retention policies consistent with [TELEMETRY_AND_STORAGE.md](TELEMETRY_AND_STORAGE.md#retention-and-export).

## 8. Checkpoint evaluation pipeline

The first dual-GPU mode to build: R9700 trains, 9070 XT evaluates.

```text
adapter checkpoint (manifest valid)
  ├─ native eval (in the training loop on R9700): val loss + parity set
  └─ eval queue ──► convert LoRA → GGUF (F16 adapter)
                    ──► restart llama-server on 9070xt lane with base GGUF + --lora
                    ──► parity set: GGUF F16 base, then GGUF quantized base
                    ──► full task eval on candidates (designated / best-so-far)
                    ──► record + compare ──► promote or keep previous best
```

### Attribution

Run the same fixed parity set (about 200 examples, greedy decoding) at three levels:

| Level | Measures |
|---|---|
| Native PyTorch (BF16) | Training quality |
| GGUF F16 | Conversion and engine differences |
| GGUF quantized | Quantization loss |

The differences between adjacent levels separate model quality from deployment degradation. Greedy outputs can diverge on numeric noise, so compare task metrics and token-agreement rate, not only exact strings.

### Adapter lifecycle

`llama-server` loads adapter files only at startup (`--lora`, `--lora-scaled`). `/lora-adapters` and the per-request `lora` field only change the scales of adapters that are already loaded. Each new checkpoint is therefore a lane restart, which is cheap because the base GGUF stays in page cache.

Before the first run for each base model and llama.cpp build, run a converter smoke test: convert a trivial adapter, load it, and confirm the outputs differ from the base model while staying coherent. Fail fast on a layer-count or tensor-name mismatch.

Store with every evaluation: base-model revision, base GGUF hash, tokenizer hash, adapter checksum, rank/alpha/target modules, step, tokens seen, converter and llama.cpp build versions, quantization type, eval context length, conversion time, load time, eval latency, and native vs deployment metrics.

### Queue

- Bounded and coalescing: always keep `designated` checkpoints (every Kth) and the latest; drop intermediate ones when the queue is full.
- Backpressure is visible on the dashboard (queue depth, eval lag in steps).
- Inference runs produced by evaluation are tagged with `training_run_id` and `checkpoint_step` in the existing run store, so the inference telemetry doubles as the checkpoint scorecard.

## Memory forecasting

- **Before launch:** estimate weights, gradients (trainable only), optimizer state, activations (sequence length × microbatch × layers × hidden, adjusted for checkpointing), and workspace. Show the estimate against each target card's capacity.
- **After launch:** record measured peaks per phase (forward, backward, optimizer) to the `hardware_profiles` table, keyed by model, sequence length, microbatch, precision, LoRA config, and GPU.
- Over time the measured table replaces the estimate. This is the training counterpart of the existing inference capacity check.

## 9. Storage

A separate `training.sqlite3` (WAL, foreign keys), alongside the existing metrics and chat-history databases:

```text
training_runs          id, name, status, parent_run_id, lease_id, started_at, finished_at,
                       config_json, env_json, dataset_json, git_commit, seed, summary_json
training_steps         run_id, step, ts, phase, loss, lr, grad_norm, clip_frac,
                       processed_tokens, nonpad_tokens, supervised_tokens, step_time_ms,
                       data_wait_ms, fwd_ms, bwd_ms, opt_ms, mem_alloc_bytes,
                       mem_reserved_bytes, mem_peak_bytes, nan_count, extra_json
training_rank_steps    run_id, step, rank, gpu_pci_bus, step_time_ms, allreduce_ms,
                       tokens, mem_peak_bytes
training_events        id, run_id, ts, kind, step, severity, payload_json
training_checkpoints   id, run_id, step, kind, path, manifest_json, valid, size_bytes,
                       save_ms, best, designated
checkpoint_evals       id, checkpoint_id, level (native|gguf_f16|gguf_quant), eval_set,
                       metrics_json, latency_json, versions_json, created_at
training_example_stats run_id, checkpoint_id, example_id, loss, correct, extra_json
hardware_profiles      key_json, gpu_pci_bus, peak_mem_by_phase_json, tokens_per_sec,
                       measured_at, run_id
notes                  id, run_id, ts, author, hypothesis, change, outcome
```

Device-level GPU samples (`hardware_samples`: `pci_bus`, `ts`, the fields from section 1) are shared by inference and training and live in the main metrics database. Retention follows the existing policy: raw samples are kept for 7–30 days, then downsampled. Step rows are small (about 20 MB per 100k steps) and kept for the life of the run.

## 10. API

```text
POST /api/training/runs                     create run (+ lease), returns run_id
POST /api/training/runs/{id}/metrics        batched step/rank records (also renews lease + heartbeat)
POST /api/training/runs/{id}/events
POST /api/training/runs/{id}/checkpoints
POST /api/training/runs/{id}/evals
POST /api/training/runs/{id}/finish
GET  /api/training/runs                     list with status, progress, ETA, best val metric
GET  /api/training/runs/{id}                detail + summary
GET  /api/training/runs/{id}/series?m=loss,lr,tokens_per_sec&from=&to=&downsample=
GET  /api/training/compare?ids=a,b,c        series + config diff
GET  /api/hardware                          live device-level view of both GPUs
```

## 11. Dashboard

Add two navigation entries rather than restructuring existing views:

- **Training:** run list (status, progress, ETA, tokens seen, best validation metric), run detail with one panel per catalog section (01–06), an events timeline overlaid on every chart, a checkpoint table with native/GGUF/quant metrics side by side, eval-queue depth, and run comparison with the config diff shown next to the results.
- **Hardware:** live per-card busy %, VRAM, GTT, temperatures, power, clocks, PCIe link, and the current lease holder, shared by inference and training.

Profiler traces are attached to runs as artifacts, not given a separate tab.

Alerts: NaN/Inf, loss spike, throughput drop versus rolling baseline, GPU idle above threshold, temperature above threshold, stalled run, GPU reset, eval queue saturated.

## 12. Collection frequencies

| Data | Frequency |
|---|---|
| Device hardware metrics | 1 s |
| Throughput, step time, loss, LR, memory | every 10–50 optimizer steps (one window) |
| Gradient diagnostics | every 10–50 steps; per-layer every 100–500 |
| Validation and parity eval | per designated checkpoint |
| Profiler traces | short selected windows only |
| Faults, OOM, throttling, checkpoints, resumes | as they occur |

## 13. Benchmark plan

Hold dataset, model revision, tokenizer, effective batch, sequence-length distribution, optimizer, and precision constant within each comparison. Exclude warm-up from steady-state numbers, but record it.

| ID | Configuration | Purpose |
|---|---|---|
| A | R9700 standalone | Baseline |
| I | R9700, LoRA rank/targets sweep | Main path; trainable params, memory, speed |
| C | R9700, microbatch sweep | Best batch configuration |
| D | R9700, sequence-length sweep | Context cost |
| H | R9700, gradient checkpointing on/off | Memory saved vs time added |
| E | R9700, precision/attention-kernel variants | Stability, memory, speed |
| B | 9070 XT standalone, same config as A | Sanity check: should be close to A when it fits; a large gap points to thermals, PCIe, or drivers |
| K | R9700 trains + 9070 XT runs live checkpoint eval | Contention and eval lag |
| J | Multi-hour R9700 run | Thermals, memory growth, stability |
| S | Repeat the chosen config with 3 seeds | Noise floor |
| G | DDP across both cards | Synchronization cost and scaling (last) |

## 14. Acceptance tests

| Test | Expected | Automation |
|---|---|---|
| Deploy inference onto a leased lane | 409 or queued; training is unaffected | unit (fake manager) |
| Lease holder dies | Lease expires after TTL; lane becomes free | unit |
| Kill trainer during checkpoint save | Partial checkpoint has no valid manifest; resume uses the previous one | integration (fake writer) |
| Resume after a kill | Step, LR schedule, RNG, and data position continue; loss curve is continuous | hardware |
| Stop one DDP worker | Job exits or becomes `failed`/`stalled` within the timeout; no indefinite hang | hardware |
| Telemetry source disappears | Fields become `null` with a reason; training continues | unit |
| Tracker API unreachable | Records spool to JSONL and replay; training does not stall | unit |
| Save an adapter | Native, GGUF F16, and GGUF quant evals all link to the same checkpoint ID | integration |
| Flood the eval queue | Queue stays bounded; designated checkpoints are kept | unit |
| Train and serve concurrently | Contention is visible in both runs' metrics | hardware |
| Same config, 3 seeds | Dashboard shows spread; differences within noise are flagged | unit (aggregation) |
| Thermal threshold reached | Event and time-above-threshold recorded; configured action executes | unit (fake sensor) |
| DDP numerical equivalence | One update on 2 GPUs matches the single-GPU accumulated reference within tolerance; global token counts and effective batch agree | hardware |

## Implementation status

Everything in this document is implemented in `dual_gpu_setup/training/` (plus `hostos/` and `scheduling/`), served by `app.py`, and shown under the **Training** and **Hardware** dashboard tabs, with the exceptions listed under *Not built*. The suite covers the logic with fakes and real subprocess/HTTP runs; items marked *needs the Linux GPU machine* cannot be proven on a development box.

| Area | Where | Notes |
|---|---|---|
| Windows/Linux host layer | `hostos/` | `/proc`, DRM fdinfo and POSIX signals on Linux; ctypes/PDH/taskkill on Windows. Native `llama-server` via `discovery.backend_server_binaries` (no LM Studio needed). |
| Environment/OS attribution | `training/environment.py`, `host_info.py`, `training/store.py` | A top-level `host_os` is persisted even if tracker capture is disabled. Snapshots distinguish training-process and dashboard-host OS and include Linux distribution, Python, kernel, packages, ROCm component versions and relevant accelerator environment. |
| Stable GPU identity | `lane.pci_bus`, `training/hardware.py` | PCI address -> name -> unique VRAM size. PyTorch device indices mapped to PCI (`env.gpu_map`), negotiated PCIe link and driver recorded per run (`env.server_hardware`). |
| Hardware telemetry | `training/hardware.py`, `amd_smi.py` | 1 Hz sysfs sampler, nullable sensors, overhead and dropped-tick stats, optional AMD SMI that only fills gaps, persisted throttle/source/unavailable metadata, amdgpu kernel-fault follower. |
| Leases | `scheduling/leases.py` | Atomic multi-lane, token-protected, expiring, PID start-token verified before any termination. Deploys onto leased lanes return 409; training refuses lanes inference holds. |
| Tracker + HF callback | `training/tracker.py`, `hf_callback.py` | No per-microbatch sync, JSONL spool, global token all-reduce once per window, CUDA-event phase timing (forward / backward+other / optimizer), data-wait, all-reduce time. |
| Run lifecycle / recovery | `training/store.py`, `checkpoints.py` | Stalled judged on step progress, rank-aware stall detection, manifest-last checkpoints, resume continuity report. |
| Evaluation pipeline | `training/evaluation.py`, `selection.py` | Bounded coalescing queue, adapter compatibility checks, native / GGUF F16 / GGUF quant with explicit failures, changeover timing, deployed-score checkpoint selection, regression detection. *Needs the Linux GPU machine* for the real converter and `llama-server`. |
| Experiments | `training/analysis.py`, `benchmarks.py` | Equal-token-budget comparison with seed-noise gating, batch/context/LoRA/checkpointing/precision sweeps that only recommend, thermal, contention, energy, MFU (estimate). |
| Memory forecasting | `training/memory_forecast.py` | Measured profiles replace the analytic estimate per configuration key. |
| DDP | `training/ddp.py` | Per-rank rows, straggler/idle/comm analysis, scaling efficiency, hang-safe launch settings, CPU/GPU numerical-equivalence check (*run it on the machine*; skipped where torch is absent). |
| Privacy | `training/privacy.py` | Content-bearing keys dropped, strings truncated, e-mail/phone-like text redacted before storage. |

### Using it

```python
from dual_gpu_setup.training import TrainingTracker, HFTrackerCallback, find_latest_valid

tracker = TrainingTracker("http://127.0.0.1:8090", name="tt-lora-r16-seed1", lanes=["r9700"],
                          config=cfg, log_every=20, require_lease=True)   # fail fast if the GPU is taken
trainer = Trainer(..., callbacks=[HFTrackerCallback(tracker)])             # set logging_steps=20 too
trainer.train(resume_from_checkpoint=find_latest_valid(output_dir))        # newest manifest-valid checkpoint
```

For resume, call `tracker.resume_from(checkpoint_dir)` first: it validates the manifest, links the parent run and logs whether optimizer/scheduler/RNG state was present. For a custom loop call `tracker.microbatch(...)` or `tracker.microbatch_counts(...)`, `tracker.step_begin()`, `tracker.step_completed(step)`, `tracker.log_scalars(...)`, `tracker.checkpoint(step, path)` and `tracker.finish()`.

Multi-GPU (torchrun): construct the tracker on every rank. Rank 0 owns the run; every rank reports its own timing, memory and all-reduce rows. Merge `ddp.launch_env()` into the launch environment and pass `ddp.process_group_timeout()` to `init_process_group` so a lost worker fails the job instead of hanging it. Time gradient all-reduces with `ddp.install_comm_timing(model, tracker)`.

Evaluation (config):

```toml
[evaluation]
enabled = true
lane = "9070xt"
base_model = "Qwen/Qwen2.5-3B-Instruct"
base_f16_gguf = "G:/models/qwen-3b-f16.gguf"
base_quant_gguf = "G:/models/qwen-3b-Q4_K_M.gguf"
converter_script = "G:/llama.cpp/convert_lora_to_gguf.py"
native_command = ["python", "eval_native.py", "--adapter", "{adapter_dir}", "--set", "{parity_set}", "--out", "{output_json}"]
gguf_command   = ["python", "eval_gguf.py", "--base-url", "{base_url}", "--set", "{parity_set}", "--out", "{output_json}"]
parity_set = "data/parity_200.jsonl"
designate_every = 3
primary_metric = "task_score"
```

Each command writes `{"metrics": {...}, "latency": {...}}` to `{output_json}`. The metrics must be aggregates: raw resumes and evaluation examples never belong in telemetry (see `privacy.py`).

### Not built

- Preference pairs from compare mode and coding-agent trajectories (the "long-term data" list): they belong in the chat layer, not training telemetry. Per-example loss/correctness (`/examples`), the lab notebook, hardware profiles and the failure corpus (events) are in.
- FSDP/ZeRO and pipeline parallelism (deferred until DDP measurements justify them).
- Automatic restart/tuning: recommendations only, by design.

### Open items still to verify on the training machine

The original verification list below still applies: sysfs/AMD SMI coverage on gfx1201, PCI fields in `torch.cuda.get_device_properties`, the LoRA -> GGUF converter for the chosen base model, RDNA4 stability claims against the installed ROCm, and which integration (`Trainer` callback or manual loop) TalentTensor uses.

## 15. Phases

| Phase | Deliverables |
|---|---|
| 0 | Everything in [Phase 0](#phase-0-minimum-for-the-first-r9700-lora-run) |
| 1: reliability | Kernel-fault follower, resume verification, phase classification, memory forecaster, alerts |
| 2: quality and deployment | Checkpoint eval pipeline on the 9070 XT, parity levels, eval queue, best-checkpoint promotion, dataset/leakage logging |
| 3: performance | Bottleneck breakdown, profiler windows, benchmark matrix A–K, contention analysis |
| 4: distributed | DDP equivalence test, per-rank metrics, RCCL profiling, scaling efficiency; FSDP only if measurements justify it |

## Long-term data worth keeping

These are cheap to store now and valuable later:

- **Preference pairs** from compare mode (winner and rating), which become DPO data.
- **Rated chat turns and coding-agent trajectories** (success/failure), which become SFT and agent data.
- **Per-example loss and correctness** across checkpoints: data maps for finding mislabeled or hard examples.
- **Hardware profiles:** a measured memory/throughput table used to predict fit and duration.
- **Failure corpus:** every OOM, NaN, reset, and hang with its full config.
- **Lab notebook:** hypothesis → change → outcome, linked to run IDs.

## Open items to verify on the training machine

- Which AMD SMI and sysfs fields are populated on gfx1201 with the installed ROCm and kernel.
- Whether `torch.cuda.get_device_properties()` exposes PCI fields in the installed PyTorch build.
- The current state of LoRA → GGUF conversion for the chosen base model (run the converter smoke test).
- Release-note claims about RDNA4 training stability, checked against the exact installed ROCm version.
- Whether the training project uses HF `Trainer`/TRL or a custom loop; this decides whether the callback or the manual API is the primary integration.
