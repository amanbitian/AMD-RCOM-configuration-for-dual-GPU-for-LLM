# Dual GPU Setup: Improvement Plan (17 November 2026)

Project: `/media/pratap/Project and Data/Projects/Dual_gpu_setup`
Live service on Ubuntu: `~/dual-gpu-data` (config, databases, `service.log`), port 8090
Status: analysis and plan only. No code or live service was changed.

---

## 1. Principles this plan must keep

| # | Principle |
|---|-----------|
| 1 | Run one model on one GPU without activating the other GPU |
| 2 | Run both GPUs in parallel for LLM inference |
| 3 | Use both GPUs together to run one larger model |
| 4 | Save all telemetry and LLM process data |
| 5 | See the data on either operating system, and lose no old data in the update |
| 6 | Optimisations must increase GPU performance |
| 7 | Full hardware coverage while a model runs (temperature, fan speed, RAM, CPU, GPU, wattage), saved as a log |
| 8 | Fix any other optimisation gap found in the code |

**Hardware log rule.** Record only while a GPU is in use by the app: a model is loaded, a request is running, or a training job holds the GPU. Write nothing when idle. This is read as "either GPU in use". Logging only when both are active together is a one-line setting if preferred.

**Corrections to earlier proposals.** Pruning old runs and thinning samples were withdrawn because they break principle 4. Old data is archived to compressed files and never deleted. Keeping content out of the archive stays an explicit opt-in. Routing, preloading and automatic restart must never place a model on a GPU that was not chosen.

---

## 2. What the last three days show

A two-GPU training run (`stage1_pretrain_350m`) started on 14 November at 08:37 and was still running on 17 November at 05:42. No llama-server inference has run on Ubuntu yet, so this says nothing about the inference path on Linux.

### 2.1 Healthy

| Check | Result |
|-------|--------|
| Training speed | Steady at about 23,700 tokens per second across all six 12-hour blocks |
| Loss | Fell from 2.66 to 2.10 |
| NaN events and allocation retries | 0 and 0 |
| Kernel log (about 10,000 lines) | No GPU resets, machine checks, out-of-memory kills, disk or PCIe errors |
| Swap and RAM | No swap used, RAM peaked at 33% |
| VRAM spill | None. R9700 peaked at 29.8 GB of 32 GB, which is tight |
| Sampler | 251,277 ticks, 0 dropped |
| Service requests | 347,331 answered with success, 4 not-found probes |
| GPU lease | Renewed correctly throughout |
| Checkpoints on disk | Every 30 minutes, latest at step 11,332 |

### 2.2 Issues found

| # | Issue | Evidence | Planned solution |
|---|-------|----------|------------------|
| 1 | R9700 has no cooling headroom | Fan at its maximum (about 5,000 rpm) for the whole run. Junction averaged 91 °C, peaked at 97 °C, and spent 1,288 seconds at 95 °C or more. Memory touched 90 °C for 110 seconds. | Check airflow and card spacing. Between runs, test a small power cap and measure tokens per watt. Log intake temperature. |
| 2 | It ran hotter before the fans were raised | 14 November 07:10 to 07:47: 103 °C junction and 100 °C memory at about 3,800 rpm | Pre-run check that fans are in the right mode before a long job |
| 3 | No alert was recorded | Alert list empty, event table holds only start and finish entries. Alerts are computed on request and never saved. Thresholds are 100 °C junction and 90 °C memory. | Save every alert with start, end and peak value |
| 4 | Throttle status never captured | Blank in all 754,608 hardware rows. The `amdsmi` Python module is missing from the `slm-rocm` environment, though `amd-smi` is installed. | Install the module or fall back to the tool. Show a missing source in health. |
| 5 | PCIe link reading is wrong | App reports 16 lanes for both cards. Real slot links at the root ports are 8 lanes each at 32 GT/s, normal for this board. | Read the link at the slot so a real downgrade is detected |
| 6 | Host sensors not recorded | At the time of the check: CPU 80 °C, SSD 64 °C, RAM modules 52 °C, coolant 42 °C, pump and fan speeds available. None is stored. | Add them to the hardware log |
| 7 | Power margin unknown | GPUs averaged 554 W together and peaked at 770 W. About 38 kWh so far. Single readings spiked to 467 W and 496 W. CPU power is not recorded. | Record CPU power. Compare the total with the power supply rating. |
| 8 | Checkpoints invisible to the service | Checkpoints exist on disk, yet the service lists none for this run | Trace why the report does not arrive and add a test |
| 9 | Service unmanaged and busy | Started with nohup, not the systemd unit. `service.log` reached 30 MB (347,248 request lines). The service used 9 h 26 min of CPU, mostly answering page polls. | Install the unit, rotate the log, stop logging routine polls, cache poll answers |
| 10 | Deletion deadline | Hardware samples older than 30 days are deleted at start. Any restart after 14 December removes this run's first readings. | Remove the delete before then |
| 11 | Integrated GPU fills a third of the log | 251,536 of 754,608 hardware rows belong to the integrated GPU | Log only the GPUs the app is using |
| 12 | Training database growth | 186 MB in three days | Compressed daily archive files alongside the database |

Readings for the 9070 XT were comfortable: about 82 °C junction and memory, fan near 4,070 rpm.

---

## 3. Check against each principle

### Principle 1: one model on one GPU

Works today. Single-GPU mode starts one llama-server with a device flag for that lane, and each GPU loads and unloads on its own.

| Gap | Fix |
|-----|-----|
| Isolation relies on the device flag alone. No visibility variable is set, so the runtime in that process can still see the second card. Not verified whether this wakes it. | Hide the other card from single-lane servers. Add a hardware test that its VRAM and load stay at idle. |
| On Linux, each free-VRAM check starts a llama-server process that enumerates both cards | Read free VRAM from the kernel files the sampler already uses |
| On-demand loading may pick either GPU | Per-lane "do not use" switch respected by every automatic feature |
| Serving identifies a card by name, telemetry by PCI address | One identity scheme, PCI first |
| Config requires exactly two lanes | Allow start with one card present |

### Principle 2: both GPUs in parallel

Works today. Parallel mode starts both servers together and sends prompts to both.

| Gap | Fix |
|-----|-----|
| On-demand loads take one lock for both cards | Plan under the shared lock, load under a per-GPU lock |
| Both servers share all CPU cores | Test core pinning, keep only if the benchmark improves |
| Windows sampler starts PowerShell every two seconds | Direct counter calls |

### Principle 3: both GPUs for one larger model

Supported in code, not yet trustworthy. Splitting buys capacity, not speed.

| Gap | Fix |
|-----|-----|
| Split output is corrupt on the b11699 ROCm build, so Windows routes split models to LM Studio's runtime | Keep that routing until a build passes |
| The Ubuntu config uses the configured ROCm build for split models. Unverified, and nothing checks the output. | Correctness canary at load: fixed prompt compared with a single-GPU reference. Refuse the split if it fails. |
| Capacity check ignores CPU-offloaded experts and refuses a 59 GB model | Subtract offloaded layers |
| Cache precision fixed at 16-bit in the capacity check | Per-model precision option counted correctly |

### Principle 4: save all telemetry and process data

Partly true today. Each run stores request, output, reasoning, token counts, timings and per-run samples of CPU, RAM and VRAM. The conversation archive falls back to a file if its database is unavailable.

| Gap | Fix |
|-----|-----|
| GPU temperature, fan, power and clocks are saved only during training, not inference | Hardware log whenever a GPU is in use |
| Hardware samples older than 30 days are deleted at every start | Remove the delete, archive instead |
| Kernel fault events are recorded only into active training runs | Record whenever a GPU is in use |
| Runs stay "running" forever if the service dies mid-request | Mark them interrupted at startup |
| Archive write failures are only printed | Count them and show them in health |
| Alerts are not saved | Save alert history |

### Principle 5: data on both systems, nothing lost

Not true yet. The Windows history holds 18,080 runs (14,289 tagged Windows, 3,791 untagged). Linux has its own separate, nearly empty run database.

| Rule for the update | Reason |
|---------------------|--------|
| Full copy and integrity check before any migration | The 2.1 GB history has only one older backup |
| Migrations only add columns and tables | Nothing existing is rewritten or dropped |
| Row counts compared before and after | Proof that nothing was lost |
| Untagged runs marked Windows explicitly | Today that is only assumed at read time |
| Each system keeps its database on its own native disk | A live database on the shared NTFS partition is the main loss risk |
| New records exchanged as append-only files in a shared folder | Plain files are safe on NTFS |
| First Linux start imports the full Windows history | Both systems show the same union |
| Models matched by fingerprint, not path | Paths differ per system |
| Schema version checked at start | Two systems may run different code |
| Windows hardware clock set to UTC | Avoids a 5 h 30 min shift after dual boot |

### Principle 6: optimisation must raise GPU performance

This cannot be guaranteed from reading code. Only measurement can.

| Group | Items |
|-------|-------|
| Expected to raise GPU throughput | Multiple slots for concurrent requests, tuned batch sizes, draft models, power profile and link checks, thermal headroom |
| Saves waiting time, not raw speed | Prompt cache with session affinity, preloading, per-GPU load lock, task routing |
| No effect on GPU speed | Data, security, logging, code structure |
| Can reduce speed if misapplied | More slots lower per-request speed, 8-bit cache, core pinning, heavy sampling on Windows |

Acceptance rule: every change sits behind a setting that is off by default, and is kept only if the benchmark on both systems is at or above baseline.

### Principle 7: hardware coverage and log

| Reading | Linux now | Windows now | Saved during inference today |
|---------|-----------|-------------|------------------------------|
| GPU usage and VRAM | Yes | Yes | Per process only |
| GPU temperatures (edge, junction, memory) | Yes | Yes, sensor ids inferred | No |
| GPU fan speed | Yes | Yes | No |
| GPU power in watts | Yes | Yes | No |
| GPU clocks | Yes | Yes | No |
| PCIe link | Wrong value | Not checked | No |
| GPU throttle status | Source missing | No | No |
| CPU usage | Total only | Total only | Yes |
| CPU temperature | No | No | No |
| CPU power | No | No | No |
| CPU fan, pump, coolant, case fans | No | No | No |
| RAM and swap | Yes | Yes | Yes |
| RAM module temperature | No | No | No |
| SSD temperature, wear, free space | No | No | No |
| Whole-system wattage | No | No | No |

Whole-system wattage cannot be read from sensors. GPU plus CPU power gives an estimate. True wall power needs a metering plug or a power supply with telemetry.

### Principle 8: other gaps

Covered in sections 2.2, 5 and 6.

---

## 4. Hardware log design

| Aspect | Design |
|--------|--------|
| When it records | Only while a GPU is in use by the app. Starts when a model begins loading or a training job takes the GPU. Stops at unload or release. |
| Rate | One sample per second |
| GPU fields | Usage, VRAM, spill, edge, junction and memory temperature, fan speed, power, clocks, slot link, throttle status |
| Host fields | CPU usage, CPU temperature, CPU power, RAM used, swap, memory pressure, pump and fan speeds, coolant temperature, SSD temperature, RAM module temperature, free disk space |
| Context on each sample | Operating system, loaded model, lane, active run or training id |
| Events | Alerts with start, end and peak. Kernel faults. Server crashes. |
| Per-run summary | Minimum, average and peak temperatures, average and peak watts, energy used, peak fan speed, throttle time |
| Storage | Database for recent data, daily compressed files for the archive. Nothing deleted. |
| Export | CSV or JSON lines for any run or time range |
| Both systems | Same schema. A missing sensor is blank, never zero. |
| Not logged | The integrated GPU, and idle time |

The separate hardware monitor app on this workstation already reads many host sensors. One collector feeding both apps would avoid double polling. Its interfaces still need checking.

---

## 5. Performance and scheduling items

| Area | Gap | Enhancement |
|------|-----|-------------|
| Concurrency | One serving slot per GPU | Slot count per workload, with each agent session tied to its own slot |
| Prompt cache | Switched off in RAM | A bounded cache of a few GB |
| Context size | Three fixed tiers | Largest context that fits, from the existing estimate |
| Tuning | Batch sizes and threads set by hand | Auto-tune run that saves the best profile per model, GPU, build and system |
| Draft models | Manual | Suggest pairs, switch off when acceptance stays low |
| Loading | One global load lock | Per-GPU load lock |
| Recovery | Crashed server stays down | Automatic restart with back-off and crash-loop detection |
| Routing | Caller picks the lane | Autocomplete to the small card, agent work to the large card, interactive ahead of batch |
| Cancel | Only by unloading the model | Per-request cancel |
| Model storage on Linux | Would read from the NTFS partition | Native disk for model files |
| Host settings | Unchecked | Read-only preflight of governor, GPU power profile and slot link |

---

## 6. Code, security and operations items

| Area | Gap | Fix |
|------|-----|-----|
| Repo | 29 uncommitted files. Corrupted ignore entry keeps `full_run.dual_gpu.toml` tracked. | Commit in logical pieces, fix the entry |
| Security | No authentication. Deploy and unload accept any content type. | Require JSON on every POST. Optional token if the service is ever exposed beyond this machine. |
| Process cleanup | Unmanaged llama-server on a lane port is stopped automatically | Only stop processes the app started |
| Dashboard data | Loads the full run history into Python on each refresh | Aggregate in SQL, fetch only needed columns, cache briefly |
| Database locking | One lock for every read and write | Lock-free reads, single writer |
| Logging | Print statements, no rotation | Logging module with levels and rotation |
| Server limits | No thread cap or request timeout | Cap concurrency, add timeouts |
| Windows sampling | PowerShell started every two seconds | Direct counter calls |
| Windows autostart | None | Task Scheduler entry |
| Linux service | Run with nohup | Install the existing systemd unit |
| Structure | 5,120-line server file | Split later. It serves no principle directly but lowers change risk. |

---

## 6A. Training metrics to show on the dashboard

Added from a separate metrics review. The "current status" column is taken from that review and has not been checked against the training package or dashboard code.

| Metric | Current status | Recommendation |
|--------|----------------|----------------|
| Raw training loss | Available | Keep |
| Rolling mean training loss | Not apparent | **Add** |
| Validation loss | Logged by trainer | Display prominently |
| Validation perplexity | Logged by trainer | Display prominently |
| Resume validation loss | Not shown | Add |
| JD validation loss | Not shown | Add |
| Knowledge validation loss | Not shown | Add |
| General-text validation loss | Not shown | Add |
| Z-loss term | Available in trainer | Display |
| Total optimization objective | Available in trainer | Display |
| Gradient norm | Available | Keep |
| Learning rate | Available | Keep |

Notes:
- Everything shown must also be stored (principle 4), so each new series needs a column or table in the run store, added by an add-only migration (principle 5).
- Open question: the rolling window size for the mean loss (steps or tokens).
- Open question: whether the resume, JD, knowledge and general-text validation sets are already evaluated by the trainer or only the combined validation loss is.

---

## 7. Plan and order of work

| Step | Work | Principles |
|------|------|------------|
| 1. Protect data | Remove the 30-day delete. Back up and verify. Schema versions with add-only migrations. Mark interrupted runs. | 4, 5 |
| 2. Hardware log | Section 4 in full | 7, 4 |
| 3. Fix the three-day issues | Alert history, throttle source, slot link, checkpoint reporting, service management and log | 8 |
| 4. Both operating systems | Import Windows history on Linux, exchange files, model fingerprint, one config with a section per system | 5 |
| 5. GPU modes | Isolation and its test, free VRAM from the kernel, per-GPU load lock, split canary, capacity fix | 1, 2, 3 |
| 6. Performance | Benchmark baseline on both systems, then gated changes from section 5 | 6 |
| 7. Reliability | Automatic restart, logging, JSON-only POSTs, commit pending work | 8 |
| 8. Training metrics | Section 6A: rolling mean loss, per-set validation losses, prominent validation loss and perplexity, z-loss and total objective | 4, 8 |

Step 1 has a deadline of 14 December because of the 30-day delete.

---

## 8. What is needed and what is not

| Verdict | Items |
|---------|-------|
| Required by the principles | Hardware log, removing the 30-day delete, safe migration with schema versions, databases off NTFS, cross-system exchange, split canary, isolation test, capacity fix, benchmark baseline |
| Worth doing | Slots and session affinity, tuned profiles, adaptive context, automatic restart, per-GPU load lock, JSON-only POSTs, logging, committing pending work, training metrics display (section 6A) |
| Not needed for these goals | More than two lanes, a container image, a metrics endpoint for external tools, a live event stream, dashboard restructuring, archive encryption, a client library |
| Only if shared or exposed | Authentication, a licence, a wider CI matrix |

---

## 9. Constraints, open questions and inputs needed

**Constraints**
- Applying changes means restarting the service. That waits until the training run finishes or a pause is chosen.
- A power cap or fan change writes to the GPU. Only between runs, and only with a go-ahead.

**Inputs needed**
- Power supply wattage, to judge the margin against a 770 W GPU peak plus the CPU.
- Whether "in use" means either GPU or both together.
- The Windows hardware clock setting.

**Still unverified**
- Whether the device flag alone leaves the other GPU untouched.
- Split-model correctness on Linux.
- Every performance gain.
- The Windows fan sensor mapping.
- Behaviour under heavy concurrent inference load.
- The training package and dashboard script were only skimmed, so more gaps may exist there.

**Checked and sound**
- The dashboard escapes model text before display.
- No shell or unsafe-load calls.
- Leases expire by timeout and guard against recycled process ids.
- Shutdown flushes the background writers.
- Tests: 462 passed, 4 skipped, in a throwaway environment.

---

## 10. Implementation status (20 November 2026)

Training finished on 20 Nov, so the work was done then. Verified here: 527 tests pass; the new code ran against a copy of
the live Ubuntu database (373 MB, 1.5 M GPU rows): migration lost no rows, six daily archives verified, validation loss
and the three unreported checkpoints were recovered, and the dashboard panels rendered. **The live service was not
restarted** (it is still the old code, started by hand); nothing below is live until it is.

| Step | Done | Not done / not verifiable here |
|---|---|---|
| 1. Protect data | 30-day delete removed (`prune_hardware` is a no-op); verified backup before first start of a version; newer-schema refusal; interrupted runs marked; `origin_os` tag; add-only schema | Hardware-sample deletion is gone but the database now grows ~60 MB/day; compaction of archived rows was deliberately not built (it would delete data) |
| 2. Hardware log | Whole section 4 except: Windows host sensors read LibreHardwareMonitor/OpenHardwareMonitor through WMI | Windows sensors and the Linux CPU power/coolant/pump fields are unproven on the workstation (no sensor of that kind may be exposed; each is reported unavailable with a reason, never 0). A model's *loading* phase is not logged until it is registered |
| 3. Three-day issues | Alert history; throttle via `amd-smi` fallback; slot link from the root port; checkpoints (cause: the trainer never calls `tracker.checkpoint`; the service now registers them from disk); service installers; log rotation without deletion; polls not logged and answers cached; process cleanup only for servers the service started | On this GPU `amd-smi` reports throttle as N/A, so throttle stays unavailable (and says why). The trainer should also call `tracker.checkpoint(step, path)` after saving |
| 4. Both OSes | `--exchange-dir`, `--import-runs-from`, `--import-from`, OS tags, model fingerprint, schema-version guard, NTFS warning | Never run against the real Windows database or a real shared folder |
| 5. GPU modes | Free VRAM from kernel files; lane `auto_use`; one-lane start; capacity check counts CPU-offloaded experts and KV precision; split canary; opt-in isolation + hardware test | **Per-GPU load lock not built**: releasing the shared lock early would let the orphan scanner treat a starting server as an orphan. **One identity scheme (PCI first)** not built: serving still resolves a card by name. Isolation and split correctness need `pytest -m rocm_hardware` on the workstation |
| 6. Performance | Baseline recorder and regression check (`scripts/benchmark_inference.py`) | No tuning change was made, because none can be judged without running the baseline on both systems first (acceptance rule). Slots/session affinity already existed |
| 7. Reliability | Opt-in auto-restart with back-off and crash-loop stop; connection cap and idle timeout; JSON content type now required on deploy, unload, model refresh and the lane switch (before: only the LM Studio and service-control endpoints); corrupted ignore entry fixed | Lock-free reads / single writer for the run database, SQL-side dashboard aggregation and splitting `app.py` were not done. The 45 pending files were not committed (no instruction to) |
| 8. Training metrics | Rolling mean loss, per-set validation loss and perplexity, z-loss, total objective, dashboard charts and KPI. The open question is answered: the trainer evaluates one `val/<task>_loss` per task with validation data, and logs it only locally | Resume / JD / knowledge / general-text sets appear only if the trainer evaluates them (stage 1 has one combined `pretrain` set) |
