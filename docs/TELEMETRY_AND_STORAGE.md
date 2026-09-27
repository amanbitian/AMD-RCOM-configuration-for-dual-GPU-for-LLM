# Telemetry, Token Accounting, and Run Storage

## Purpose

Every model invocation must be reproducible and measurable. This document defines what to collect, how to calculate it, where to store it, and how to represent unavailable data.

A **run** means one request to one model. Sending one prompt to two models creates two runs linked by one `comparison_id`.

## Measurement principles

1. Store raw counters and timestamps in addition to calculated rates.
2. Use a monotonic clock for durations and UTC wall-clock timestamps for correlation.
3. Store bytes, tokens, nanoseconds, and joules as base units where possible; format them only in the UI.
4. Identify the source of every metric: backend response, parsed server timing, OS process counter, GPU API, or derived value.
5. Represent an unavailable measurement as `null`, with an optional `unavailable_reason`. Never turn missing data into zero.
6. Sample resource data before generation, during generation, and once after completion.
7. Keep per-GPU samples. Aggregate values alone cannot explain an uneven tensor split.
8. Version the schema and record application, runtime, driver, and model identifiers on every deployment or run.

## Token definitions

Token fields must have unambiguous meanings:

| Field | Definition |
|---|---|
| `input_tokens` | Tokens sent to the model after the final chat template is applied. Includes system prompt, retained history, tool definitions/results, attachments represented as tokens, and the new user message. |
| `cached_input_tokens` | Input tokens reused from a prompt/KV cache when the backend reports them. |
| `uncached_input_tokens` | `input_tokens - cached_input_tokens`; `null` if cache usage is unknown. |
| `prefill_tokens` | Tokens actually processed during prompt evaluation. Prefer the backend's prompt-evaluation count. This may differ from `input_tokens` when cache reuse is active. |
| `thinking_tokens` | Completion tokens used for hidden or separately reported reasoning. `null` if the backend cannot separate them. |
| `visible_output_tokens` | Tokens in the user-visible assistant answer, excluding separately reported thinking tokens. |
| `output_tokens` | All generated completion tokens. When reasoning is separate, `thinking_tokens + visible_output_tokens`. |
| `total_tokens` | `input_tokens + output_tokens`. |
| `max_output_tokens` | Request limit for completion tokens according to the backend's accounting rule. |
| `thinking_budget` | Requested maximum reasoning tokens, not the number actually used. |
| `context_window` | Effective model/server token capacity for this run. |

When the backend uses a different definition, preserve the native fields under `backend_usage_json` and map to normalized fields only when the mapping is defensible.

## Timing and throughput definitions

Use these monotonic timestamps:

```text
t0 = request accepted by API
t1 = dispatched to ready model server
t2 = backend acknowledges or prompt evaluation begins, when observable
t3 = first output token received
t4 = final output token received
t5 = final run record committed
```

Derived metrics:

```text
queue_duration_ms       = (t1 - t0) * 1000
time_to_first_token_ms  = (t3 - t1) * 1000
end_to_end_latency_ms   = (t5 - t0) * 1000
decode_duration_s       = backend decode duration when reported,
                          otherwise t4 - t3
tokens_per_second       = output_tokens / backend completion duration
visible_tokens_per_sec  = visible_output_tokens / decode_duration_s
prefill_tokens_per_sec  = prefill_tokens / prefill_duration_s
```

Prefer native backend timing because network buffering can make client-observed timing inaccurate. Store both native and observed fields when available.

Do not calculate decode rate from `t4 - t3` for a zero- or one-token completion. Return `null` instead. Label rolling live token rate separately from authoritative final throughput.

## Metrics catalog

### Identity and reproducibility

| Metric | Collection point |
|---|---|
| Run, conversation, message, comparison, and deployment IDs | API/database |
| Requested and actual execution mode | Scheduler |
| Model display name, canonical ID, path hash, file size, quantization, architecture | Model catalog/runtime metadata |
| Model file modification time or content fingerprint | Model catalog |
| Backend and runtime version | Server adapter |
| Application/schema version | Application |
| GPU name, stable ID, driver version, backend device ID | GPU discovery |
| Tensor split, GPU layers, CPU-offloaded layers | Deployment/backend |
| Context requested and allocated | Request and server metadata |
| Exact generation settings and reasoning profile | Request snapshot |
| Seed and chat-template identifier/hash | Request snapshot |

Hashing a very large model file on every load is expensive. Cache the content fingerprint using canonical path, size, and modification time, or use a manifest-provided checksum.

### Output and status

Store:

- Original user message and the exact prompt/message snapshot submitted.
- Partial and final assistant output.
- Separately reported reasoning text only when policy permits it.
- Finish reason such as `stop`, `length`, `tool_call`, `cancelled`, `error`, or backend-native value.
- Run state and error type/message.
- Retry count, timeout, cancellation timestamp, and whether the client disconnected.
- Tool calls and tool results as structured JSON if tools are supported.

### GPU metrics, per physical GPU

Capture these fields when the driver/backend exposes them:

- Utilization percent.
- Dedicated VRAM used, free, total, and peak used.
- Shared GPU memory used and peak used.
- Memory-controller utilization.
- Core clock and memory clock.
- Temperature, hotspot temperature, fan speed.
- Board power in watts, energy in joules if supported, and power limit.
- PCIe/link transfer rate or cumulative bytes, if useful and available.
- Throttling or performance-limit reason.
- Backend device identifier and physical GPU stable identifier.

For multi-GPU mode, store one sample row per GPU per timestamp. Summary fields should include peak VRAM and average/peak utilization for each GPU and for the deployment as a whole.

Windows shared GPU memory is the primary signal for VRAM spill but can include allocations that are not model weights. Report it as **shared GPU memory**, and derive a `spill_suspected` flag using a documented threshold; do not label every shared byte as definite model spillover.

Suggested spill indicators:

```text
shared_gpu_memory_peak_bytes > configured_absolute_threshold
AND
shared_gpu_memory_peak_bytes / max(dedicated_vram_peak_bytes, 1) > configured_ratio_threshold
```

Also correlate process RSS growth and backend offload logs before classifying spill severity.

### CPU and system RAM metrics

Capture:

- Model-server process CPU percent, normalized consistently. Store logical-core count so values can be interpreted.
- Total system CPU percent.
- Model-server process resident set size/working set, private bytes, and peak working set.
- Total system RAM used, available, and percent.
- Pagefile/commit used and available.
- Process read/write bytes and rates when diagnosing offload bottlenecks.
- CPU thread count and configured inference/batch threads.

`process_cpu_pct` must document whether 100% means one fully used logical core or the entire machine. Prefer normalization where 100% means all logical cores fully used in the UI, while retaining the native raw value if the OS counter differs.

### Latency and throughput metrics

Capture:

- Queue duration.
- Model load duration and warm-up duration.
- Time to first token.
- Prefill/prompt-evaluation duration and throughput.
- Decode/completion duration and throughput.
- End-to-end duration.
- Per-token arrival timestamps or a histogram of inter-token latency when detailed analysis is enabled.
- Requests per second and tokens per second at deployment level.
- Active request count, queue depth, and server slot utilization.
- Cache hit/reused-token count.

Per-token timestamps can grow rapidly. Make them an opt-in diagnostic feature or compact them into histograms after calculating percentiles.

### Additional useful measurements

- Estimated KV-cache bytes and actual cache allocation when reported.
- Prompt length by component: system, history, user, tool schema/results, attachments, template overhead.
- Context utilization percent.
- First-load versus warm-run marker.
- Model load failures and out-of-memory events.
- Server restart count and health-check failures.
- Power/energy per output token when energy data is available.
- Effective output cost in time and energy, even though local inference has no API token price.
- Comparison skew: difference between the two parallel run start times.
- User feedback, rating, or selected winner for compare mode.
- Host OS, CPU, total RAM, application version, runtime version, and driver versions.

## Sampling strategy

Recommended defaults:

- Idle dashboard: one sample every 2 seconds.
- Loading and active generation: one sample every 500 milliseconds.
- Final sample immediately when generation ends.
- Optional diagnostic mode: 100–250 milliseconds, with an explicit warning about overhead and storage volume.

The sampler should collect all counters for the same logical timestamp and tag each sample with `run_id`, `deployment_id`, process ID, GPU ID, source, and quality. For parallel requests, shared system samples may be referenced by both runs, but process and GPU ownership must remain unambiguous.

Resource summary calculations:

- `avg_*`: time-weighted average, not a simple average if intervals vary.
- `peak_*`: maximum valid sample during the run.
- `baseline_*`: median of samples just before request dispatch when available.
- `delta_*`: peak or average minus baseline, where that interpretation is useful.

## Data model

SQLite is sufficient for the first version. Enable foreign keys and WAL mode. Use migrations and never rely on application startup to silently rebuild production tables.

### Core tables

```text
conversations
  id, title, created_at, updated_at, archived_at, metadata_json

messages
  id, conversation_id, parent_message_id, role, content_json,
  created_at, model_run_id, status

deployments
  id, requested_mode, actual_mode, state, created_at, ready_at, stopped_at,
  configuration_json, runtime_version, error_json

deployment_models
  id, deployment_id, lane_key, model_id, model_path, model_fingerprint,
  gpu_ids_json, tensor_split, context_allocated, process_id, endpoint,
  server_arguments_json

runs
  id, comparison_id, conversation_id, request_message_id, response_message_id,
  deployment_id, deployment_model_id, status, created_at, started_at,
  first_token_at, finished_at, finish_reason, error_json,
  request_json, prompt_snapshot_json, response_json, backend_usage_json

run_token_metrics
  run_id, input_tokens, cached_input_tokens, uncached_input_tokens,
  prefill_tokens, thinking_tokens, visible_output_tokens, output_tokens,
  total_tokens, max_output_tokens, thinking_budget, context_window,
  context_used_pct

run_timing_metrics
  run_id, queue_duration_ms, time_to_first_token_ms, prefill_duration_ms,
  decode_duration_ms, end_to_end_duration_ms, prefill_tokens_per_second,
  tokens_per_second, visible_tokens_per_second

metric_samples
  id, run_id, deployment_id, sampled_at, monotonic_offset_ms,
  process_cpu_pct, system_cpu_pct, process_rss_bytes, process_private_bytes,
  system_ram_used_bytes, system_ram_available_bytes, pagefile_used_bytes,
  queue_depth, source_json

gpu_metric_samples
  id, metric_sample_id, gpu_id, utilization_pct, memory_utilization_pct,
  dedicated_used_bytes, dedicated_total_bytes, shared_used_bytes,
  temperature_c, hotspot_temperature_c, power_w, energy_j,
  core_clock_hz, memory_clock_hz, fan_pct, throttle_reason, source

run_resource_summaries
  run_id, summary_json

model_profiles
  model_id, capability_source, max_context, default_context,
  reasoning_supported, thinking_tokens_reported, reasoning_profiles_json,
  updated_at
```

Use JSON only for backend-specific or evolving structures. Keep frequently filtered numeric fields in typed columns.

## Canonical persisted run example

```json
{
  "schema_version": 1,
  "run_id": "01J...",
  "comparison_id": null,
  "status": "completed",
  "mode": "single_large_model",
  "model": {
    "id": "gpt-oss-120b-GGUF",
    "fingerprint": "sha256:...",
    "quantization": "MXFP4"
  },
  "placement": {
    "gpu_ids": ["gpu-a", "gpu-b"],
    "tensor_split": [36, 12],
    "gpu_layers": 999,
    "cpu_offloaded_layers": 20
  },
  "request": {
    "context_window": 16384,
    "max_output_tokens": 4096,
    "reasoning_profile": "medium",
    "thinking_budget": 2048,
    "temperature": 0.7,
    "seed": 42
  },
  "tokens": {
    "input_tokens": 1260,
    "cached_input_tokens": 512,
    "prefill_tokens": 748,
    "thinking_tokens": 886,
    "visible_output_tokens": 604,
    "output_tokens": 1490,
    "total_tokens": 2750
  },
  "timing": {
    "queue_duration_ms": 2.4,
    "time_to_first_token_ms": 834.2,
    "prefill_duration_ms": 601.5,
    "prefill_tokens_per_second": 1243.6,
    "decode_duration_ms": 38776.0,
    "tokens_per_second": 38.43,
    "end_to_end_duration_ms": 39690.1
  },
  "resources": {
    "gpu": [
      {"gpu_id": "gpu-a", "peak_vram_bytes": 30064771072, "avg_utilization_pct": 88.2},
      {"gpu_id": "gpu-b", "peak_vram_bytes": 14602888806, "avg_utilization_pct": 76.5}
    ],
    "peak_shared_gpu_memory_bytes": 268435456,
    "spill_suspected": false,
    "avg_process_cpu_pct": 22.4,
    "peak_process_rss_bytes": 4294967296
  }
}
```

The values are illustrative. Production values must come from measurement.

## Collection sources on Windows

Use the best available source and record which source produced each field:

- `llama-server` response usage and timing fields for token counts and native timing.
- `llama-server` logs or metrics endpoint for prompt evaluation, decode timing, KV cache, slots, and offload information when exposed by the installed version.
- Windows performance counters for process CPU, working set/private bytes, dedicated GPU memory, and shared GPU memory.
- Vendor tooling or management libraries for physical GPU utilization, VRAM, clocks, temperature, and power. AMD ROCm/Vulkan availability varies by GPU, driver, and Windows runtime.
- `psutil` for process/system CPU, RAM, disk I/O, and process lifecycle where supported.

The current repository reads Windows `GPU Process Memory` counters once after load. The chatbot requires a continuous collector and a reliable mapping from model-server PID to physical GPU. If a counter cannot identify the physical adapter, combine PID counters with the scheduler's device reservation and clearly mark the attribution method.

## Retention and export

Recommended retention controls:

- Keep conversation and summary records until the user deletes them.
- Keep 500 ms raw samples for 7–30 days, configurable.
- Downsample older samples to 5-second or per-run summaries before deleting raw rows.
- Allow prompt/output retention to be disabled while retaining anonymous performance metrics.
- Support JSON export for complete runs and CSV/Parquet export for tabular metrics.
- Use cascading deletion so deleting a conversation removes its messages and content; allow performance summaries to be retained only with explicit anonymization policy.

## UI presentation rules

- Display `—` or `Unavailable` for `null`; never `0`.
- Put the measurement source and definition in metric tooltips.
- Clearly separate live rolling token rate from final backend-reported rate.
- Show dedicated VRAM and shared GPU memory independently.
- Show per-GPU charts in multi-GPU mode.
- Mark estimated fields, such as predicted KV cache, as estimates.
- Let users compare runs only when configuration differences are visible alongside results.

