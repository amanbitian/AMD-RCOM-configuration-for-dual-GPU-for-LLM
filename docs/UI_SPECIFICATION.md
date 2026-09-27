# Chatbot and Monitoring UI Specification

## Product structure

The UI should make the common path simple—choose a mode, load model(s), and chat—while keeping low-level inference controls and telemetry available without crowding the composer.

Recommended primary navigation:

1. **Chat**: conversations, model responses, and request controls.
2. **Deployment**: execution mode, GPU/model placement, fit validation, and load/unload state.
3. **Live metrics**: GPU, CPU, RAM, throughput, and queue charts.
4. **Run history**: stored prompts/outputs, settings, metrics, comparison, and export.
5. **Models**: model catalog, context limits, reasoning profiles, and compatibility.
6. **Settings**: storage, retention, sampling, security, and advanced runtime defaults.

## Global status bar

The global status bar remains visible on every page and shows:

- Service health and active execution mode.
- GPU A and GPU B name, utilization, dedicated VRAM used/total, shared memory, and temperature if available.
- Loaded model name on each GPU or `Shared by <model>` for a multi-GPU deployment.
- Active requests and queue depth.
- Load/unload progress and errors.
- An emergency **Stop generation** action; model unloading is a separate confirmed action.

## Deployment page

Use three large mode cards:

### One large model / both GPUs

Fields:

- Model selector with size, quantization, trained context, and compatibility badge.
- GPU A + GPU B reservation diagram.
- Tensor split selector: `Auto` or custom numeric weights.
- Context-window selector and estimated KV-cache memory.
- Reasoning profile selector.
- Expandable runtime settings.

Validation panel:

- Estimated model weights per GPU.
- Estimated KV cache and runtime overhead.
- Safety reserve.
- Expected shared-memory spill risk.
- Any backend mismatch.

### Two models / parallel

Show two columns, one per GPU. Each column contains:

- Physical GPU and lane name.
- Model selector filtered to models estimated to fit.
- Context and reasoning selectors.
- Estimated memory stack.
- Endpoint/port in advanced details.

The page should warn if both choices resolve to the same physical GPU, ports conflict, or either model exceeds its lane budget.

### One GPU only

Fields:

- GPU selector.
- Model selector filtered for that GPU.
- Context and reasoning selectors.
- Toggle for optional explicit failover, off by default.

The other GPU is displayed as `Not reserved by chatbot`.

### Apply behavior

Before **Apply configuration** is enabled, call the validation endpoint and show:

- valid/invalid result;
- memory estimate and confidence/source;
- processes that will stop and start;
- conversations or active runs that would be affected;
- whether a reload is required.

Applying a profile opens a progress panel:

```text
Draining requests -> stopping old server -> reserving GPU(s)
-> loading model(s) -> health check -> warm-up -> ready
```

If activation fails, show the exact failed stage and whether the previous profile was restored.

## Chat page

### Layout

```text
+----------------+--------------------------------------+------------------+
| Conversations  | Message timeline                     | Run inspector    |
|                |                                      |                  |
| New / search   | User and assistant messages          | Tokens           |
| Branches       | Parallel answers side by side        | Timing           |
| Archived       | Composer + common controls           | Resources        |
+----------------+--------------------------------------+------------------+
```

The right inspector may collapse on small screens.

### Composer controls

Always visible:

- Target: model A, model B, both/compare, or the single active model.
- Context usage bar.
- Reasoning: Off, Low, Medium, High, Custom, limited to the model's capabilities.
- Maximum visible output tokens.
- Temperature.
- Send/stop.

Advanced drawer:

- `top_p`, `top_k`, `min_p`, seed, repeat controls, and stop sequences.
- Context-overflow policy.
- System-prompt profile.
- Request timeout.
- Whether to retain prompt/output and detailed token timestamps.

### Context usage bar

Use distinct segments:

```text
[system][history][tools][new message][reserved thinking][reserved visible output][free]
```

The user can hover or focus each segment for exact token counts. If the selected model uses a different tokenizer in compare mode, show one bar per model.

Warnings:

- Yellow at 80% effective context usage.
- Red when the next request cannot fit.
- Explicit notice when changing context requires a model reload.

### Reasoning display

Show the selected preset and numeric budget together, for example `Medium · 4,096 tokens`. Values come from the model profile.

If the model/backend does not support a numeric budget, disable the custom field and explain why. If thinking tokens cannot be measured separately, show `Thinking tokens: unavailable from backend`, not `0`.

Reasoning text should be hidden by default and displayed only when it is returned, storage policy permits it, and the user chooses to reveal it.

### Parallel comparison

When target is **Both / Compare**:

- Freeze one prompt snapshot and show two response cards.
- Stream each response independently.
- Show state, stop action, time to first token, output speed, token totals, and resource summary for each card.
- Allow stopping one model without stopping the other.
- Keep an error card visible if one model fails.
- Allow the user to select a preferred answer and optionally continue a new branch from either answer.
- Show configuration differences directly above the responses.

Do not claim the faster model is better; speed and user preference are separate measures.

## Run inspector

For the selected response, show:

### Token panel

- Input tokens.
- Cached and uncached input tokens.
- Prefill tokens.
- Thinking tokens and requested thinking budget.
- Visible output tokens.
- Total output and total tokens.
- Effective context window and percent used.

### Timing panel

- Queue time.
- Time to first token.
- Prefill time and prefill tokens/second.
- Decode time and output tokens/second.
- End-to-end latency.
- Load/warm-up time when the run caused a deployment change.

### Resource panel

- Per-GPU average/peak utilization and peak dedicated VRAM.
- Peak shared GPU memory and spill warning.
- Average/peak process CPU.
- Peak process RAM and total system RAM pressure.
- Temperature, power, energy/token, and throttling when available.

### Configuration panel

- Model fingerprint and quantization.
- Mode and actual GPU IDs.
- Context, reasoning, and generation controls.
- Tensor split, GPU layers, CPU offload, threads, batch settings, flash attention, and cache settings.
- Runtime, driver, and application versions.

## Live metrics page

Default charts:

- GPU utilization by device.
- Dedicated VRAM and shared GPU memory by device.
- Process and system CPU.
- Process RAM and total system RAM.
- Rolling prefill/decode token rate.
- Active requests and queue depth.
- Temperature and power when available.

Interaction:

- Time ranges: last 1 minute, 5 minutes, active run, and custom.
- Vertical markers for request start, first token, completion, cancellation, server load, and error.
- Toggle raw versus smoothed data.
- Hover synchronization across charts.
- Filter by run, model, GPU, or deployment.

Use consistent colors for GPU A and GPU B everywhere. Avoid charting unavailable fields as a zero line.

## Run history page

Table columns:

- Time, conversation, model, mode, GPU(s), status.
- Input/output/thinking token counts.
- Time to first token, prefill tokens/second, output tokens/second, total latency.
- Peak VRAM/shared memory, process RAM, and average GPU utilization.
- Context and reasoning preset.

Filters:

- Date range, model, mode, GPU, status, conversation, context preset, reasoning preset, spill warning, and cold/warm run.

Actions:

- Open complete run details.
- Re-run using the same settings against the same or a different model.
- Compare selected runs.
- Export JSON, CSV, or Parquet where supported.
- Delete a run or its prompt/output content according to retention policy.

The compare view must show configuration differences before performance differences so users do not compare incompatible runs unknowingly.

## Model catalog page

For each model show:

- Name, path, file size, architecture, quantization, and fingerprint.
- Trained maximum context and configured safe maximum.
- Estimated fit on GPU A, GPU B, and both GPUs.
- Supported reasoning modes and model-specific Low/Medium/High mappings.
- Whether thinking tokens, cached tokens, and native timing are reported.
- Recommended tensor split and context presets, with source/confidence.
- Last successful load, load duration, peak memory, and known errors.

Allow editing configuration overrides, but label probed/model-metadata values separately from user overrides.

## Accessibility and responsive behavior

- All state and chart colors need text/icon equivalents.
- Keyboard users must be able to select a conversation, send/cancel, and inspect metrics.
- Use live-region announcements for generation start, completion, cancellation, and errors, but do not announce every token.
- Tables and charts need textual summaries.
- Preserve focus when streaming content updates.
- On narrow screens, stack compare cards vertically and move the inspector into a drawer.

## Empty, loading, and error states

- No deployment: explain the three modes and link to Deployment.
- Loading: show stage, elapsed time, server-log link, and cancel option when safe.
- No conversations: focus the new-chat action.
- Metrics unavailable: identify the missing source and keep chat functional.
- Model failure: show the error, captured log location, and safe recovery actions.
- Context overflow: show exact token breakdown and explicit truncation/summary/branch choices.

## UI acceptance checklist

- The active mode and actual GPU placement are visible without opening settings.
- Context and reasoning controls are visible at message composition time.
- Model-specific reasoning budgets replace global hard-coded presets.
- Compare mode streams and controls both models independently.
- Every response links to its stored run and full configuration.
- Live and final token rates are visually distinguished.
- Shared GPU memory is not mislabeled as definitive spill.
- Missing metrics render as unavailable.
- Settings that trigger reload are labeled before the user changes them.
- All destructive actions, including deleting history or unloading a busy model, require clear scope and confirmation.

