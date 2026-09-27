# Chatbot Layer Implementation and Validation Plan

## Boundary with the current code

The repository already has useful building blocks:

- `config.py` loads lanes, models, context defaults, and reasoning values.
- `lmstudio.py` discovers runtimes, model paths, and backend devices.
- `server.py` constructs and manages `llama-server` processes and performs a one-time GPU-memory check.
- `orchestrator.py` runs two single-GPU workers followed by multi-GPU models.
- `tasks.py` runs external programs against a temporary endpoint.

The interactive chatbot should reuse these concepts, but it should not call the batch `DualGPUOrchestrator.run()` for each message. A persistent service needs independent deployment lifecycle, request scheduling, and conversation lifecycle.

## Proposed package layout

```text
dual_gpu_setup/
  api/
    app.py
    dependencies.py
    routes_system.py
    routes_deployments.py
    routes_conversations.py
    routes_runs.py
    schemas.py
  chat/
    context_builder.py
    generation.py
    scheduler.py
    events.py
  deployments/
    manager.py
    models.py
    validator.py
  backends/
    base.py
    llama_cpp.py
    capabilities.py
  telemetry/
    collector.py
    gpu_windows.py
    process.py
    aggregation.py
  storage/
    database.py
    migrations/
    repositories.py
  web/
    ...frontend...
```

Keep the existing batch CLI operational. Add a command such as:

```powershell
dual-gpu --config .\example.dual_gpu.toml serve
```

## Work phases

### Phase 1: lifecycle refactor

- Extract device reservation, server start/stop, health, and state transitions from the batch orchestrator.
- Give lanes stable physical GPU identifiers in addition to display-name matching.
- Add deployment specifications for `single_large_model`, `parallel_models`, and `single_gpu`.
- Add preflight validation for ports, backend compatibility, model path, estimated memory, context, tensor split, and conflicting reservations.
- Preserve current CLI behavior with regression tests.

Exit condition: all three profiles can be started/stopped through a Python API, and the existing batch run still works.

### Phase 2: persistence

- Add SQLite configuration, migrations, WAL mode, foreign keys, and transaction boundaries.
- Implement conversation, message, deployment, run, and metric repositories.
- Recover interrupted runs on startup.
- Add retention and export foundations.

Exit condition: a synthetic run round-trips through storage without losing request, response, token, timing, placement, or error fields.

### Phase 3: normalized backend adapter

- Wrap `llama-server` health, model metadata, tokenization, streaming chat, cancellation, usage, and timing.
- Probe runtime version and capabilities.
- Normalize backend-native fields while retaining raw payloads.
- Parse offload, cache, and timing information only behind versioned parsers.

Exit condition: unsupported fields return `null` with a reason and supported fields pass fixture-based parser tests.

### Phase 4: telemetry

- Implement process/system CPU and RAM sampling.
- Implement per-process dedicated/shared GPU memory sampling.
- Add physical GPU utilization, VRAM, temperature, clock, and power adapters where available.
- Aggregate time-weighted means, peaks, baselines, and spill indicators.
- Measure sampling overhead.

Exit condition: a completed run has raw samples and deterministic summary metrics, and inference continues if an optional metric source fails.

### Phase 5: chat and compare APIs

- Add context construction and model-specific token counting.
- Validate context accounting before inference.
- Stream normalized SSE events.
- Support direct routing and parallel compare fan-out.
- Add cancellation, timeout, bounded queues, and backpressure.

Exit condition: direct and compare requests persist partial/final output and always reach a terminal state.

### Phase 6: UI

- Build deployment configuration and preflight views.
- Build conversations, branching, composer controls, and compare cards.
- Build live charts, run inspector, history, filtering, and exports.
- Add accessible empty/loading/error states.

Exit condition: the UI acceptance checklist in [UI_SPECIFICATION.md](UI_SPECIFICATION.md) passes.

### Phase 7: hardening

- Add startup reconciliation for orphan processes and interrupted runs.
- Add localhost security defaults, authentication option, audit events, and redaction.
- Add database backup/restore and retention jobs.
- Add load, soak, cancellation, out-of-memory, server-crash, and power-loss tests.

Exit condition: the service recovers safely from process/application crashes without losing completed run data or leaving incorrect GPU reservations.

## Testing strategy

### Unit tests

- Deployment state-machine transitions.
- Reservation conflicts for all three modes.
- Context token budget and overflow policies.
- Reasoning preset mapping per model.
- Token metric normalization and missing-value behavior.
- Timing formulas and zero-token edge cases.
- Time-weighted metric aggregation.
- Spill-warning thresholds.
- Server log/response parsers against versioned fixtures.

### Integration tests

- Fake backend streaming success, cancellation, timeout, malformed chunks, and process death.
- SQLite migrations and restart recovery.
- SSE reconnect and final authoritative event.
- Two compare runs start from identical prompt snapshots.
- One compare target failing does not cancel the other.
- Model profile change invalidates or refreshes capability caches.

### Hardware tests

Run on the actual two-GPU Windows host:

1. Load one small model on GPU A and confirm GPU B remains unreserved.
2. Repeat on GPU B.
3. Load two models simultaneously and confirm each PID maps to its assigned physical GPU.
4. Send simultaneous requests and verify independent streaming and metric attribution.
5. Load one large model across both GPUs and verify per-GPU allocation and tensor split.
6. Test the largest supported context for each mode and validate KV-cache estimates against observed memory.
7. Force an unsafe context/model combination and verify rejection occurs before load.
8. Cancel during prefill and decode; verify partial output, final samples, and released request slots.
9. Kill one server during compare mode and verify the other finishes.
10. Restart the app during an active run and verify it is marked interrupted and processes are reconciled.

### Performance tests

- Measure telemetry overhead at 2 s, 500 ms, 250 ms, and 100 ms intervals.
- Compare cold and warm time to first token.
- Measure prefill and decode separately over increasing input/output sizes.
- Compare single-GPU versus multi-GPU throughput for the same model when both modes fit.
- Measure parallel comparison wall time versus sequential execution.
- Track memory and throughput as context size increases.
- Run a multi-hour soak test and check server handles, database growth, memory leaks, and metric gaps.

## Definition of done

- The implementation meets the acceptance criteria in [CHATBOT_LAYER.md](CHATBOT_LAYER.md).
- All persisted metrics follow [TELEMETRY_AND_STORAGE.md](TELEMETRY_AND_STORAGE.md).
- All user-visible behavior follows [UI_SPECIFICATION.md](UI_SPECIFICATION.md).
- Existing `inspect`, `plan`, and `run` CLI flows remain functional.
- Tests cover success, cancellation, partial failure, crash recovery, and unavailable telemetry.
- Documentation clearly labels backend-dependent fields and contains no assumed measurements presented as facts.

