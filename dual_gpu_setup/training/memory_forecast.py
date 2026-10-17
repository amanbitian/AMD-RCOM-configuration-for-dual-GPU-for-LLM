"""Training memory forecasting: measured profiles first, an analytic estimate otherwise.

The estimate is deliberately conservative and always labelled as one. After every real run the
measured peak is recorded against a key (model, sequence length, micro-batch, precision, LoRA,
checkpointing, GPU), and from then on the measurement replaces the estimate for that key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .store import TrainingStore

GIB = 1024**3
_BYTES_PER_PARAM = {"fp32": 4.0, "float32": 4.0, "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0, "float16": 2.0,
                    "int8": 1.06, "8bit": 1.06, "nf4": 0.56, "int4": 0.56, "4bit": 0.56}


def _first(config: dict[str, Any], *names: str) -> Any:
    for name in names:
        if config.get(name) is not None:
            return config[name]
    return None


def profile_key(config: dict[str, Any] | None, gpu: str | None) -> dict[str, Any]:
    c = config or {}
    precision = _first(c, "precision", "torch_dtype")
    if precision is None:
        precision = "bf16" if c.get("bf16") else "fp16" if c.get("fp16") else None
    targets = _first(c, "lora_target_modules", "target_modules")
    return {
        "model": _first(c, "base_model", "model_name_or_path", "model"),
        "seq_len": _first(c, "max_seq_length", "max_length", "cutoff_len", "max_seq_len"),
        "microbatch": _first(c, "per_device_train_batch_size", "micro_batch_size", "batch_size"),
        "precision": str(precision).lower() if precision else None,
        "quantized_base": _first(c, "load_in_4bit", "load_in_8bit", "quantization"),
        "lora_r": _first(c, "lora_r", "r"),
        "lora_targets": sorted(targets) if isinstance(targets, list) else targets,
        "gradient_checkpointing": bool(c.get("gradient_checkpointing", False)),
        "gpu": gpu,
    }


def key_hash(key: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def estimate(config: dict[str, Any] | None, *, flash_attention: bool = True) -> dict[str, Any]:
    """Analytic estimate of peak training memory in bytes, broken down by component.

    Activations follow the standard transformer accounting (about 34 bytes per token-hidden per
    layer in bf16, plus the attention-score term when flash attention is off); gradient
    checkpointing keeps only layer inputs and recomputes one layer at a time."""
    c = config or {}
    key = profile_key(c, None)
    total_params = _first(c, "total_params", "num_parameters")
    trainable = _first(c, "trainable_params") or total_params
    hidden = _first(c, "hidden_size", "n_embd")
    layers = _first(c, "num_hidden_layers", "num_layers", "n_layer")
    heads = _first(c, "num_attention_heads", "n_head")
    vocab = _first(c, "vocab_size")
    seq, batch = key["seq_len"], key["microbatch"]
    missing = [name for name, value in (("total_params", total_params), ("seq_len", seq), ("microbatch", batch)) if not value]
    if missing:
        return {"estimate": True, "complete": False, "missing": missing, "components": {}, "total_bytes": None}
    precision = key["precision"] or "bf16"
    weight_bytes = float(total_params) * _BYTES_PER_PARAM.get(str(key["quantized_base"] or precision).lower(),
                                                              _BYTES_PER_PARAM.get(precision, 2.0))
    adapter_dtype = 4.0 if trainable != total_params else _BYTES_PER_PARAM.get(precision, 2.0)
    grads = float(trainable) * adapter_dtype
    optimizer = float(trainable) * 8.0  # AdamW: two fp32 moments
    components: dict[str, float] = {"weights": weight_bytes, "gradients": grads, "optimizer_state": optimizer}
    notes = []
    if hidden and layers:
        per_layer_tokens = float(seq) * float(batch) * float(hidden)
        attention_term = 0.0 if flash_attention or not heads else 5.0 * heads * seq / hidden
        if key["gradient_checkpointing"]:
            activations = layers * 2.0 * per_layer_tokens + per_layer_tokens * (34.0 + attention_term)
        else:
            activations = layers * per_layer_tokens * (34.0 + attention_term)
        components["activations"] = activations
    else:
        notes.append("hidden_size/num_hidden_layers missing: activations not included")
    if vocab:
        components["logits_and_loss"] = float(batch) * float(seq) * float(vocab) * 10.0  # bf16 logits + fp32 upcast + grad
    subtotal = sum(components.values())
    components["allocator_and_workspace"] = 1.0 * GIB + 0.08 * subtotal
    return {"estimate": True, "complete": not notes, "missing": [], "notes": notes, "components": components,
            "total_bytes": sum(components.values())}


def forecast(store: TrainingStore, config: dict[str, Any] | None, lane: dict[str, Any], *,
             safety_fraction: float = 0.90, flash_attention: bool = True) -> dict[str, Any]:
    """Will this configuration fit this lane? `lane` needs key, vram_gb (and optionally pci_bus)."""
    key = profile_key(config, lane.get("pci_bus") or lane.get("key"))
    measured = store.get_profile(key_hash(key))
    budget = lane["vram_gb"] * GIB * safety_fraction
    if measured and measured["peak_mem_bytes"]:
        peak, source, breakdown = measured["peak_mem_bytes"], "measured", None
    else:
        est = estimate(config, flash_attention=flash_attention)
        peak, source, breakdown = est["total_bytes"], "estimate", est
    return {
        "lane": lane.get("key"), "source": source, "key": key, "peak_bytes": peak, "budget_bytes": budget,
        "fits": None if peak is None else peak <= budget,
        "headroom_bytes": None if peak is None else budget - peak,
        "measured_tokens_per_sec": measured["tokens_per_sec"] if measured else None,
        "estimate": breakdown,
    }


def record_run_profile(store: TrainingStore, run: dict[str, Any], gpu: str | None) -> bool:
    """Persist a finished run's measured peak (call after a completed run)."""
    if run["status"] != "completed" or not run["peak_mem_bytes"]:
        return False
    key = profile_key(run.get("config"), gpu)
    if not key["seq_len"] or not key["microbatch"]:
        return False
    store.record_profile(key_hash(key), key, gpu_pci_bus=gpu, peak_mem_bytes=run["peak_mem_bytes"],
                         tokens_per_sec=run["throughput"]["tokens_per_sec"], run_id=run["id"])
    return True
