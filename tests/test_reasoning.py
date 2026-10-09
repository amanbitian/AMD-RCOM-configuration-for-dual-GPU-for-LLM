from pathlib import Path
from types import SimpleNamespace

import pytest

import app
from dual_gpu_setup.config import ModelConfig, load_config
from dual_gpu_setup.reasoning import coding_request, model_reasoning, template_reasoning


QWEN_TEMPLATE = """
{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{% if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
{{ raise_exception('Unexpected effort') }}
{% endif %}
"""


def test_native_levels_are_not_token_budgets():
    info = template_reasoning(QWEN_TEMPLATE)
    assert info["efforts"] == ["default", "low", "medium", "xhigh"]
    assert info["default_effort"] == "xhigh"
    assert info["budget_tokens"] is None
    assert template_reasoning("")["efforts"] == ["default"]


def test_template_capability_cache_invalidates_when_file_changes(gguf_writer):
    path = gguf_writer("qwen.gguf", {"tokenizer.chat_template": QWEN_TEMPLATE})
    assert model_reasoning(path)["default_effort"] == "xhigh"
    path.write_bytes(b"invalid replacement")
    assert model_reasoning(path)["source"] == "unavailable"


@pytest.mark.parametrize("effort", ["default", "low", "medium", "xhigh"])
def test_coding_preserves_native_effort_without_synthetic_budget(effort):
    original = {"messages": [], "reasoning_effort": effort, "max_tokens": 1234,
                "chat_template_kwargs": {"preserve_thinking": True}}
    result, info = coding_request(original, capabilities=template_reasoning(QWEN_TEMPLATE),
                                  default_effort="default", server_budget=-1, server_mode="auto")
    assert info["effective_effort"] == ("xhigh" if effort == "default" else effort)
    assert info["budget_tokens"] is None
    assert info["output_limit_tokens"] == 1234
    assert "reasoning_budget_tokens" not in result
    assert ("reasoning_effort" in result) == (effort != "default")
    assert result["chat_template_kwargs"]["preserve_thinking"] is True
    assert original["reasoning_effort"] == effort


@pytest.mark.parametrize("payload,budget,mode,match", [
    ({}, 8192, "on", "Reload"), ({}, -1, "off", "Reload"),
    ({"thinking_budget_tokens": 1000}, -1, "auto", "numeric"),
    ({"reasoning_effort": "high"}, -1, "auto", "Unsupported"),
])
def test_no_silent_caps_or_unsupported_levels(payload, budget, mode, match):
    with pytest.raises(ValueError, match=match):
        coding_request(payload, capabilities=template_reasoning(QWEN_TEMPLATE),
                       default_effort="default", server_budget=budget, server_mode=mode)


def test_coding_deploy_overrides_legacy_budget_and_off_policy(gguf_writer):
    path = gguf_writer("qwen.gguf", {"tokenizer.chat_template": QWEN_TEMPLATE})
    manager = object.__new__(app.DeploymentManager)
    manager.config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    manager._model = lambda _: ModelConfig("qwen", str(path), reasoning_budget=0, reasoning_mode="off")
    model, _ = manager._configured_model({"model": "qwen", "context_window": 32768,
                                         "workload_kind": "coding_agent", "reasoning_budget": 8192,
                                         "reasoning_effort": "medium"}, 32)
    assert model.reasoning_budget == -1
    assert model.reasoning_mode == "auto"
    assert model.reasoning_effort == "medium"


def test_handle_resolves_capabilities_once_and_skips_reglob(monkeypatch):
    # The model file is fixed for a deployment, so native levels are read once per
    # handle from the already-resolved path -- never re-globbed per request.
    calls = {"reasoning": 0}
    sentinel = {"efforts": ["default", "medium"], "default_effort": "medium",
                "budget_tokens": None, "budget_policy": "model_controlled", "source": "gguf_template"}

    def fake_reasoning(path):
        calls["reasoning"] += 1
        assert str(path) == "resolved.gguf"
        return sentinel

    monkeypatch.setattr(app, "model_reasoning", fake_reasoning)
    monkeypatch.setattr(app, "resolve_model_path",
                        lambda *_: pytest.fail("re-resolved an already-known model path"))
    process = SimpleNamespace(model_path="resolved.gguf")
    handle = app.ServerHandle("r9700", ModelConfig("m", "pattern*.gguf"), [], process)
    config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    assert handle.reasoning_capabilities(config) is sentinel
    assert handle.reasoning_capabilities(config) is sentinel
    assert calls["reasoning"] == 1


def test_missing_reasoning_counts_are_unknown_not_visible_output():
    result = app.ChatService._normalize_usage({"reasoning_observed": True,
                                               "usage": {"completion_tokens": 50}})
    assert result["thinking_tokens"] is None
    assert result["thinking_tokens_estimated"] is None
    assert result["visible_output_tokens"] is None
    assert result["thinking_tokens_source"] == "unavailable"


def test_exact_backend_reasoning_count_wins_over_character_estimate():
    result = app.ChatService._normalize_usage({
        "reasoning_characters": 400,
        "usage": {"completion_tokens": 120,
                  "completion_tokens_details": {"reasoning_tokens": 90}}})
    assert result["thinking_tokens"] == 90
    assert result["thinking_tokens_estimated"] is None
    assert result["thinking_tokens_source"] == "backend"
    assert result["visible_output_tokens"] == 30  # 120 - 90, exact subtraction only


def test_streamed_reasoning_characters_yield_labeled_estimate():
    result = app.ChatService._normalize_usage({
        "reasoning_characters": 400, "usage": {"completion_tokens": 120}})
    assert result["thinking_tokens"] is None
    assert result["thinking_tokens_estimated"] == 100  # 400 / 4 chars-per-token
    assert result["thinking_characters"] == 400
    assert result["thinking_tokens_source"] == "estimated_from_characters"
    assert result["visible_output_tokens"] is None  # never subtract an estimate as exact


def test_non_streamed_reasoning_text_is_counted_for_the_estimate():
    result = app.ChatService._normalize_usage({
        "choices": [{"message": {"content": "answer", "reasoning_content": "x" * 40}}],
        "usage": {"completion_tokens": 20}})
    assert result["thinking_characters"] == 40
    assert result["thinking_tokens_estimated"] == 10
    assert result["thinking_tokens_source"] == "estimated_from_characters"


def test_dashboard_coding_chat_has_no_implicit_output_cap_or_counter_capture(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    model = ModelConfig("fake", "unused.gguf", reasoning_mode="auto", reasoning_budget=-1)
    process = SimpleNamespace(base_url="http://backend", backend="rocm", device="ROCm0",
                              ctx_size=32768, tensor_split="", build_command=lambda: [])
    sampler = SimpleNamespace(latest=lambda: None, since=lambda _: [])
    manager = SimpleNamespace(config=config, sampler=sampler, deployment_id=None,
                              active_profile={}, mode="single_gpu")
    store = app.RunStore(tmp_path / "runs.db")
    service = app.ChatService(manager, store)
    requests = []
    def backend(url, payload, timeout):
        requests.append(payload)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}}}
    service._post_json = backend
    result = service._run_one("r9700", app.ServerHandle("r9700", model, [], process),
                              [{"role": "user", "content": "code"}], {}, None, workload_kind="coding_agent")
    assert result["status"] == "completed"
    assert "max_tokens" not in requests[0]
    assert result["reasoning"]["budget_policy"] == "model_controlled"
    assert store.recent(1)[0]["configuration"]["reasoning"]["server_budget_tokens"] == -1
