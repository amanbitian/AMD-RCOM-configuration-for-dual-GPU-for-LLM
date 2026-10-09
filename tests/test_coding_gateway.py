"""Exercise the agent gateway without loading a model or requiring GPU hardware."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

import app
from dual_gpu_setup.config import LaneConfig, ModelConfig, load_config
from dual_gpu_setup.server import LlamaServerProcess


@pytest.fixture
def gateway(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    store = app.RunStore(tmp_path / "runs.sqlite3")
    lane = LaneConfig("r9700", "R9700", "R9700", 32, 8081)
    process = SimpleNamespace(base_url="http://backend", device="ROCm0", ctx_size=32768)
    handle = app.ServerHandle("r9700", ModelConfig("test-model", "test.gguf", reasoning_budget=-1,
                                                 reasoning_mode="auto"), [lane], process)
    sampler = SimpleNamespace(
        latest=Mock(return_value=None), since=Mock(return_value=[]),
        capture=Mock(side_effect=AssertionError("Blocking GPU sampling on request path")),
    )
    manager = SimpleNamespace(lock=threading.Lock(), state="ready", handles={"r9700": handle},
                              deployment_id=None, mode="single_gpu", sampler=sampler, config=config)
    state = SimpleNamespace(config=config, store=store, manager=manager,
                            chat=app.ChatService(manager, store))
    handler = object.__new__(app.RequestHandler)
    handler.server = SimpleNamespace(app_state=state)
    handler.headers = {}
    handler.wfile = io.BytesIO()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler._send_json = Mock()
    return handler, store, sampler


def test_stream_preserves_tool_calls_and_usage_without_blocking_sampling(gateway, monkeypatch):
    handler, store, sampler = gateway
    tool = {"index": 0, "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"app.py"}'}}
    chunks = [
        {"choices": [{"delta": {"tool_calls": [tool]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
         "timings": {"cache_n": 30, "prompt_n": 10, "predicted_n": 10, "predicted_ms": 400}},
        {"choices": [], "usage": None},
    ]
    wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    backend = Mock(return_value=io.BytesIO(wire))
    monkeypatch.setattr(app.urlrequest, "urlopen", backend)
    payload = {"model": "r9700", "stream": True, "messages": [{"role": "user", "content": "Inspect"}],
               "tools": [{"type": "function", "function": {"name": "read_file"}}], "temperature": 0.3}
    handler._proxy_openai_chat("/v1/r9700/developer/chat/completions", payload)
    sent = json.loads(backend.call_args.args[0].data)
    assert sent["tools"] == payload["tools"]
    assert sent["messages"] == payload["messages"]
    assert sent["temperature"] == 0.3
    assert sent["stream_options"]["include_usage"] is True
    assert "stream_options" not in payload
    assert handler.wfile.getvalue() == wire
    run = store.get(store.recent(1)[0]["id"])
    assert run["status"] == "completed"
    assert run["finish_reason"] == "tool_calls"
    assert run["timing"]["time_to_first_token_ms"] is not None
    assert run["timing"]["tokens_per_second"] == 25
    assert run["usage"]["output_tokens"] == 10
    assert run["usage"]["tool_calls_total"] == 1  # accumulated across stream deltas
    assert run["usage"]["tool_calls_malformed"] == 0  # '{"path":"app.py"}' parses
    assert run["usage"]["tool_call_valid_rate"] == 1.0
    assert run["output_text"] is None
    assert run["request"]["messages"][0]["content_redacted"] is True
    sampler.capture.assert_not_called()


def test_streamed_tool_call_with_broken_json_args_is_counted_malformed(gateway, monkeypatch):
    handler, store, _ = gateway
    # Arguments arrive in two fragments that together are NOT valid JSON.
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                "function": {"name": "edit", "arguments": '{"path":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'oops'}}]},
                      "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 8}},
    ]
    wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    monkeypatch.setattr(app.urlrequest, "urlopen", Mock(return_value=io.BytesIO(wire)))
    handler._proxy_openai_chat("/v1/r9700/developer/chat/completions",
                               {"stream": True, "messages": [], "tools": [{"type": "function"}]})
    run = store.get(store.recent(1)[0]["id"])
    assert run["usage"]["tool_calls_total"] == 1
    assert run["usage"]["tool_calls_malformed"] == 1
    assert run["usage"]["tool_call_valid_rate"] == 0.0


def test_invalid_grammar_is_rejected_before_calling_the_backend(gateway, monkeypatch):
    handler, store, _ = gateway
    backend = Mock()
    monkeypatch.setattr(app.urlrequest, "urlopen", backend)
    with pytest.raises(app.AppError, match="grammar"):
        handler._proxy_openai_chat("/v1/r9700/developer/chat/completions",
                                   {"messages": [], "grammar": 123})
    backend.assert_not_called()
    assert store.count_runs() == 0


def test_structured_output_validation_rules():
    ok = app.ChatService._validate_structured_output
    ok({"grammar": "root ::= \"x\""})
    ok({"response_format": {"type": "json_object"}})
    ok({"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {}}}})
    for bad, match in [
        ({"grammar": ""}, "grammar"),
        ({"grammar": 5}, "grammar"),
        ({"response_format": {"type": "xml"}}, "response_format.type"),
        ({"response_format": {"type": "json_schema"}}, "json_schema"),
        ({"grammar": "root ::= \"x\"", "response_format": {"type": "json_object"}}, "not both"),
    ]:
        with pytest.raises(ValueError, match=match):
            app.ChatService._validate_structured_output(bad)


def test_tool_call_stats_classifies_each_call():
    stats = app.ChatService._tool_call_stats([
        {"function": {"name": "read", "arguments": '{"p":"a"}'}},  # valid
        {"function": {"name": "noargs", "arguments": ""}},          # valid (no-arg call)
        {"function": {"name": "edit", "arguments": "{broken"}},    # malformed JSON
        {"function": {"arguments": '{"p":1}'}},                      # malformed: no name
    ])
    assert stats["tool_calls_total"] == 4
    assert stats["tool_calls_malformed"] == 2
    assert stats["tool_call_valid_rate"] == 0.5
    assert app.ChatService._tool_call_stats(None)["tool_call_valid_rate"] is None


@pytest.mark.parametrize("stream", [False, True])
def test_response_and_opt_out_preserved(gateway, monkeypatch, stream):
    handler, store, sampler = gateway
    handler.headers = {"X-DGPU-Capture-Content": "true"}
    result = {"choices": [{"message": {"content": "ok"}, "delta": {"content": "ok"},
                           "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
    wire = ("data: " + json.dumps(result) + "\n\ndata: [DONE]\n\n" if stream else json.dumps(result)).encode()
    backend = Mock(return_value=io.BytesIO(wire))
    monkeypatch.setattr(app.urlrequest, "urlopen", backend)
    handler._proxy_openai_chat("/v1/r9700/developer/chat/completions", {
        "stream": stream, "stream_options": {"include_usage": False}, "messages": [],
    })
    assert json.loads(backend.call_args.args[0].data)["stream_options"]["include_usage"] is False
    run = store.get(store.recent(1)[0]["id"])
    assert run["output_text"] == "ok"
    assert run["status"] == "completed"
    sampler.capture.assert_not_called()


def test_missing_pinned_lane_never_falls_back_to_other_model(gateway, monkeypatch):
    handler, store, _ = gateway
    backend = Mock()
    monkeypatch.setattr(app.urlrequest, "urlopen", backend)
    with pytest.raises(app.AppError, match="not loaded"):
        handler._proxy_openai_chat("/v1/9070xt/qa/chat/completions", {"model": "r9700"})
    backend.assert_not_called()
    assert store.count_runs() == 0


def test_backend_failure_does_not_block_on_sampling(gateway, monkeypatch):
    handler, store, sampler = gateway
    monkeypatch.setattr(app.urlrequest, "urlopen", Mock(side_effect=HTTPError(
        "http://backend", 400, "bad context", {}, io.BytesIO(b"context exceeded"))))
    with pytest.raises(app.AppError, match="context exceeded"):
        handler._proxy_openai_chat("/v1/r9700/developer/chat/completions", {"messages": []})
    assert store.recent(1)[0]["status"] == "failed"
    sampler.capture.assert_not_called()


@pytest.mark.parametrize("budget,mode,expected", [(1024, None, "on"), (0, None, "off"),
                                                 (4096, "auto", "auto"), (4096, "off", "off")])
def test_reasoning_selector_reaches_server_command(tmp_path, monkeypatch, budget, mode, expected):
    config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    source = ModelConfig("test", str(model_path))
    manager = object.__new__(app.DeploymentManager)
    manager.config = config
    manager._model = lambda _: source
    raw = {"model": "test", "context_window": 32768, "reasoning_budget": budget}
    if mode is not None:
        raw["reasoning_mode"] = mode
    model, context = manager._configured_model(raw, 32)
    monkeypatch.setattr("dual_gpu_setup.server.server_binary", lambda *_: Path("llama-server.exe"))
    process = LlamaServerProcess(config=config, backend="rocm", device="ROCm0", port=8081,
                                 model=model, model_path=model_path, ctx_size=context,
                                 host="127.0.0.1", run_dir=tmp_path)
    command = process.build_command()
    assert command[command.index("--reasoning") + 1] == expected
    assert command[command.index("--reasoning-budget") + 1] == str(budget)


def test_speculative_draft_flags_reach_server_command(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    draft_path = tmp_path / "draft.gguf"
    draft_path.touch()
    model = ModelConfig("test", str(model_path), draft_model=str(draft_path),
                        draft_max=6, draft_min=1, draft_p_min=0.8, draft_gpu_layers=40)
    monkeypatch.setattr("dual_gpu_setup.server.server_binary", lambda *_: Path("llama-server.exe"))
    process = LlamaServerProcess(config=config, backend="rocm", device="ROCm0", port=8081,
                                 model=model, model_path=model_path, ctx_size=16384,
                                 host="127.0.0.1", run_dir=tmp_path)
    command = process.build_command()
    assert command[command.index("--model-draft") + 1] == str(draft_path)
    assert command[command.index("--gpu-layers-draft") + 1] == "40"
    assert command[command.index("--device-draft") + 1] == "ROCm0"  # shares the target's lane
    assert command[command.index("--draft-max") + 1] == "6"
    assert command[command.index("--draft-min") + 1] == "1"
    assert command[command.index("--draft-p-min") + 1] == "0.8"


def test_no_draft_flags_when_speculation_disabled(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "example.dual_gpu.toml")
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    monkeypatch.setattr("dual_gpu_setup.server.server_binary", lambda *_: Path("llama-server.exe"))
    process = LlamaServerProcess(config=config, backend="rocm", device="ROCm0", port=8081,
                                 model=ModelConfig("test", str(model_path)), model_path=model_path,
                                 ctx_size=16384, host="127.0.0.1", run_dir=tmp_path)
    command = process.build_command()
    assert "--model-draft" not in command
    assert "--device-draft" not in command


def test_speculative_acceptance_rate_from_backend_timings():
    timing = app.ChatService._normalize_timing(
        {"timings": {"predicted_ms": 1000, "draft_n": 20, "draft_n_accepted": 15}},
        {"output_tokens": 50}, 2)
    assert timing["draft_tokens"] == 20
    assert timing["draft_accepted_tokens"] == 15
    assert timing["draft_acceptance_rate"] == 0.75
    # No draft model loaded -> no speculative stats, never a fabricated zero.
    plain = app.ChatService._normalize_timing({"timings": {"predicted_ms": 1000}}, {"output_tokens": 50}, 2)
    assert plain["draft_tokens"] is None
    assert plain["draft_acceptance_rate"] is None


def test_fully_cached_prompt_and_zero_completion_are_not_missing():
    payload = {"usage": {"prompt_tokens": 100, "completion_tokens": 0},
               "timings": {"prompt_n": 0, "cache_n": 100, "prompt_ms": 0, "predicted_n": 0}}
    usage = app.ChatService._normalize_usage(payload)
    timing = app.ChatService._normalize_timing(payload, usage, 2)
    assert usage["prefill_tokens"] == 0
    assert usage["cached_input_tokens"] == 100
    assert usage["output_tokens"] == 0
    assert timing["prefill_duration_ms"] == 0
    assert timing["backend_prompt_tokens"] == 0


def test_native_prompt_counts_include_cache_in_total():
    usage = app.ChatService._normalize_usage({"timings": {"prompt_n": 10, "cache_n": 90, "predicted_n": 5}})
    assert usage["input_tokens"] == 100
    assert usage["prefill_tokens"] == 10
    assert usage["total_tokens"] == 105


def test_prefill_latency_is_not_reported_as_decode_speed():
    usage = {"output_tokens": 100}
    timing = app.ChatService._normalize_timing({}, usage, 20)
    assert timing["tokens_per_second"] is None
    assert timing["end_to_end_tokens_per_second"] == 5
    timing = app.ChatService._normalize_timing({"timings": {"predicted_ms": 4000}}, usage, 20)
    assert timing["tokens_per_second"] == 25


def test_done_stops_reading_without_waiting_for_backend_eof(gateway, monkeypatch):
    handler, store, _ = gateway
    class PersistentStream(io.BytesIO):
        def readline(self, *args):
            assert self.tell() < len(self.getvalue()), "Gateway waited for EOF after [DONE]"
            return super().readline(*args)
    monkeypatch.setattr(app.urlrequest, "urlopen", Mock(return_value=PersistentStream(b"data: [DONE]\n\n")))
    handler._proxy_openai_chat("/v1/r9700/developer/chat/completions", {"stream": True, "messages": []})
    assert store.recent(1)[0]["status"] == "completed"


def test_truncated_stream_is_not_a_successful_completion(gateway, monkeypatch):
    handler, store, _ = gateway
    monkeypatch.setattr(app.urlrequest, "urlopen", Mock(return_value=io.BytesIO(
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')))
    handler._proxy_openai_chat("/v1/r9700/developer/chat/completions", {"stream": True, "messages": []})
    run = store.get(store.recent(1)[0]["id"])
    assert run["status"] == "failed"
    assert "truncated" in run["error_text"]


def test_reasoning_stream_tracks_span_without_counting_chunks_as_tokens(gateway, monkeypatch):
    handler, store, _ = gateway
    wire = (b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],'
            b'"usage":{"completion_tokens":50}}\n\ndata: [DONE]\n\n')
    monkeypatch.setattr(app.urlrequest, "urlopen", Mock(return_value=io.BytesIO(wire)))
    handler._proxy_openai_chat("/v1/r9700/developer/chat/completions", {"stream": True, "messages": []})
    run = store.get(store.recent(1)[0]["id"])
    assert run["usage"]["thinking_tokens"] is None
    assert run["usage"]["thinking_characters"] == 8
    assert run["usage"]["thinking_tokens_estimated"] == 2  # len("thinking")/4
    assert run["usage"]["thinking_tokens_source"] == "estimated_from_characters"
    assert run["timing"]["reasoning_duration_ms"] is not None
    assert run["timing"]["time_to_first_visible_token_ms"] is not None
    assert run["reasoning_text"] is None
