"""llama-server runtime selection: official llama.cpp builds vs LM Studio's bundled runtime."""
from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dual_gpu_setup import lmstudio
from dual_gpu_setup.config import ModelConfig, load_config
from dual_gpu_setup.lmstudio import RuntimeInfo, parse_runtime, runtime_info, vendor_bin
from dual_gpu_setup.server import LlamaServerProcess, compat_args

EXAMPLE = Path(__file__).resolve().parents[1] / "example.dual_gpu.toml"
VERSION_TEXT = "version: 0.7.0-dev (build 11699, commit 7182ef71e)\nbuilt with Clang 20.1.8 for Windows x86_64\n"
HELP_TEXT = """
-m,    --model FNAME                    model path
-lm,   --load-mode MODE                 model loading mode (default: auto)
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM
"""
NEW_BUILD = parse_runtime("llama-server.exe", "llama.cpp", VERSION_TEXT, HELP_TEXT)
OLD_BUILD = RuntimeInfo("llama-server.exe", "lmstudio", flags=frozenset({"--model", "--no-mmap", "--gpu-layers"}))
UNPROBED = RuntimeInfo("llama-server.exe", "llama.cpp")


def _discovery(tmp_path: Path, native: str = "") -> SimpleNamespace:
    return SimpleNamespace(discovery=SimpleNamespace(
        lmstudio_home=str(tmp_path / ".lmstudio"), backend_vendor_bin_overrides={},
        backend_runtime_overrides={}, backend_server_binaries={"rocm": native} if native else {}))


def _process(tmp_path: Path, runtime: RuntimeInfo, extra_args: list[str] | None = None) -> LlamaServerProcess:
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    return LlamaServerProcess(config=load_config(EXAMPLE), backend="rocm", device="ROCm0", port=8081,
                              model=ModelConfig("test", str(model_path), extra_args=extra_args or []),
                              model_path=model_path, ctx_size=16384, run_dir=tmp_path, _runtime=runtime)


def test_parse_runtime_reads_build_and_every_long_flag():
    assert (NEW_BUILD.version, NEW_BUILD.build, NEW_BUILD.commit) == ("0.7.0-dev", 11699, "7182ef71e")
    assert {"--model", "--load-mode", "--fit", "--gpu-layers", "--n-gpu-layers"} <= NEW_BUILD.flags
    assert not NEW_BUILD.supports("--no-mmap")
    assert NEW_BUILD.public() == {"binary": "llama-server.exe", "source": "llama.cpp", "version": "0.7.0-dev",
                                  "build": 11699, "commit": "7182ef71e"}


def test_parse_runtime_tolerates_unknown_version_text():
    info = parse_runtime("x", "lmstudio", "something else", "")
    assert info.build is None and info.version == "" and info.flags == frozenset()


def test_renamed_flag_is_rewritten_only_for_builds_that_dropped_it():
    assert compat_args(NEW_BUILD, ["--no-mmap", "--foo"]) == ["--load-mode", "none", "--foo"]
    assert compat_args(OLD_BUILD, ["--no-mmap"]) == ["--no-mmap"]
    assert compat_args(UNPROBED, ["--no-mmap"]) == ["--no-mmap"]  # unknown build: pass through untouched


def test_new_builds_get_fit_off_so_the_repo_keeps_sizing_control(tmp_path):
    command = _process(tmp_path, NEW_BUILD, ["--no-mmap"]).build_command()
    assert command[0] == "llama-server.exe"
    assert command[command.index("--fit") + 1] == "off"
    assert "--no-mmap" not in command and command[command.index("--load-mode") + 1] == "none"


def test_old_or_unprobed_builds_get_an_unchanged_command(tmp_path):
    for runtime in (OLD_BUILD, UNPROBED):
        command = _process(tmp_path, runtime, ["--no-mmap"]).build_command()
        assert "--fit" not in command and "--load-mode" not in command and "--no-mmap" in command


def test_explicit_fit_in_extra_args_wins(tmp_path):
    command = _process(tmp_path, NEW_BUILD, ["--fit", "on"]).build_command()
    assert command.count("--fit") == 1 and command[command.index("--fit") + 1] == "on"


MEMORY_BUILD = RuntimeInfo("llama-server.exe", "llama.cpp",
                           flags=frozenset({"--load-mode", "--cache-ram", "--fit", "--no-host"}))
NO_MMAP_BUILD = RuntimeInfo("llama-server.exe", "lmstudio", flags=frozenset({"--no-mmap", "--cache-ram"}))


def _value(command, flag):
    return command[command.index(flag) + 1]


def test_host_ram_flags_keep_weights_out_of_ram_and_cap_the_prompt_cache(tmp_path):
    command = _process(tmp_path, MEMORY_BUILD).build_command()
    assert _value(command, "--load-mode") == "none" and _value(command, "--cache-ram") == "0"


def test_builds_with_only_no_mmap_get_no_mmap(tmp_path):
    command = _process(tmp_path, NO_MMAP_BUILD).build_command()
    assert "--no-mmap" in command and "--load-mode" not in command and _value(command, "--cache-ram") == "0"


def test_host_ram_flags_follow_policy_and_yield_to_extra_args(tmp_path):
    process = _process(tmp_path, MEMORY_BUILD, ["--cache-ram", "4096", "--load-mode", "mmap"])
    command = process.build_command()
    assert command.count("--cache-ram") == 1 and _value(command, "--cache-ram") == "4096"
    assert command.count("--load-mode") == 1 and _value(command, "--load-mode") == "mmap"

    process = _process(tmp_path, MEMORY_BUILD)
    process.config.policy.load_mode = "auto"  # llama.cpp's own default: pass nothing
    process.config.policy.cache_ram_mib = 2048
    command = process.build_command()
    assert "--load-mode" not in command and _value(command, "--cache-ram") == "2048"


def test_unprobed_builds_get_no_host_ram_flags(tmp_path):
    command = _process(tmp_path, UNPROBED).build_command()
    assert "--cache-ram" not in command and "--load-mode" not in command and "--no-mmap" not in command


GUARD_BUILD = RuntimeInfo("llama-server.exe", "llama.cpp", flags=frozenset({"--kv-unified", "--cache-ram", "--load-mode"}))


def test_single_slot_servers_get_a_spare_warm_up_slot_over_a_unified_kv_pool(tmp_path):
    process = _process(tmp_path, GUARD_BUILD)
    command = process.build_command()
    assert process.warm_slot() == 1
    assert _value(command, "--parallel") == "2" and "--kv-unified" in command


def test_no_warm_up_slot_when_the_guard_is_off_slots_are_explicit_or_the_build_lacks_it(tmp_path):
    process = _process(tmp_path, GUARD_BUILD)
    process.config.policy.residency_guard = False
    assert process.warm_slot() is None and _value(process.build_command(), "--parallel") == "1"

    process = _process(tmp_path, GUARD_BUILD, ["-np", "4"])
    assert process.warm_slot() is None and "--kv-unified" not in process.build_command()

    process = _process(tmp_path, MEMORY_BUILD)  # no --kv-unified in this build
    assert process.warm_slot() is None and _value(process.build_command(), "--parallel") == "1"


def test_load_mode_is_validated(tmp_path):
    bad = tmp_path / "bad.toml"
    text = EXAMPLE.read_text(encoding="utf-8")
    assert 'load_mode = "none"' in text
    bad.write_text(text.replace('load_mode = "none"', 'load_mode = "lazy"'), encoding="utf-8")
    with pytest.raises(SystemExit, match="load_mode"):
        load_config(bad)


def test_configured_build_does_not_inherit_lm_studio_vendor_dlls(tmp_path):
    vendor = tmp_path / ".lmstudio" / "extensions" / "backends" / "vendor" / "win-llama-rocm-vendor-v6"
    (vendor / "bin").mkdir(parents=True)
    binary = tmp_path / "llama.cpp" / "llama-server.exe"
    binary.parent.mkdir()
    binary.touch()
    binary.chmod(0o755)
    assert vendor_bin(_discovery(tmp_path), "rocm") == vendor / "bin"
    assert vendor_bin(_discovery(tmp_path, str(binary)), "rocm") is None
    explicit = _discovery(tmp_path, str(binary))
    explicit.discovery.backend_vendor_bin_overrides = {"rocm": str(vendor)}
    assert vendor_bin(explicit, "rocm") == vendor  # an explicit override still applies


def test_runtime_info_probes_once_and_labels_the_source(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args[1])
        return subprocess.CompletedProcess(args, 0, VERSION_TEXT if args[1] == "--version" else HELP_TEXT, "")

    monkeypatch.setattr(lmstudio.subprocess, "run", fake_run)
    monkeypatch.setattr(lmstudio, "_RUNTIME_CACHE", {})
    config = _discovery(tmp_path)
    official = tmp_path / "llama.cpp" / "llama-server.exe"
    bundled = tmp_path / ".lmstudio" / "extensions" / "backends" / "llama.cpp-win-x86_64-amd-rocm-avx2-2.55.0" / "llama-server.exe"
    for path in (official, bundled):
        path.parent.mkdir(parents=True)
        path.touch()

    first = runtime_info(config, "rocm", official)
    assert runtime_info(config, "rocm", official) is first and calls == ["--version", "--help"]
    assert first.source == "llama.cpp" and first.build == 11699 and first.supports("--fit")
    assert runtime_info(config, "rocm", bundled).source == "lmstudio"


def test_missing_binary_yields_an_unprobed_runtime(tmp_path):
    info = runtime_info(_discovery(tmp_path), "rocm", tmp_path / "nope.exe")
    assert info.flags == frozenset() and info.build is None


def test_device_listing_accepts_linux_logger_output_on_stderr(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(lmstudio, "server_binary", lambda _config, _backend: binary)
    monkeypatch.setattr(lmstudio, "runtime_dir", lambda _config, _backend: tmp_path)
    monkeypatch.setattr(lmstudio, "backend_env", lambda _config, _backend: {})
    monkeypatch.setattr(
        lmstudio,
        "get_host",
        lambda: SimpleNamespace(control=SimpleNamespace(spawn_kwargs=lambda: {})),
    )
    monkeypatch.setattr(
        lmstudio.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "", "ROCm0: AMD Radeon Test (16384 MiB, 12000 MiB free)\n"
        ),
    )

    assert lmstudio.list_devices(SimpleNamespace(), "rocm") == [
        ("ROCm0", "AMD Radeon Test (16384 MiB, 12000 MiB free)")
    ]


def test_linux_native_binary_must_be_executable(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(lmstudio.sys, "platform", "linux")
    monkeypatch.setattr(lmstudio.os, "access", lambda path, mode: False)
    with pytest.raises(SystemExit, match="chmod \\+x"):
        lmstudio.native_server_binary(_discovery(tmp_path, str(binary)), "rocm")


def test_split_models_can_fall_back_to_lm_studio_while_lanes_keep_the_configured_build():
    config = load_config(EXAMPLE)
    official = config.discovery.backend_server_binaries
    assert official and config.discovery.multi_gpu_runtime == "lmstudio"
    split = lmstudio.multi_gpu_config(config)
    assert split.discovery.backend_server_binaries == {}
    assert config.discovery.backend_server_binaries == official  # the shared config is untouched
    assert split.policy is config.policy and split.lanes is config.lanes

    configured = replace(config, discovery=replace(config.discovery, multi_gpu_runtime="configured"))
    assert lmstudio.multi_gpu_config(configured) is configured


def test_multi_gpu_runtime_is_validated(tmp_path):
    text = EXAMPLE.read_text(encoding="utf-8").replace('multi_gpu_runtime = "lmstudio"', 'multi_gpu_runtime = "vulkan"')
    bad = tmp_path / "bad.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit, match="multi_gpu_runtime"):
        load_config(bad)


def test_start_error_quotes_the_servers_own_message(tmp_path):
    process = _process(tmp_path, NEW_BUILD)
    process.log_path = tmp_path / "server.log"
    process.log_path.write_text(
        "\n=== 2026-10-17 :: old launch ===\nold noise\n"
        "\n=== 2026-10-17 :: llama-server.exe --no-mmap ===\nerror: invalid argument: --no-mmap\n",
        encoding="utf-8",
    )
    assert process._log_tail() == "error: invalid argument: --no-mmap"
