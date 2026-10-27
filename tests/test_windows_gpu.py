"""Windows GPU telemetry (ADL + Windows counters), exercised with a fake ADL library and canned PowerShell output."""
from __future__ import annotations

import ctypes
import time
from types import SimpleNamespace

import pytest

from dual_gpu_setup.training import windows_gpu as wg
from dual_gpu_setup.training.hardware import HardwareSampler, discover_gpus, sample_gpu
from dual_gpu_setup.training.windows_gpu import (
    AdapterInfo, AdlAdapter, AdlLibrary, AdlUnavailable, WindowsGpuReader, parse_link_lines, parse_memory_lines, speed_label)

R9700_LUID = "0x00000000_0x00019cb5"
XT_LUID = "0x00000000_0x000158fb"
GB = 1024**3

DXGI = [
    {"name": "AMD Radeon RX 9070 XT", "luid": XT_LUID, "dedicated_bytes": int(15.8 * GB)},
    {"name": "AMD Radeon(TM) Graphics", "luid": "0x00000000_0x0001ce67", "dedicated_bytes": 2 * GB},
    {"name": "AMD Radeon AI PRO R9700", "luid": R9700_LUID, "dedicated_bytes": int(31.7 * GB)},
    {"name": "Microsoft Basic Render Driver", "luid": "0x00000000_0x00019c73", "dedicated_bytes": 0},
]
MEMORY_OUTPUT = f"""mem|dedicated usage|luid_{XT_LUID}_phys_0|{14 * GB}
mem|shared usage|luid_{XT_LUID}_phys_0|{2 * GB}
mem|dedicated usage|luid_{R9700_LUID}_phys_0|{25 * GB}
mem|shared usage|luid_{R9700_LUID}_phys_0|{GB}
mem|dedicated usage|luid_0x00000000_0x00019c73_phys_0|0
garbage line
"""
LINK_OUTPUT = """link|PCI bus 3, device 0, function 0|5|16|5|16|32.0.31036.15
link|PCI bus 7, device 0, function 0|4|8|5|16|32.0.31036.15
link|not a pci location|5|16|5|16|x
"""
# Sensor ids -> readings, as ADL reports them for an RDNA4 card.
R9700_SENSORS = {1: 2517, 2: 2505, 8: 78, 9: 96, 14: 5000, 19: 95, 27: 94, 73: 299, 40: 4, 41: 8}
XT_SENSORS = {1: 3011, 2: 2505, 8: 57, 9: 90, 14: 3623, 19: 91, 27: 84, 73: 304}


class FakeAdl:
    def __init__(self, sensors=None, adapters=None):
        self.sensor_map = sensors if sensors is not None else {10: R9700_SENSORS, 5: XT_SENSORS}
        self.adapter_list = adapters if adapters is not None else [
            AdlAdapter(0, "AMD Radeon(TM) Graphics", 127, 0, 0),
            AdlAdapter(5, "AMD Radeon RX 9070 XT", 7, 0, 0),
            AdlAdapter(10, "AMD Radeon AI PRO R9700", 3, 0, 0),
        ]
        self.closed = False

    def adapters(self):
        return self.adapter_list

    def sensors(self, index):
        return self.sensor_map.get(index)

    def close(self):
        self.closed = True


def reader(adl=None, memory=MEMORY_OUTPUT, links=LINK_OUTPUT, dxgi=DXGI):
    def powershell(script):
        return memory if "Adapter Memory" in script else links

    return WindowsGpuReader(adl or FakeAdl(), lambda: dxgi, powershell)


# --- parsing -------------------------------------------------------------------------------------

def test_memory_lines_become_dedicated_and_shared_per_luid():
    usage = parse_memory_lines(MEMORY_OUTPUT)
    assert usage[R9700_LUID] == {"dedicated": 25 * GB, "shared": GB}
    assert usage[XT_LUID] == {"dedicated": 14 * GB, "shared": 2 * GB}
    assert usage["0x00000000_0x00019c73"] == {"dedicated": 0}


def test_link_lines_are_keyed_by_the_linux_style_pci_address():
    links = parse_link_lines(LINK_OUTPUT)
    assert links["0000:03:00.0"] == {"speed": 5, "width": 16, "max_speed": 5, "max_width": 16, "driver_version": "32.0.31036.15"}
    assert links["0000:07:00.0"]["speed"] == 4 and len(links) == 2  # the unparsable location is skipped


def test_pcie_generation_is_reported_in_the_linux_unit():
    assert speed_label(5) == "32.0 GT/s PCIe" and speed_label(4) == "16.0 GT/s PCIe"
    assert speed_label(None) is None and speed_label(99) is None


# --- discovery -----------------------------------------------------------------------------------

def test_discovery_matches_adl_cards_to_windows_adapters_by_name():
    gpus = reader().discover()
    by_name = {g["name"]: g for g in gpus}
    assert [g["pci_bus"] for g in gpus] == ["0000:03:00.0", "0000:07:00.0", "0000:7f:00.0"]
    assert by_name["AMD Radeon AI PRO R9700"]["luid"] == R9700_LUID and by_name["AMD Radeon AI PRO R9700"]["adl_index"] == 10
    assert by_name["AMD Radeon RX 9070 XT"]["vram_total_bytes"] == int(15.8 * GB)


def test_two_identical_cards_cannot_be_matched_by_name_so_memory_is_unavailable_not_guessed():
    twins = [{"name": "AMD Radeon RX 9070 XT", "luid": f"0x0_0x{i}", "dedicated_bytes": GB} for i in (1, 2)]
    adl = FakeAdl(adapters=[AdlAdapter(1, "AMD Radeon RX 9070 XT", 3, 0, 0), AdlAdapter(2, "AMD Radeon RX 9070 XT", 7, 0, 0)])
    gpus = reader(adl, dxgi=twins).discover()
    assert all(g["luid"] is None and g["vram_total_bytes"] is None for g in gpus)


# --- one sample ----------------------------------------------------------------------------------

def _gpu(r, name):
    return next(g for g in r.discover() if g["name"] == name)


def _sample(r, name):
    g = _gpu(r, name)
    return r.sample(g["adl_index"], g["pci_bus"], g["luid"], g["vram_total_bytes"])


def test_sensors_map_to_the_fields_the_dashboard_reads():
    r = reader()
    r.refresh_memory()
    r.refresh_links()
    s = _sample(r, "AMD Radeon AI PRO R9700")
    assert (s["gpu_busy_pct"], s["temp_edge_c"], s["temp_junction_c"], s["temp_memory_c"]) == (95, 78.0, 94.0, 96.0)
    assert (s["power_w"], s["sclk_mhz"], s["mclk_mhz"], s["fan_rpm"]) == (299.0, 2517.0, 2505.0, 5000)
    assert s["vram_used_bytes"] == 25 * GB and s["gtt_used_bytes"] == GB and s["vram_total_bytes"] == int(31.7 * GB)
    assert (s["pcie_speed"], s["pcie_width"], s["pcie_max_width"]) == ("32.0 GT/s PCIe", 16, 16)
    assert s["unavailable"] == {}


def test_sensor_ids_40_and_41_are_not_reported_as_the_pcie_link():
    """On this hardware they read 4 and 8 while Windows reports Gen5 x16: they are something else."""
    r = reader()
    r.refresh_links()
    s = _sample(r, "AMD Radeon AI PRO R9700")
    assert s["pcie_width"] == 16 and s["pcie_speed"] == "32.0 GT/s PCIe"


def test_unsupported_sensors_are_none_with_a_reason_never_zero():
    r = reader(FakeAdl(sensors={0: {1: 600, 2: 3200, 19: 0}}))
    r.refresh_memory()
    s = _sample(r, "AMD Radeon(TM) Graphics")
    assert s["gpu_busy_pct"] == 0  # a real zero is kept
    assert s["temp_edge_c"] is None and s["power_w"] is None and s["fan_rpm"] is None
    assert "sensor 8 not supported" in s["unavailable"]["temp_edge_c"] and "power_w" in s["unavailable"]


def test_a_failed_adl_read_reports_every_sensor_unavailable():
    r = reader(FakeAdl(sensors={}))  # sensors() returns None for the adapter
    s = _sample(r, "AMD Radeon AI PRO R9700")
    assert s["gpu_busy_pct"] is None and s["temp_edge_c"] is None
    assert "could not read" in s["unavailable"]["gpu_busy_pct"]


def test_slow_values_say_not_read_yet_until_the_background_read_lands():
    r = reader()
    s = _sample(r, "AMD Radeon AI PRO R9700")
    assert s["vram_used_bytes"] is None and "not read yet" in s["unavailable"]["vram_used_bytes"]
    assert s["pcie_speed"] is None and "not read yet" in s["unavailable"]["pcie_speed"]
    assert s["temp_edge_c"] == 78.0  # the fast sensors do not wait for it
    r.refresh_memory()
    assert _sample(r, "AMD Radeon AI PRO R9700")["vram_used_bytes"] == 25 * GB


def test_a_card_without_a_matching_windows_adapter_says_so():
    r = reader(dxgi=[])
    s = _sample(r, "AMD Radeon AI PRO R9700")
    assert s["vram_used_bytes"] is None and "no Windows adapter matched" in s["unavailable"]["vram_used_bytes"]


# --- through the sampler ---------------------------------------------------------------------------

LANES = [SimpleNamespace(key="r9700", display="R9700", match="R9700", vram_gb=32, pci_bus=""),
         SimpleNamespace(key="9070xt", display="9070 XT", match="9070 XT", vram_gb=16, pci_bus="")]


def test_sampler_on_windows_assigns_lanes_and_samples_every_card():
    r = reader()
    r.refresh_memory()
    r.refresh_links()
    sampler = HardwareSampler(LANES, platform_name="win32", windows_reader=r)
    assert sampler.unavailable_reason is None and {g.lane for g in sampler.gpus} == {"r9700", "9070xt", None}
    samples = sampler.sample_once()
    by_lane = {s["lane"]: s for s in samples}
    assert by_lane["r9700"]["power_w"] == 299.0 and by_lane["9070xt"]["temp_junction_c"] == 84.0
    snapshot = sampler.snapshot()
    assert snapshot["available"] is True and len(snapshot["gpus"]) == 3 and snapshot["sampler"]["read_errors"] == 0


def test_a_lane_can_be_pinned_by_pci_address_on_windows():
    lanes = [SimpleNamespace(key="big", display="x", match="nothing", vram_gb=32, pci_bus="0000:03:00.0")]
    sampler = HardwareSampler(lanes, platform_name="win32", windows_reader=reader())
    assert {g.pci_bus: g.lane for g in sampler.gpus}["0000:03:00.0"] == "big"


def test_environment_records_the_negotiated_link_and_driver():
    r = reader()
    r.refresh_links()
    env = HardwareSampler(LANES, platform_name="win32", windows_reader=r).hardware_environment()
    assert env["amdgpu_driver"] == "32.0.31036.15"
    r9700 = next(g for g in env["gpus"] if g["lane"] == "r9700")
    assert (r9700["pcie_speed"], r9700["pcie_width"], r9700["vbios_version"]) == ("32.0 GT/s PCIe", 16, None)


def test_sampler_start_and_stop_drive_the_reader_and_release_adl():
    adl = FakeAdl()
    r = reader(adl)
    sampler = HardwareSampler(LANES, platform_name="win32", windows_reader=r, interval_s=0.02)
    sampler.start()
    deadline = time.time() + 3
    while time.time() < deadline and not (sampler.ticks > 2 and r.link("0000:03:00.0")):
        time.sleep(0.02)
    sampler.stop()
    time.sleep(0.1)
    assert sampler.ticks > 2 and r.link("0000:03:00.0") is not None and adl.closed is True


def test_windows_without_a_reader_still_says_why_nothing_is_shown():
    sampler = HardwareSampler(platform_name="win32")
    assert sampler.snapshot()["available"] is False and "Linux" in sampler.snapshot()["unavailable_reason"]
    assert discover_gpus(platform_name="win32")[0] == []


def test_a_card_sampled_through_a_reader_does_not_touch_sysfs():
    gpu = discover_gpus(platform_name="win32", windows_reader=reader())[0][0]
    assert gpu.reader is not None and sample_gpu(gpu)["pci_bus"] == gpu.pci_bus


# --- the ADL wrapper itself ------------------------------------------------------------------------

class FakeDll:
    """Just enough of atiadlxx.dll: fills the ctypes buffers the way ADL does."""

    def __init__(self, create_code=0, pmlog_code=0):
        self.create_code, self.pmlog_code, self.destroyed = create_code, pmlog_code, False

    def ADL2_Main_Control_Create(self, callback, enum_connected, context_ref):
        context_ref._obj.value = 1234  # the real library hands back a context handle
        return self.create_code

    def ADL2_Adapter_NumberOfAdapters_Get(self, context, count_ref):
        count_ref._obj.value = 4
        return 0

    def ADL2_Adapter_AdapterInfo_Get(self, context, infos, size):
        rows = [(0, 1002, 3, "AMD Radeon AI PRO R9700", 1), (1, 1002, 3, "AMD Radeon AI PRO R9700", 1),
                (2, 1002, 7, "AMD Radeon RX 9070 XT", 1), (3, 4318, 1, "Other Vendor GPU", 1)]
        for info, (index, vendor, bus, name, present) in zip(infos, rows):
            info.iAdapterIndex, info.iVendorID, info.iBusNumber, info.iPresent = index, vendor, bus, present
            info.strAdapterName = name.encode()
        return 0

    def ADL2_New_QueryPMLogData_Get(self, context, index, out_ref):
        if self.pmlog_code:
            return self.pmlog_code
        sensors = out_ref._obj.sensors
        sensors[8].supported, sensors[8].value = 1, 77
        sensors[73].supported, sensors[73].value = 1, 300
        sensors[20].supported, sensors[20].value = 0, 55  # present in the array but not supported
        return 0

    def ADL2_Main_Control_Destroy(self, context):
        self.destroyed = True
        return 0


def test_adl_adapters_are_deduplicated_and_non_amd_cards_ignored():
    adapters = AdlLibrary(FakeDll()).adapters()
    assert [(a.name, a.pci_bus) for a in adapters] == [("AMD Radeon AI PRO R9700", "0000:03:00.0"),
                                                       ("AMD Radeon RX 9070 XT", "0000:07:00.0")]


def test_adl_sensors_return_only_supported_ones_and_none_on_error():
    assert AdlLibrary(FakeDll()).sensors(0) == {8: 77, 73: 300}
    assert AdlLibrary(FakeDll(pmlog_code=-1)).sensors(0) is None


def test_adl_close_destroys_the_context_once():
    dll = FakeDll()
    adl = AdlLibrary(dll)
    adl.close()
    adl.close()
    assert dll.destroyed is True


def test_adl_that_will_not_start_says_why():
    with pytest.raises(AdlUnavailable, match="returned -1"):
        AdlLibrary(FakeDll(create_code=-1))


def test_missing_adl_library_is_reported_not_raised_into_the_app(monkeypatch):
    monkeypatch.setattr(wg.sys, "platform", "win32")
    monkeypatch.setattr(wg.ctypes, "CDLL", lambda name: (_ for _ in ()).throw(OSError("not found")))
    reader_, why = WindowsGpuReader.try_create()
    assert reader_ is None and "atiadlxx.dll not found" in why


def test_struct_layout_matches_adls_adapter_info():
    """A wrong field order or type shifts every later field. 9 ints and six 256-byte strings = 1572 bytes
    (the size the real driver accepted on the reference machine); the PMLog buffer is a size int plus
    256 sensors of two ints."""
    assert ctypes.sizeof(AdapterInfo) == 9 * 4 + 6 * 256 == 1572
    assert ctypes.sizeof(wg.PMLogOutput) == 4 + 8 * wg.ADL_PMLOG_MAX_SENSORS


# --- a discovered GPU must not break measured memory forecasts -----------------------------------------

def test_measured_forecast_matches_the_gpu_a_finished_run_was_recorded_under(live_app):
    """finish_run stores the profile under the discovered PCI address; the forecast must look it up the
    same way when the lane config has no pci_bus. (Found when Windows GPUs started being discovered.)"""
    live = live_app()
    live.state.service.hardware = HardwareSampler(LANES, platform_name="win32", windows_reader=reader())
    config = {"max_seq_length": 2048, "per_device_train_batch_size": 1, "bf16": True, "total_params": 1e9,
              "trainable_params": 10e6, "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 16,
              "vocab_size": 32000}
    assert live.post("/api/training/forecast", {"lane": "9070xt", "config": config})[1]["source"] == "estimate"
    status, _ = live.post("/api/training/runs", {"id": "w", "name": "w", "config": config, "lanes": ["9070xt"]})
    assert status == 200
    rows = [{"step": i * 10, "window_steps": 10, "phase": "train", "loss": 2.0, "processed_tokens": 1200, "nonpad_tokens": 1000,
             "supervised_tokens": 400, "step_time_ms": 500, "mem_peak_bytes": 4 * GB} for i in range(1, 6)]
    assert live.post("/api/training/runs/w/metrics", {"steps": rows, "heartbeat": {"last_step": 50}})[0] == 200
    live.post("/api/training/runs/w/finish", {"status": "completed"})
    measured = live.post("/api/training/forecast", {"lane": "9070xt", "config": config})[1]
    assert measured["source"] == "measured" and measured["peak_bytes"] == 4 * GB
    assert measured["key"]["gpu"] == "0000:07:00.0"  # the discovered PCI address, not the lane key
