import json

from typer.testing import CliRunner

from bandscribe import env_check
from bandscribe.cli import app
from bandscribe.env_check import Check


def which_map(mapping):
    return lambda name: mapping.get(name)


def test_gpu_parsed_from_nvidia_smi():
    c = env_check.check_nvidia_smi(
        which_map({"nvidia-smi": "smi"}),
        lambda cmd: (0, "NVIDIA GeForce RTX 3080, 10240, 812, 610.88\n"))
    assert c.status == "ok"
    assert c.details["gpus"][0] == {"name": "NVIDIA GeForce RTX 3080", "vram_total_gb": 10.0,
                                    "vram_used_gb": 0.79, "driver": "610.88"}


def test_small_gpu_warns():
    c = env_check.check_nvidia_smi(which_map({"nvidia-smi": "smi"}),
                                   lambda cmd: (0, "GTX 1060, 6144, 100, 560.1\n"))
    assert c.status == "warn" and "10 GB" in c.hint


def test_no_nvidia_driver_is_warning_not_failure():
    # Spec 4.1: CPU is allowed with a warning.
    assert env_check.check_nvidia_smi(which_map({}), lambda cmd: (1, "")).status == "warn"


def test_nvidia_smi_garbage_output():
    c = env_check.check_nvidia_smi(which_map({"nvidia-smi": "smi"}), lambda cmd: (0, "???\n"))
    assert c.status == "warn"


def test_node_ok_old_and_missing():
    w = which_map({"node": "node", "npm": "npm"})
    ok = env_check.check_node(w, lambda cmd: (0, "v22.14.0\n" if cmd[0] == "node" else "10.9.2\n"))
    assert ok.status == "ok" and ok.details["npm"] == "10.9.2"
    old = env_check.check_node(w, lambda cmd: (0, "v16.20.0\n" if cmd[0] == "node" else "8.0.0\n"))
    assert old.status == "fail"
    assert env_check.check_node(which_map({}), lambda cmd: (127, "")).status == "fail"
    no_npm = env_check.check_node(which_map({"node": "node"}), lambda cmd: (0, "v22.14.0\n"))
    assert no_npm.status == "warn"


def test_ffmpeg_from_path():
    c = env_check.check_ffmpeg(which_map({"ffmpeg": "ff"}),
                               lambda cmd: (0, "ffmpeg version 7.1-full Copyright\n"))
    assert c.status == "ok" and "7.1-full" in c.summary and c.details["origin"] == "PATH"


def test_ffmpeg_missing_everywhere(monkeypatch):
    import imageio_ffmpeg

    def boom():
        raise RuntimeError("no exe")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", boom)
    assert env_check.check_ffmpeg(which_map({}), lambda cmd: (127, "")).status == "fail"


def test_alphatab_missing(tmp_path):
    c = env_check.check_alphatab(tmp_path, which_map({}))
    assert c.status == "warn" and "npm install" in c.hint


def test_alphatab_gp7_exporter_probe(tmp_path):
    pkg = tmp_path / "node_modules" / "@coderline" / "alphatab"
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(json.dumps({"version": "1.0.0", "license": "MPL-2.0"}))
    bad = env_check.check_alphatab(tmp_path, which_map({"node": "node"}),
                                   lambda cmd: (0, "GP7_MISSING\n"))
    assert bad.status == "fail"
    ok = env_check.check_alphatab(tmp_path, which_map({"node": "node"}),
                                  lambda cmd: (0, "GP7_OK\n"))
    assert ok.status == "ok" and ok.details["gp7_exporter"] is True


def test_musescore_env_override(tmp_path, monkeypatch):
    exe = tmp_path / "MuseScore4.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("BANDSCRIBE_MUSESCORE", str(exe))
    assert env_check.check_musescore(which_map({})).status == "ok"
    monkeypatch.setenv("BANDSCRIBE_MUSESCORE", str(tmp_path / "missing.exe"))
    assert env_check.check_musescore(which_map({})).status == "warn"


def test_registry_check_reports_unpinned_and_unbenched():
    c = env_check.check_registry()
    assert c.status == "ok"
    assert "muscriptor_medium" in c.details["non_commercial"]
    assert "bs_roformer_sw" in c.details["vram_not_benchmarked"]


def test_crashing_check_becomes_fail_entry():
    def check_explodes():
        raise ZeroDivisionError("x")

    (c,) = env_check.run_checks((check_explodes,))
    assert (c.name, c.status) == ("explodes", "fail") and "ZeroDivisionError" in c.summary


def test_overall_status_and_format():
    checks = [Check("a", "ok", "fine"), Check("b", "warn", "meh", hint="do x")]
    assert env_check.overall_status(checks) == "warn"
    assert env_check.overall_status(checks + [Check("c", "fail", "bad")]) == "fail"
    text = env_check.format_report(checks)
    assert "[ OK ] a" in text and "[WARN] b" in text and "-> do x" in text
    assert "Overall: WARN (1 ok, 1 warn, 0 fail)" in text


def test_doctor_cli_covers_every_required_area():
    result = CliRunner().invoke(app, ["doctor", "--json"])
    report = json.loads(result.stdout)
    names = {c["name"] for c in report["checks"]}
    assert {"os", "python", "uv", "gpu", "torch", "ffmpeg", "node", "alphatab", "musescore",
            "registry", "disk"} <= names
    assert report["overall"] in {"ok", "warn", "fail"}
    assert result.exit_code == (1 if report["overall"] == "fail" else 0)


def test_doctor_exit_code_on_failure(monkeypatch):
    monkeypatch.setattr(env_check, "run_checks", lambda: [Check("x", "fail", "broken")])
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1 and "[FAIL] x" in result.stdout


def test_models_list_cli():
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "mvsep_mega53" in result.stdout and "disabled" in result.stdout
    assert "CC BY-NC 4.0" in result.stdout
