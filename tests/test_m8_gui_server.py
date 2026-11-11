import io
import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from bandscribe.gui.server import HOST, create_app
from bandscribe.pipeline import STAGES, PipelineCancelled

SR = 44100


def wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    t = np.arange(int(seconds * SR)) / SR
    sf.write(buf, np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1), SR, format="WAV")
    return buf.getvalue()


class FakeRunner:
    """Stands in for run_pipeline: reports progress, writes a minimal run directory."""

    def __init__(self):
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail = False

    def __call__(self, audio_path, config, out_dir, progress, cancel):
        self.calls.append((audio_path, config))
        for stage in STAGES:
            progress(stage, 0.0, f"{stage}...")
            self.gate.wait(5)
            if cancel():
                raise PipelineCancelled("cancelled")
            progress(stage, 1.0, "")
        if self.fail:
            raise RuntimeError("model exploded")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "exports").mkdir(exist_ok=True)
        (out_dir / "exports" / "song.gp").write_bytes(b"gp-bytes")
        (out_dir / "manifest.json").write_text(json.dumps({"status": "done", "warnings": []}))
        return out_dir


@pytest.fixture
def client(tmp_path):
    runner = FakeRunner()
    app = create_app(tmp_path / "projects", runner, tmp_path / "no-dist")
    with TestClient(app) as c:
        c.runner = runner
        yield c


def upload(client, name="song.wav"):
    r = client.post("/api/projects", files={"file": (name, wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def wait_done(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_server_binds_to_loopback_only():
    assert HOST == "127.0.0.1"


def test_info_lists_models_licenses_and_tunings(client):
    info = client.get("/api/info").json()
    by_id = {m["id"]: m for m in info["models"]}
    assert by_id["muscriptor_medium"]["license_weights"] == "CC BY-NC 4.0"
    assert by_id["muscriptor_medium"]["non_commercial"] and by_id["adtof"]["non_commercial"]
    assert "commercial" in info["notice"]
    assert any(t["key"] == "guitar_drop_d" for t in info["tunings"])
    assert info["defaults"]["quality"]["separator_primary"] == "bs_roformer_sw"
    assert info["defaults"]["balanced"]["separator_secondary"] is None


def test_upload_returns_waveform_and_rejects_bad_files(client):
    meta = upload(client)
    assert meta["name"] == "song" and meta["waveform"]["duration"] == pytest.approx(1.0, abs=0.01)
    assert len(meta["waveform"]["min"]) == len(meta["waveform"]["max"]) > 100
    assert client.post("/api/projects", files={"file": ("x.ogg", b"abc", "audio/ogg")}).status_code == 400
    assert client.post("/api/projects", files={"file": ("x.wav", b"", "audio/wav")}).status_code == 400
    assert client.post("/api/projects", files={"file": ("x.wav", b"not audio", "audio/wav")}).status_code == 400
    assert [p["id"] for p in client.get("/api/projects").json()][0] == meta["id"]


def test_run_job_progress_and_result(client):
    pid = upload(client)["id"]
    job = client.post(f"/api/projects/{pid}/run",
                      json={"config": {"profile": "balanced", "strict": True,
                                       "guitar_tuning": "guitar_drop_d"}}).json()
    assert job["status"] in ("queued", "running") and job["stages"] == list(STAGES)
    final = wait_done(client, job["id"])
    assert final["status"] == "done" and final["overall"] == 1.0
    _audio, cfg = client.runner.calls[0]
    assert cfg.profile == "balanced" and cfg.strict and cfg.guitar_tuning == "guitar_drop_d"
    assert cfg.separator_secondary is None  # the profile came from defaults.yaml
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["manifest"]["status"] == "done" and proj["exports"] == ["song.gp"]
    assert proj["config"]["strict"] is True and proj["job"] is None
    assert client.get(f"/api/projects/{pid}/files/run/exports/song.gp").content == b"gp-bytes"


def test_progress_is_monotonic_over_websocket(client):
    pid = upload(client)["id"]
    client.runner.gate.clear()
    job = client.post(f"/api/projects/{pid}/run", json={"config": {}}).json()
    seen = []
    with client.websocket_connect(f"/api/ws/jobs/{job['id']}") as ws:
        client.runner.gate.set()
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg["status"] in ("done", "failed", "cancelled"):
                break
    overall = [m["overall"] for m in seen]
    assert overall == sorted(overall) and seen[-1]["status"] == "done"


def test_cancel(client):
    pid = upload(client)["id"]
    client.runner.gate.clear()
    job = client.post(f"/api/projects/{pid}/run", json={"config": {}}).json()
    assert client.post(f"/api/projects/{pid}/run", json={"config": {}}).status_code == 409
    client.post(f"/api/jobs/{job['id']}/cancel")
    client.runner.gate.set()
    assert wait_done(client, job["id"])["status"] == "cancelled"


def test_failure_is_reported(client):
    pid = upload(client)["id"]
    client.runner.fail = True
    job = client.post(f"/api/projects/{pid}/run", json={"config": {}}).json()
    final = wait_done(client, job["id"])
    assert final["status"] == "failed" and "model exploded" in final["error"]


def test_invalid_config_rejected_before_queueing(client):
    pid = upload(client)["id"]
    assert client.post(f"/api/projects/{pid}/run", json={"config": {"reuse_cache": False}}).status_code == 400
    assert client.post(f"/api/projects/{pid}/run", json={"config": {"profile": "turbo"}}).status_code == 400
    assert client.post(f"/api/projects/{pid}/run", json={"config": {"instruments": ["banjo"]}}).status_code == 400
    assert client.runner.calls == []


def test_stem_override_is_passed_to_the_pipeline(client):
    pid = upload(client)["id"]
    r = client.post(f"/api/projects/{pid}/stems/guitar/override", json={"state": "absent"})
    assert r.json()["stem_overrides"] == {"guitar": "absent"}
    saved = client.get(f"/api/projects/{pid}").json()["config"]["stem_overrides"]
    job = client.post(f"/api/projects/{pid}/run", json={"config": {"stem_overrides": saved}}).json()
    wait_done(client, job["id"])
    assert client.runner.calls[-1][1].stem_overrides == {"guitar": "absent"}
    assert client.post(f"/api/projects/{pid}/stems/guitar/override", json={"state": None}).json() == {"stem_overrides": {}}
    assert client.post(f"/api/projects/{pid}/stems/guitar/override", json={"state": "maybe"}).status_code == 400


def test_file_access_is_confined_to_the_project(client, tmp_path):
    pid = upload(client)["id"]
    (tmp_path / "secret.txt").write_text("x")
    for path in ("../secret.txt", "..%2Fsecret.txt", "../../secret.txt", "run/missing.gp"):
        assert client.get(f"/api/projects/{pid}/files/{path}").status_code == 404
    assert client.get("/api/projects/..%2F..%2Fetc/files/x").status_code == 404
    assert client.get("/api/projects/deadbeef/files/x").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_missing_frontend_gives_a_clear_message(client):
    r = client.get("/")
    assert r.status_code == 503 and "npm run build" in r.json()["detail"]
