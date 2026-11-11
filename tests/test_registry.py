import hashlib

import pytest
import yaml

from bandscribe.models import registry as reg_mod
from bandscribe.models.registry import (
    HashMismatchError,
    RegistryError,
    load_registry,
    sha256_file,
    verify_file,
)


@pytest.fixture(scope="module")
def reg():
    return load_registry()


def test_registry_loads_and_has_required_models(reg):
    for mid in ("bs_roformer_sw", "scnet_xl_ihf", "htdemucs_6s", "subroformer", "mvsep_mega53",
                "muscriptor_medium", "basic_pitch", "adtof", "beat_this"):
        assert mid in reg.models


def test_every_model_has_source_license_and_vram_fields(reg):
    for m in reg.models.values():
        assert m.source_urls and all(u.startswith("https://") for u in m.source_urls)
        assert m.license_note or m.license_weights or m.license_code
        assert m.vram_note
        assert m.verification


def test_spec_defaults(reg):
    # Spec 12.2: Mega 53 disabled on 10 GB; MuScriptor medium is the default.
    assert reg.get("mvsep_mega53").status == "disabled"
    assert reg.get("muscriptor_medium").enabled_by_default
    assert not reg.get("muscriptor_large").enabled_by_default
    assert reg.get("htdemucs_6s").status == "baseline"
    assert reg.get("bs_roformer_sw").enabled_by_default


def test_guitar_stem_only_from_six_stem_models(reg):
    # Spec 2.2 / 5.1: 4-stem models have no guitar stem; `other` is never guitar.
    for m in reg.by_task("separation"):
        if "guitar" in m.stems:
            assert "other" in m.stems and m.id in {"bs_roformer_sw", "htdemucs_6s"}
    assert "guitar" not in reg.get("scnet_xl_ihf").stems
    assert "guitar" not in reg.get("subroformer").stems


def test_non_commercial_weights_flagged(reg):
    assert reg.get("muscriptor_medium").non_commercial
    assert reg.get("adtof").non_commercial
    assert not reg.get("htdemucs_6s").non_commercial


def test_unknown_model_id(reg):
    with pytest.raises(KeyError, match="unknown model id"):
        reg.get("nope")


def _write(tmp_path, mutate):
    raw = yaml.safe_load(reg_mod.REGISTRY_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(schema_version=2),
    lambda r: r["models"]["adtof"].update(task="cooking"),
    lambda r: r["models"]["adtof"].update(status="maybe"),
    lambda r: r["models"]["adtof"].update(source_urls=[]),
    lambda r: r["models"]["adtof"].update(source_urls=["http://insecure.example/x"]),
    lambda r: r["models"]["adtof"].pop("license"),
    lambda r: r["models"]["adtof"].update(verification=["TRUST-ME"]),
    lambda r: r["models"]["scnet_xl_ihf"]["files"][0].update(sha256="abc"),
    lambda r: r["models"]["scnet_xl_ihf"]["files"][0].update(url="ftp://x/y.ckpt"),
])
def test_malformed_registry_rejected(tmp_path, mutate):
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, mutate))


def test_verify_file(tmp_path):
    p = tmp_path / "w.ckpt"
    p.write_bytes(b"weights" * 1000)
    digest = hashlib.sha256(b"weights" * 1000).hexdigest()
    assert sha256_file(p, chunk=7) == digest
    assert verify_file(p, digest) == digest
    assert verify_file(p, None) == digest  # unpinned: hash reported, nothing asserted
    with pytest.raises(HashMismatchError):
        verify_file(p, "0" * 64)


def test_unpinned_files_listed(tmp_path):
    def unpin(raw):
        raw["models"]["scnet_xl_ihf"]["files"][0]["sha256"] = None

    reg = load_registry(_write(tmp_path, unpin))
    assert ("scnet_xl_ihf", "model_scnet_ep_36_sdr_10.0891.ckpt") in reg.unpinned_files()


def test_pinned_hashes_are_wellformed(reg):
    for m in reg.models.values():
        for f in m.files:
            assert f.url and f.url.startswith("https://")
            assert f.sha256 is None or len(f.sha256) == 64
