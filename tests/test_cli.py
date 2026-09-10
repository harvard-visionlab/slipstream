"""Tests for the ``slipstream`` command-line interface.

All tests here run offline: the lab-dataset registry (visionlab-datasets) is
replaced with a fake, and S3 calls are patched. Real S3 checks are marked
``@pytest.mark.s3``.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from slipstream import cli
from slipstream.cache import MANIFEST_FILE
from slipstream.utils.cache_dir import CACHE_DIR_ENV_VAR

REMOTE_BASE = "s3://test-bucket/slipstream-cache"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class _Platform:
    value = "cpu_workstation"


def make_fake_registry(cache_dir: Path):
    """A stand-in for the ``visionlab.datasets`` module."""
    configs = {
        "tiny": types.SimpleNamespace(
            name="tiny",
            num_classes=3,
            metadata={"num_val": 4, "num_train": 100},
            remote_cache={
                ("val", "jpeg"): f"{REMOTE_BASE}/tiny/tiny-jpeg-val",
                ("val", "yuv420"): f"{REMOTE_BASE}/tiny/tiny-yuv420-val",
                ("train", "jpeg"): f"{REMOTE_BASE}/tiny/tiny-jpeg-train",
            },
        ),
        "other": types.SimpleNamespace(
            name="other",
            num_classes=2,
            remote_cache={("val", "jpeg"): f"{REMOTE_BASE}/other/other-jpeg-val"},
        ),
        "empty": types.SimpleNamespace(name="empty", num_classes=5, remote_cache={}),
    }
    vd = types.SimpleNamespace(
        __version__="9.9.9",
        list_datasets=lambda: sorted(configs),
        get_config=lambda name: configs[name],
        detect_platform=lambda: _Platform(),
        get_platform_cache_dir=lambda plat=None: str(cache_dir),
    )
    return vd


def write_cache(path: Path, *, complete: bool = True, n: int = 4) -> None:
    """Write a minimal but valid slipcache (label + index fields only)."""
    path.mkdir(parents=True, exist_ok=True)
    label = np.arange(n, dtype=np.int64)
    np.save(path / "label.npy", label)
    np.save(path / "index.npy", label)
    manifest = {
        "version": 1,
        "num_samples": n,
        "fields": {
            "label": {"type": "int", "num_samples": n},
            "index": {"type": "int", "num_samples": n},
        },
        "file_sizes": {
            "label.npy": (path / "label.npy").stat().st_size,
            "index.npy": (path / "index.npy").stat().st_size,
        },
    }
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    if not complete:
        (path / "index.npy").unlink()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    d.mkdir()
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(d))
    return d


@pytest.fixture
def fake_registry(cache_dir):
    vd = make_fake_registry(cache_dir)
    with patch.object(cli, "_import_registry", return_value=vd):
        yield vd


@pytest.fixture
def no_registry():
    with patch.object(cli, "_import_registry", return_value=None):
        yield


def _run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_status_without_registry(self, cache_dir, no_registry, capsys):
        write_cache(cache_dir / "slipcache-abc12345")
        code, out = _run(["status", "--no-remote", "--json"], capsys)
        data = json.loads(out)
        assert data["cache"]["path"] == str(cache_dir)
        assert CACHE_DIR_ENV_VAR in data["cache"]["source"]
        assert data["cache"]["access"]["readable"] is True
        assert data["cache"]["access"]["writable"] is True
        assert data["visionlab_datasets_version"] is None
        assert data["datasets"] == []
        assert [o["name"] for o in data["other_caches"]] == ["slipcache-abc12345"]
        # Missing registry is reported but is not a hard failure.
        assert code == 0

    def test_status_local_dataset_states(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-jpeg-val")
        write_cache(cache_dir / "tiny-yuv420-val", complete=False)
        # tiny-jpeg-train and other-jpeg-val are missing.
        code, out = _run(["status", "--no-remote", "--json"], capsys)
        data = json.loads(out)
        by_name = {e["cache_name"]: e for e in data["datasets"]}
        assert by_name["tiny-jpeg-val"]["local_status"] == "ok"
        assert by_name["tiny-jpeg-val"]["local_bytes"] > 0
        assert by_name["tiny-yuv420-val"]["local_status"] == "incomplete"
        assert any("index.npy" in p for p in by_name["tiny-yuv420-val"]["local_problems"])
        assert by_name["tiny-jpeg-train"]["local_status"] == "missing"
        assert by_name["other-jpeg-val"]["local_status"] == "missing"
        assert all(e["remote_status"] == "unchecked" for e in data["datasets"])
        assert by_name["tiny-jpeg-val"]["local_path"] == str(cache_dir / "tiny-jpeg-val")
        assert data["visionlab_datasets_version"] == "9.9.9"

    def test_status_text_output(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-jpeg-val")
        code, out = _run(["status", "--no-remote"], capsys)
        assert "Cache directory" in out
        assert str(cache_dir) in out
        assert "Lab datasets" in out
        assert "tiny-jpeg-val" in out
        assert "to fetch:   slipstream sync " in out  # hint for a missing cache
        assert "--split val --fmt jpeg" in out

    def test_sample_count_cross_check(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-jpeg-val", n=4)  # matches num_val=4
        write_cache(cache_dir / "tiny-yuv420-val", n=3)  # mismatch
        code, out = _run(["status", "--no-remote", "--json"], capsys)
        data = json.loads(out)
        by_name = {e["cache_name"]: e for e in data["datasets"]}
        assert by_name["tiny-jpeg-val"]["local_status"] == "ok"
        assert by_name["tiny-jpeg-val"]["local_problems"] == []
        assert by_name["tiny-jpeg-val"]["expected_samples"] == 4
        assert by_name["tiny-yuv420-val"]["local_status"] == "ok"  # files intact; count is a warning
        assert any("sample count 3 != registry num_val 4" in m for m in by_name["tiny-yuv420-val"]["local_problems"])
        assert data["datasets_without_caches"] == ["empty"]
        code, out = _run(["status", "--no-remote"], capsys)
        assert "sample count 3 != registry num_val 4" in out
        assert "registered but no remote caches yet: empty" in out

    def test_config_is_alias(self, cache_dir, fake_registry, capsys):
        code, out = _run(["config", "--no-remote"], capsys)
        assert "Cache directory" in out

    def test_status_remote_checks(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-jpeg-val")

        def fake_listing(remote, **kw):
            if remote.endswith("tiny-jpeg-train/"):
                return 0, 0
            if remote.endswith("other-jpeg-val/"):
                raise RuntimeError("An error occurred (AccessDenied)")
            return 5, 12345

        s3info = cli.S3Info(
            s5cmd_path="/usr/bin/s5cmd",
            credentials_found=True,
            identity_arn="arn:aws:iam::1:user/x",
            bucket_url=REMOTE_BASE + "/",
            bucket_readable=True,
        )
        with (
            patch.object(cli, "remote_listing", side_effect=fake_listing),
            patch.object(cli, "check_s3", return_value=s3info),
        ):
            code, out = _run(["status", "--json"], capsys)
        data = json.loads(out)
        by_name = {e["cache_name"]: e for e in data["datasets"]}
        assert by_name["tiny-jpeg-val"]["remote_status"] == "ok"
        assert by_name["tiny-jpeg-val"]["remote_bytes"] == 12345
        assert by_name["tiny-jpeg-train"]["remote_status"] == "missing"
        assert by_name["other-jpeg-val"]["remote_status"] == "denied"
        assert code == 0

    def test_status_unreadable_cache_dir_is_hard_failure(
        self, tmp_path, monkeypatch, no_registry, capsys
    ):
        if os.geteuid() == 0:
            pytest.skip("root ignores permission bits")
        d = tmp_path / "locked"
        d.mkdir()
        d.chmod(0o000)
        monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(d))
        try:
            code, out = _run(["status", "--no-remote"], capsys)
        finally:
            d.chmod(0o755)
        assert code == 1
        assert "No read access" in out

    def test_status_missing_cache_dir(self, tmp_path, monkeypatch, no_registry, capsys):
        d = tmp_path / "not-yet"
        monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(d))
        code, out = _run(["status", "--no-remote", "--json"], capsys)
        data = json.loads(out)
        assert data["cache"]["access"]["exists"] is False
        assert data["cache"]["access"]["can_create"] is True
        assert code == 0


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


class TestDatasets:
    def test_lists_registry(self, cache_dir, fake_registry, capsys):
        code, out = _run(["datasets"], capsys)
        assert code == 0
        assert "tiny  (3 classes)" in out
        assert "empty  (5 classes)\n  (no remote caches registered)" in out
        assert f"{REMOTE_BASE}/tiny/tiny-jpeg-train" in out

    def test_without_registry(self, no_registry, capsys):
        code, out = _run(["datasets"], capsys)
        assert code == 1
        assert "not installed" in out


# --------------------------------------------------------------------------- #
# sync
# --------------------------------------------------------------------------- #


class TestSync:
    def test_dry_run_downloads_nothing(self, cache_dir, fake_registry, capsys):
        with (
            patch.object(cli, "remote_listing", return_value=(5, 1000)),
            patch("slipstream.s3_sync.download_s3_cache") as dl,
        ):
            code, out = _run(["sync", "tiny", "--dry-run"], capsys)
        assert code == 0
        dl.assert_not_called()
        # default fmt=jpeg, split=all -> val + train
        assert "tiny-jpeg-val" in out and "tiny-jpeg-train" in out
        assert "tiny-yuv420-val" not in out
        assert "[dry-run]" in out

    def test_sync_downloads_missing_and_skips_present(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-jpeg-val")

        def fake_download(remote, local, **kw):
            write_cache(Path(local))
            return True

        with (
            patch.object(cli, "remote_listing", return_value=(5, 1000)),
            patch("slipstream.s3_sync.download_s3_cache", side_effect=fake_download) as dl,
        ):
            code, out = _run(["sync", "tiny", "--split", "all", "--fmt", "jpeg"], capsys)
        assert code == 0
        assert dl.call_count == 1
        remote, local = dl.call_args[0]
        assert remote == f"{REMOTE_BASE}/tiny/tiny-jpeg-train/"
        assert Path(local) == cache_dir / "tiny-jpeg-train"
        assert (cache_dir / "tiny-jpeg-train" / MANIFEST_FILE).exists()
        assert "already present" in out

    def test_sync_force_redownloads(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-yuv420-val")
        with (
            patch.object(cli, "remote_listing", return_value=(5, 1000)),
            patch("slipstream.s3_sync.download_s3_cache", return_value=True) as dl,
        ):
            code, out = _run(
                ["sync", "tiny", "--split", "val", "--fmt", "yuv420", "--force"], capsys
            )
        assert code == 0
        assert dl.call_count == 1

    def test_sync_incomplete_is_redownloaded(self, cache_dir, fake_registry, capsys):
        write_cache(cache_dir / "tiny-jpeg-val", complete=False)

        def fake_download(remote, local, **kw):
            write_cache(Path(local))
            return True

        with (
            patch.object(cli, "remote_listing", return_value=(5, 1000)),
            patch("slipstream.s3_sync.download_s3_cache", side_effect=fake_download) as dl,
        ):
            code, out = _run(["sync", "tiny", "--split", "val"], capsys)
        assert code == 0
        assert dl.call_count == 1
        assert "incomplete locally" in out

    def test_sync_s3_url_target(self, cache_dir, no_registry, capsys):
        url = f"{REMOTE_BASE}/tiny/tiny-jpeg-val"
        with (
            patch.object(cli, "remote_listing", return_value=(5, 1000)),
            patch("slipstream.s3_sync.download_s3_cache", return_value=True) as dl,
        ):
            code, out = _run(["sync", url, "--dest", str(cache_dir / "alt")], capsys)
        # download succeeded but no manifest was written -> integrity failure
        assert code == 1
        remote, local = dl.call_args[0]
        assert remote == url + "/"
        assert Path(local) == cache_dir / "alt" / "tiny-jpeg-val"

    def test_sync_remote_missing_fails(self, cache_dir, fake_registry, capsys):
        with (
            patch.object(cli, "remote_listing", return_value=(0, 0)),
            patch("slipstream.s3_sync.download_s3_cache") as dl,
        ):
            code, out = _run(["sync", "tiny", "--split", "val"], capsys)
        assert code == 1
        dl.assert_not_called()
        assert "remote missing" in out

    def test_sync_unknown_dataset(self, cache_dir, fake_registry, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["sync", "nope"])
        assert "Unknown dataset" in str(exc.value)

    def test_sync_bad_split(self, cache_dir, fake_registry, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["sync", "other", "--split", "train"])
        assert "no cache for" in str(exc.value)

    def test_sync_name_without_registry(self, cache_dir, no_registry, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["sync", "tiny"])
        assert "visionlab-datasets is not installed" in str(exc.value)

    def test_sync_insufficient_disk(self, cache_dir, fake_registry, capsys):
        huge = 10**18
        with (
            patch.object(cli, "remote_listing", return_value=(5, huge)),
            patch("slipstream.s3_sync.download_s3_cache") as dl,
        ):
            code, out = _run(["sync", "tiny", "--split", "val"], capsys)
        assert code == 1
        dl.assert_not_called()
        assert "not enough free disk space" in out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_fmt_bytes():
    assert cli._fmt_bytes(0) == "0 B"
    assert cli._fmt_bytes(1536) == "1.5 KB"
    assert cli._fmt_bytes(17_089_829_376) == "15.9 GB"
    assert cli._fmt_bytes(None) == "?"


def test_mask_account():
    assert cli._mask_account("arn:aws:iam::777749968893:user/alvarez") == "arn:aws:iam::********8893:user/alvarez"
    assert cli._mask_account("arn:aws:sts::123456789012:assumed-role/r/s") == "arn:aws:sts::********9012:assumed-role/r/s"
    assert cli._mask_account("no-account-here") == "no-account-here"


def test_colorize_only_when_enabled(monkeypatch):
    monkeypatch.setattr(cli, "_COLOR", False)
    assert cli._colorize(f"  read  {cli.OK}") == f"  read  {cli.OK}"
    monkeypatch.setattr(cli, "_COLOR", True)
    out = cli._colorize(f"  {cli.OK} ok  {cli.BAD} missing  {cli.WARN} incomplete")
    assert "\033[32m" + cli.OK in out
    assert "\033[31m" + cli.BAD in out
    assert "\033[33m" + cli.WARN in out
    assert "\033[31mmissing" in out


def test_no_color_when_not_tty(cache_dir, no_registry, capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    code, out = _run(["status", "--no-remote"], capsys)
    assert "\033[" not in out
    monkeypatch.setenv("FORCE_COLOR", "1")
    code, out = _run(["status", "--no-remote"], capsys)
    assert "\033[32m" in out
    code, out = _run(["status", "--no-remote", "--no-color"], capsys)
    assert "\033[" not in out


def test_split_s3():
    assert cli._split_s3("s3://bucket/a/b/") == ("bucket", "a/b/")
    assert cli._split_s3("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        cli._split_s3("/local/path")


def test_python_dash_m_entry():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "slipstream", "--version"], capture_output=True, text=True
    )
    assert out.returncode == 0
    assert "slipstream" in out.stdout


# --------------------------------------------------------------------------- #
# real S3 (opt-in)
# --------------------------------------------------------------------------- #


@pytest.mark.s3
def test_real_s3_read_check():
    info = cli.check_s3(cli.DEFAULT_REMOTE_CACHE_BASE)
    assert info.credentials_found, info
    assert info.bucket_readable, info
