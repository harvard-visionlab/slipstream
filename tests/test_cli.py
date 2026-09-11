"""Tests for the ``slipstream`` command-line interface (plumbing checks).

All tests run offline; S3 calls are patched. Real S3 checks are marked
``@pytest.mark.s3``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from slipstream import cli
from slipstream.cache import MANIFEST_FILE
from slipstream.utils.cache_dir import CACHE_DIR_ENV_VAR


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


def _run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_status_json(self, cache_dir, capsys):
        write_cache(cache_dir / "slipcache-abc12345")
        (cache_dir / "not-a-cache").mkdir()
        code, out = _run(["status", "--no-remote", "--json"], capsys)
        data = json.loads(out)
        assert data["cache"]["path"] == str(cache_dir)
        assert CACHE_DIR_ENV_VAR in data["cache"]["source"]
        assert data["cache"]["access"]["readable"] is True
        assert data["cache"]["access"]["writable"] is True
        assert [o["name"] for o in data["caches"]] == ["slipcache-abc12345"]
        assert data["caches"][0]["bytes"] > 0
        assert data["s3"]["checked"] is False
        assert "datasets" not in data
        assert code == 0

    def test_status_text(self, cache_dir, capsys):
        write_cache(cache_dir / "slipcache-abc12345")
        code, out = _run(["status", "--no-remote"], capsys)
        assert "Cache directory" in out
        assert str(cache_dir) in out
        assert "Slipstream caches in cache dir" in out
        assert "slipcache-abc12345" in out
        assert "lab users" not in out  # env var is set
        assert "slipstream sync" not in out
        assert "visionlab-datasets not installed" not in out

    def test_default_cache_dir_hint(self, monkeypatch, capsys):
        monkeypatch.delenv(CACHE_DIR_ENV_VAR, raising=False)
        code, out = _run(["status", "--no-remote"], capsys)
        assert "slipstream default (~/.slipstream)" in out
        assert "visionlab-datasets status" in out

    def test_config_is_alias(self, cache_dir, capsys):
        code, out = _run(["config", "--no-remote"], capsys)
        assert "Cache directory" in out

    def test_status_with_remote(self, cache_dir, capsys):
        s3info = cli.S3Info(
            s5cmd_path="/usr/bin/s5cmd",
            s5cmd_version="v2.3.0",
            credentials_found=True,
            credentials_method="env",
            identity_arn="arn:aws:iam::********8893:user/x",
            bucket_url=cli.DEFAULT_REMOTE_CACHE_BASE,
            bucket_readable=True,
        )
        with patch.object(cli, "check_s3", return_value=s3info) as chk:
            code, out = _run(["status"], capsys)
        assert chk.call_args.kwargs["check_remote"] is True
        assert "arn:aws:iam::********8893:user/x" in out
        assert "Everything looks good" in out
        assert code == 0

    def test_status_no_bucket_access_is_hard_failure(self, cache_dir, capsys):
        s3info = cli.S3Info(
            s5cmd_path="/usr/bin/s5cmd",
            credentials_found=True,
            identity_arn="arn:aws:iam::********0000:user/x",
            bucket_url=cli.DEFAULT_REMOTE_CACHE_BASE,
            bucket_readable=False,
            bucket_error="AccessDenied",
        )
        with patch.object(cli, "check_s3", return_value=s3info):
            code, out = _run(["status"], capsys)
        assert code == 1
        assert "Cannot list" in out

    def test_status_no_credentials(self, cache_dir, capsys):
        s3info = cli.S3Info(s5cmd_path="/usr/bin/s5cmd", credentials_found=False)
        with patch.object(cli, "check_s3", return_value=s3info):
            code, out = _run(["status"], capsys)
        assert code == 1
        assert "No AWS credentials found" in out

    def test_status_unreadable_cache_dir_is_hard_failure(self, tmp_path, monkeypatch, capsys):
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

    def test_status_missing_cache_dir(self, tmp_path, monkeypatch, capsys):
        d = tmp_path / "not-yet"
        monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(d))
        code, out = _run(["status", "--no-remote", "--json"], capsys)
        data = json.loads(out)
        assert data["cache"]["access"]["exists"] is False
        assert data["cache"]["access"]["can_create"] is True
        assert code == 0

    def test_no_sync_or_datasets_subcommands(self):
        for cmd in (["sync", "imagenet100"], ["datasets"]):
            with pytest.raises(SystemExit) as exc:
                cli.main(cmd)
            assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# public helpers (visionlab-datasets builds on these; keep signatures stable)
# --------------------------------------------------------------------------- #


def test_inspect_dir(tmp_path):
    info = cli.inspect_dir(tmp_path)
    assert info.exists and info.is_dir and info.readable and info.writable
    assert info.owner and info.mode and info.free_bytes
    missing = cli.inspect_dir(tmp_path / "a" / "b")
    assert not missing.exists and missing.can_create


def test_dir_bytes_recursive(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert cli.dir_bytes(tmp_path) == 15


def test_find_other_caches(tmp_path):
    write_cache(tmp_path / "known-cache")
    write_cache(tmp_path / "other-cache")
    (tmp_path / "plain-dir").mkdir()
    found = cli.find_other_caches(tmp_path, {"known-cache"})
    assert [n for n, _ in found] == ["other-cache"]
    assert found[0][1] > 0
    assert cli.find_other_caches(tmp_path / "nope", set()) == []


def test_remote_listing_paginates():
    pages = [
        {"Contents": [{"Size": 10}, {"Size": 20}], "IsTruncated": True, "NextContinuationToken": "t"},
        {"Contents": [{"Size": 5}], "IsTruncated": False},
    ]
    calls = []

    class FakeS3:
        def list_objects_v2(self, **kw):
            calls.append(kw)
            return pages[len(calls) - 1]

    class FakeSession:
        def client(self, *a, **kw):
            return FakeS3()

    with patch.object(cli, "_boto_session", return_value=FakeSession()):
        assert cli.remote_listing("s3://b/prefix") == (3, 35)
    assert calls[0] == {"Bucket": "b", "Prefix": "prefix/"}
    assert calls[1]["ContinuationToken"] == "t"


def test_check_s3_no_remote():
    info = cli.check_s3("s3://b/p/", check_remote=False)
    assert info.checked is False
    assert info.bucket_readable is None


def test_fmt_bytes():
    assert cli.fmt_bytes(0) == "0 B"
    assert cli.fmt_bytes(1536) == "1.5 KB"
    assert cli.fmt_bytes(17_089_829_376) == "15.9 GB"
    assert cli.fmt_bytes(None) == "?"


def test_mask_account():
    assert (
        cli.mask_account("arn:aws:iam::777749968893:user/alvarez")
        == "arn:aws:iam::********8893:user/alvarez"
    )
    assert cli.mask_account("no-account-here") == "no-account-here"


def test_split_s3():
    assert cli.split_s3("s3://bucket/a/b/") == ("bucket", "a/b/")
    assert cli.split_s3("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        cli.split_s3("/local/path")


def test_colorize_only_when_enabled(monkeypatch):
    monkeypatch.setattr(cli, "_COLOR", False)
    assert cli._colorize(f"  read  {cli.OK}") == f"  read  {cli.OK}"
    monkeypatch.setattr(cli, "_COLOR", True)
    out = cli._colorize(f"  {cli.OK} ok  {cli.BAD} missing  {cli.WARN} incomplete")
    assert "\033[32m" + cli.OK in out
    assert "\033[31m" + cli.BAD in out
    assert "\033[33m" + cli.WARN in out
    assert "\033[31mmissing" in out


def test_no_color_when_not_tty(cache_dir, capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    code, out = _run(["status", "--no-remote"], capsys)
    assert "\033[" not in out
    monkeypatch.setenv("FORCE_COLOR", "1")
    code, out = _run(["status", "--no-remote"], capsys)
    assert "\033[32m" in out
    code, out = _run(["status", "--no-remote", "--no-color"], capsys)
    assert "\033[" not in out


def test_public_api_surface():
    """Names visionlab-datasets relies on; keep stable."""
    for name in cli.__all__:
        assert hasattr(cli, name), name
    for name in (
        "inspect_dir",
        "check_s3",
        "remote_listing",
        "find_other_caches",
        "fmt_bytes",
        "dir_bytes",
        "mask_account",
        "configure_color",
        "print_line",
        "OK",
        "BAD",
        "WARN",
        "SKIP",
        "MANIFEST_FILE",
        "DirAccess",
        "S3Info",
    ):
        assert name in cli.__all__, name
    # backward-compatible private aliases still resolve
    assert cli._fmt_bytes is cli.fmt_bytes
    assert cli._problems is cli.problems
    assert cli._configure_color is cli.configure_color
    assert cli._print is cli.print_line


def test_python_dash_m_entry():
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
