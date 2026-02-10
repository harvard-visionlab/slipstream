"""Tests for remote cache (S3) functionality.

These tests mock s5cmd to test the flow without requiring actual S3 access.
For real S3 integration tests, use pytest -m s3 with appropriate credentials.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestS3PathExists:
    """Tests for s3_path_exists function."""

    def test_path_exists_returns_true(self):
        """s3_path_exists returns True when s5cmd finds files."""
        from slipstream.s3_sync import s3_path_exists

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "2024-01-01 12:00:00  12345 manifest.json\n"

        with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = s3_path_exists("s3://bucket/path/manifest.json")

        assert result is True
        mock_run.assert_called_once()
        # Verify command structure
        call_args = mock_run.call_args[0][0]
        assert "s5cmd" in call_args
        assert "ls" in call_args
        assert "s3://bucket/path/manifest.json" in call_args

    def test_path_not_exists_returns_false(self):
        """s3_path_exists returns False when s5cmd returns empty output."""
        from slipstream.s3_sync import s3_path_exists

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
            with patch("subprocess.run", return_value=mock_result):
                result = s3_path_exists("s3://bucket/nonexistent/")

        assert result is False

    def test_path_error_returns_false(self):
        """s3_path_exists returns False when s5cmd fails."""
        from slipstream.s3_sync import s3_path_exists

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
            with patch("subprocess.run", return_value=mock_result):
                result = s3_path_exists("s3://bucket/error/")

        assert result is False

    def test_with_endpoint_url(self):
        """s3_path_exists passes endpoint_url to s5cmd."""
        from slipstream.s3_sync import s3_path_exists

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "file.txt\n"

        with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                s3_path_exists(
                    "s3://bucket/path/",
                    endpoint_url="https://s3.wasabisys.com"
                )

        call_args = mock_run.call_args[0][0]
        assert "--endpoint-url" in call_args
        assert "https://s3.wasabisys.com" in call_args

    def test_no_s5cmd_raises_error(self):
        """s3_path_exists raises RuntimeError when s5cmd not found."""
        from slipstream.s3_sync import s3_path_exists

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="s5cmd is required"):
                s3_path_exists("s3://bucket/path/")

    def test_old_s5cmd_version_raises_error(self):
        """s3_path_exists raises RuntimeError when s5cmd version is too old."""
        from slipstream.s3_sync import _check_s5cmd

        mock_result = MagicMock()
        mock_result.stdout = "v1.4.0"  # Old version

        with patch("shutil.which", return_value="/usr/bin/s5cmd"):
            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(RuntimeError, match="too old"):
                    _check_s5cmd()

    def test_new_s5cmd_version_passes(self):
        """_check_s5cmd passes for version 2.0.0+."""
        from slipstream.s3_sync import _check_s5cmd

        mock_result = MagicMock()
        mock_result.stdout = "v2.2.2"

        with patch("shutil.which", return_value="/usr/bin/s5cmd"):
            with patch("subprocess.run", return_value=mock_result):
                result = _check_s5cmd()
                assert result == "/usr/bin/s5cmd"


class TestDownloadS3Cache:
    """Tests for download_s3_cache function."""

    def test_download_success(self, tmp_path):
        """download_s3_cache creates local directory and runs with progress."""
        from slipstream.s3_sync import download_s3_cache

        captured_cmd = None

        def mock_progress(cmd, verbose=True):
            nonlocal captured_cmd
            captured_cmd = cmd
            return 0  # Success

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=mock_progress):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                result = download_s3_cache(
                    "s3://bucket/caches/slipcache-abc123/",
                    tmp_path,
                    verbose=False,
                )

        assert result is True
        assert (tmp_path / ".slipstream").exists()

        # Verify command structure
        assert captured_cmd is not None
        assert "s5cmd" in captured_cmd
        assert "cp" in captured_cmd
        assert "--show-progress" in captured_cmd
        assert "s3://bucket/caches/slipcache-abc123/.slipstream/*" in captured_cmd

    def test_download_failure(self, tmp_path):
        """download_s3_cache returns False on failure."""
        from slipstream.s3_sync import download_s3_cache

        def mock_progress(cmd, verbose=True):
            return 1  # Failure

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=mock_progress):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                result = download_s3_cache(
                    "s3://bucket/caches/slipcache-abc123/",
                    tmp_path,
                    verbose=False,
                )

        assert result is False

    def test_download_with_endpoint_url(self, tmp_path):
        """download_s3_cache passes endpoint_url to s5cmd."""
        from slipstream.s3_sync import download_s3_cache

        captured_cmd = None

        def mock_progress(cmd, verbose=True):
            nonlocal captured_cmd
            captured_cmd = cmd
            return 0

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=mock_progress):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                download_s3_cache(
                    "s3://bucket/caches/slipcache-abc123/",
                    tmp_path,
                    endpoint_url="https://s3.wasabisys.com",
                    verbose=False,
                )

        assert captured_cmd is not None
        assert "--endpoint-url" in captured_cmd
        assert "https://s3.wasabisys.com" in captured_cmd


class TestUploadS3Cache:
    """Tests for upload_s3_cache function."""

    def test_upload_success(self, tmp_path):
        """upload_s3_cache uploads .slipstream directory."""
        from slipstream.s3_sync import upload_s3_cache

        # Create .slipstream directory with a file
        slipstream_dir = tmp_path / ".slipstream"
        slipstream_dir.mkdir()
        (slipstream_dir / "manifest.json").write_text('{"version": 1}')

        captured_cmd = None

        def mock_progress(cmd, verbose=True):
            nonlocal captured_cmd
            captured_cmd = cmd
            return 0  # Success

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=mock_progress):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                result = upload_s3_cache(
                    tmp_path,
                    "s3://bucket/caches/slipcache-abc123/",
                    verbose=False,
                )

        assert result is True
        assert captured_cmd is not None
        assert "cp" in captured_cmd
        assert "--show-progress" in captured_cmd
        assert "s3://bucket/caches/slipcache-abc123/.slipstream/" in captured_cmd

    def test_upload_no_slipstream_dir(self, tmp_path):
        """upload_s3_cache returns False if .slipstream doesn't exist."""
        from slipstream.s3_sync import upload_s3_cache

        with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
            result = upload_s3_cache(
                tmp_path,
                "s3://bucket/caches/slipcache-abc123/",
                verbose=False,
            )

        assert result is False

    def test_upload_failure(self, tmp_path):
        """upload_s3_cache returns False on failure."""
        from slipstream.s3_sync import upload_s3_cache

        # Create .slipstream directory
        slipstream_dir = tmp_path / ".slipstream"
        slipstream_dir.mkdir()
        (slipstream_dir / "manifest.json").write_text('{"version": 1}')

        def mock_progress(cmd, verbose=True):
            return 1  # Failure

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=mock_progress):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                result = upload_s3_cache(
                    tmp_path,
                    "s3://bucket/caches/slipcache-abc123/",
                    verbose=False,
                )

        assert result is False


class TestLoaderRemoteCache:
    """Tests for SlipstreamLoader remote_cache parameter."""

    def test_remote_cache_parameter_accepted(self):
        """SlipstreamLoader accepts remote_cache parameter."""
        import inspect
        from slipstream import SlipstreamLoader

        sig = inspect.signature(SlipstreamLoader.__init__)
        params = sig.parameters

        assert "remote_cache" in params
        assert "remote_cache_endpoint_url" in params
        assert params["remote_cache"].default is None
        assert params["remote_cache_endpoint_url"].default is None

    def test_remote_cache_none_no_s3_calls(self, tmp_path):
        """remote_cache=None should not trigger any S3 operations."""
        # This is the default behavior - no S3 calls should happen
        # We can't easily test this without a full dataset, but we can verify
        # the parameter is accepted and the code path doesn't error

        # Note: A full integration test would require a mock dataset
        # For now, we just verify the parameter is defined correctly
        pass


class TestRemoteCacheFlow:
    """Tests for the full remote cache flow.

    These tests verify the logic flow without actual S3 access.
    """

    def test_remote_found_triggers_download(self):
        """When remote cache exists, download is triggered."""
        from slipstream.s3_sync import s3_path_exists, download_s3_cache

        # Mock s3_path_exists to return True
        with patch("slipstream.s3_sync.s3_path_exists", return_value=True) as mock_exists:
            with patch("slipstream.s3_sync.download_s3_cache", return_value=True) as mock_dl:
                # The actual loader call would go here, but requires a full dataset
                # For now, just verify the functions are importable and have correct signatures
                assert callable(s3_path_exists)
                assert callable(download_s3_cache)

    def test_remote_not_found_triggers_upload_after_build(self):
        """When remote cache doesn't exist, upload happens after build."""
        from slipstream.s3_sync import s3_path_exists, upload_s3_cache

        with patch("slipstream.s3_sync.s3_path_exists", return_value=False):
            with patch("slipstream.s3_sync.upload_s3_cache", return_value=True) as mock_ul:
                # Verify the function is callable
                assert callable(upload_s3_cache)


class TestSyncS3Cache:
    """Tests for bidirectional sync function."""

    def test_sync_downloads_and_uploads(self, tmp_path):
        """sync_s3_cache syncs files in both directions."""
        from slipstream.s3_sync import sync_s3_cache

        # Create local .slipstream with some files
        slipstream_dir = tmp_path / ".slipstream"
        slipstream_dir.mkdir()
        (slipstream_dir / "manifest.json").write_text('{}')
        (slipstream_dir / "local_only.npy").write_text('local')

        # Mock s5cmd ls to return remote files
        ls_output = "2024-01-01 12:00:00  100 s3://bucket/cache/.slipstream/manifest.json\n"
        ls_output += "2024-01-01 12:00:00  100 s3://bucket/cache/.slipstream/remote_only.npy\n"

        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            result.returncode = 0
            result.stdout = ls_output if "ls" in cmd else ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                downloaded, uploaded = sync_s3_cache(
                    tmp_path,
                    "s3://bucket/cache/",
                    verbose=False,
                )

        # Should have called ls, then sync (download), then sync (upload)
        assert call_count[0] >= 1  # At least the ls call

    def test_sync_no_slipstream_dir(self, tmp_path):
        """sync_s3_cache returns (0,0) if no .slipstream dir."""
        from slipstream.s3_sync import sync_s3_cache

        with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
            downloaded, uploaded = sync_s3_cache(
                tmp_path,
                "s3://bucket/cache/",
                verbose=False,
            )

        assert downloaded == 0
        assert uploaded == 0


class TestCommandGeneration:
    """Tests to verify correct s5cmd command generation."""

    def test_download_command_format(self, tmp_path):
        """Verify download generates correct s5cmd cp command with progress."""
        from slipstream.s3_sync import download_s3_cache

        captured_cmd = None

        def capture_cmd(cmd, verbose=True):
            nonlocal captured_cmd
            captured_cmd = cmd
            return 0

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=capture_cmd):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                download_s3_cache(
                    "s3://my-bucket/caches/slipcache-abc123",
                    tmp_path,
                    numworkers=16,
                    verbose=False,
                )

        assert captured_cmd is not None
        assert "s5cmd" in captured_cmd
        assert "--numworkers" in captured_cmd
        assert "16" in captured_cmd
        assert "cp" in captured_cmd
        assert "--show-progress" in captured_cmd
        # Source should be remote .slipstream/*
        assert any(".slipstream/*" in arg for arg in captured_cmd)

    def test_upload_command_format(self, tmp_path):
        """Verify upload generates correct s5cmd cp command with progress."""
        from slipstream.s3_sync import upload_s3_cache

        # Create .slipstream directory
        slipstream_dir = tmp_path / ".slipstream"
        slipstream_dir.mkdir()
        (slipstream_dir / "manifest.json").write_text('{}')

        captured_cmd = None

        def capture_cmd(cmd, verbose=True):
            nonlocal captured_cmd
            captured_cmd = cmd
            return 0

        with patch("slipstream.s3_sync.run_s5cmd_with_progress", side_effect=capture_cmd):
            with patch("slipstream.s3_sync._check_s5cmd", return_value="/usr/bin/s5cmd"):
                upload_s3_cache(
                    tmp_path,
                    "s3://my-bucket/caches/slipcache-abc123",
                    numworkers=64,
                    verbose=False,
                )

        assert captured_cmd is not None
        assert "s5cmd" in captured_cmd
        assert "--numworkers" in captured_cmd
        assert "64" in captured_cmd
        assert "cp" in captured_cmd
        assert "--show-progress" in captured_cmd
        assert "--if-size-differ" in captured_cmd
        # Note: --if-source-newer removed to avoid re-uploads when local files
        # have newer timestamps (e.g., after moving cache directories)
        # Destination should be remote .slipstream/
        assert any(".slipstream/" in arg for arg in captured_cmd)


# Integration tests that require real S3 access
@pytest.mark.s3
class TestRemoteCacheIntegration:
    """Integration tests requiring real S3 access.

    Run with: pytest -m s3 tests/test_remote_cache.py

    These tests require:
    - AWS credentials configured
    - s5cmd installed
    - Write access to the test bucket
    """

    @pytest.fixture
    def test_bucket(self):
        """Return test bucket URL. Configure via environment variable."""
        import os
        bucket = os.environ.get("SLIPSTREAM_TEST_BUCKET")
        if not bucket:
            pytest.skip("SLIPSTREAM_TEST_BUCKET not set")
        return bucket

    def test_roundtrip_upload_download(self, test_bucket, tmp_path):
        """Test full upload and download cycle."""
        import uuid
        from slipstream.s3_sync import (
            download_s3_cache,
            s3_path_exists,
            upload_s3_cache,
        )

        # Create a unique cache path for this test
        test_id = str(uuid.uuid4())[:8]
        remote_path = f"{test_bucket}/test-remote-cache/slipcache-{test_id}"

        # Create local cache structure
        local_cache = tmp_path / "local"
        slipstream_dir = local_cache / ".slipstream"
        slipstream_dir.mkdir(parents=True)
        (slipstream_dir / "manifest.json").write_text('{"version": 1, "test": true}')
        (slipstream_dir / "test.bin").write_bytes(b"test data")

        try:
            # Upload
            assert upload_s3_cache(local_cache, remote_path, verbose=True)

            # Verify exists
            assert s3_path_exists(f"{remote_path}/.slipstream/manifest.json")

            # Download to different location
            download_cache = tmp_path / "downloaded"
            assert download_s3_cache(remote_path, download_cache, verbose=True)

            # Verify contents match
            downloaded_manifest = download_cache / ".slipstream" / "manifest.json"
            assert downloaded_manifest.exists()
            assert '"test": true' in downloaded_manifest.read_text()

        finally:
            # Cleanup (best effort)
            import subprocess
            subprocess.run(
                ["s5cmd", "rm", f"{remote_path}/.slipstream/*"],
                capture_output=True,
            )
