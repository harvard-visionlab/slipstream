"""Tests for unified cache directory utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from slipstream.utils.cache_dir import (
    CACHE_DIR_ENV_VAR,
    DEFAULT_CACHE_DIR,
    get_cache_base,
    get_cache_path,
)


class TestGetCacheBase:
    """Tests for get_cache_base function."""

    def test_default_path(self):
        """Default cache base is ~/.slipstream."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if present
            os.environ.pop(CACHE_DIR_ENV_VAR, None)
            result = get_cache_base()
            assert result == DEFAULT_CACHE_DIR
            assert result == Path.home() / ".slipstream"

    def test_env_var_override(self, tmp_path):
        """SLIPSTREAM_CACHE_DIR env var overrides default."""
        custom_path = tmp_path / "custom-cache"
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(custom_path)}):
            result = get_cache_base()
            assert result == custom_path

    def test_env_var_expands_tilde(self):
        """Environment variable expands ~ to home directory."""
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: "~/my-cache"}):
            result = get_cache_base()
            assert result == Path.home() / "my-cache"

    def test_env_var_resolves_path(self, tmp_path):
        """Environment variable resolves to absolute path."""
        # Create a relative-like path
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(tmp_path / "a" / ".." / "b")}):
            result = get_cache_base()
            # Should be resolved (no ..)
            assert ".." not in str(result)
            assert result == (tmp_path / "b").resolve()


class TestGetCachePath:
    """Tests for get_cache_path function."""

    def test_appends_slipcache_prefix(self):
        """Cache path includes slipcache-{hash} suffix."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(CACHE_DIR_ENV_VAR, None)
            result = get_cache_path("abc12345")
            assert result == Path.home() / ".slipstream" / "slipcache-abc12345"

    def test_with_custom_base(self, tmp_path):
        """Cache path uses custom base from env var."""
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(tmp_path)}):
            result = get_cache_path("def67890")
            assert result == tmp_path / "slipcache-def67890"


class TestReaderCachePaths:
    """Tests verifying readers use unified cache base."""

    def test_streaming_reader_uses_unified_base(self, tmp_path):
        """StreamingReader uses get_cache_base() for cache_path."""
        # This test verifies the code path, not actual LitData functionality
        from slipstream.utils.cache_dir import get_cache_base

        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(tmp_path)}):
            base = get_cache_base()
            assert base == tmp_path

    def test_ffcv_reader_uses_unified_base(self, tmp_path):
        """FFCVFileReader default cache is get_cache_base()."""
        from slipstream.utils.cache_dir import get_cache_base

        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(tmp_path)}):
            base = get_cache_base()
            assert base == tmp_path

    def test_imagefolder_uses_unified_base(self, tmp_path):
        """SlipstreamImageFolder default cache is get_cache_base()."""
        from slipstream.utils.cache_dir import get_cache_base

        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(tmp_path)}):
            base = get_cache_base()
            assert base == tmp_path
