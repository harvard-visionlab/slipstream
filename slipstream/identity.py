"""Store identity: a content hash of a slipcache, independent of where it lives.

``store_identity(dir)`` = ``"sha256:" + sha256(canonical JSON of {"num_samples", "file_sha256"})`` from the
manifest, or ``None`` when the manifest carries no ``file_sha256``. It covers only the sample count and the
per-file hashes, never paths, versions or other manifest keys, so every copy of a store (S3, lab storage,
node-local) has the same identity. The value is frozen: evals pins it, and visionlab-datasets re-exports this
function as ``cache_identity``. Stdlib only (safe to import without torch / numba).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slipstream.utils.cache_dir import MANIFEST_FILE


def store_identity(cache_dir: str | Path) -> str | None:
    """``"sha256:<hex>"`` identity of the slipcache at ``cache_dir`` (``None`` without ``file_sha256``)."""
    m = json.loads((Path(cache_dir) / MANIFEST_FILE).read_text())
    if not m.get("file_sha256"):
        return None
    body = json.dumps({"num_samples": m["num_samples"], "file_sha256": m["file_sha256"]},
                      sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()
