"""Command-line interface for slipstream.

Gives lab members a one-command answer to "am I set up to train?"::

    slipstream status              # cache dir, permissions, S3 access, lab datasets
    slipstream config              # alias for status
    slipstream datasets            # list registered lab datasets (visionlab-datasets)
    slipstream sync imagenet100    # download a lab dataset cache from S3
    slipstream sync imagenet100 --split train --fmt yuv420
    slipstream sync s3://bucket/slipstream-cache/imagenet10/imagenet10-s256_l512-jpeg-val

Also runnable as ``python -m slipstream <command>``.

The lab-dataset registry lives in the optional ``visionlab-datasets`` package
(``visionlab.datasets``). When it is importable the CLI reports, per registered
(dataset, split, fmt), whether the cache is present locally and readable on S3.
Without it, the CLI still reports the cache directory, permissions, S3
credentials, and any slipstream caches found on disk.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform as _platform
import re
import shutil
import socket
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slipstream.cache import MANIFEST_FILE, OptimizedCache
from slipstream.utils.cache_dir import CACHE_DIR_ENV_VAR, DEFAULT_CACHE_DIR, get_cache_base
from slipstream.version import __version__

# Default S3 location of lab caches (used for the bucket read test when the
# registry is unavailable). Registry remote paths override this.
DEFAULT_REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/"

OK = "✓"
BAD = "✗"
WARN = "⚠"
SKIP = "-"

# ANSI colors, enabled in main() when stdout is a TTY (or FORCE_COLOR is set),
# disabled by --no-color or NO_COLOR.
_COLOR = False
_GREEN, _RED, _YELLOW, _BOLD = "\033[32m", "\033[31m", "\033[33m", "\033[1m"


def _paint(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _COLOR else text


def _colorize(line: str) -> str:
    """Color status glyphs and a few key phrases in a line of output."""
    if not _COLOR:
        return line
    line = line.replace(OK, _paint(OK, _GREEN)).replace(BAD, _paint(BAD, _RED))
    line = line.replace(WARN, _paint(WARN, _YELLOW))
    for word in ("missing", "incomplete", "unreadable", "denied", "not found", "error"):
        line = re.sub(rf"(?<=[{BAD}{WARN}]\033\[0m )({word})", lambda m: _paint(m.group(1), _RED), line)
    if line.startswith("Problems"):
        line = _paint(line, _RED + _BOLD)
    elif "Everything looks good" in line:
        line = _paint(line, _GREEN + _BOLD)
    return line


def _print(line: str = "") -> None:
    print(_colorize(line))


def _configure_color(no_color: bool = False) -> None:
    global _COLOR
    if no_color or os.environ.get("NO_COLOR"):
        _COLOR = False
    elif os.environ.get("FORCE_COLOR"):
        _COLOR = True
    else:
        _COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)())


def _mask_account(arn: str) -> str:
    """Hide all but the last 4 digits of the AWS account id in an ARN."""
    return re.sub(r"::(\d{12}):", lambda m: f"::********{m.group(1)[-4:]}:", arn)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def _split_s3(url: str) -> tuple[str, str]:
    """'s3://bucket/a/b/' -> ('bucket', 'a/b/')."""
    if not url.startswith("s3://"):
        raise ValueError(f"Not an S3 URL: {url}")
    rest = url[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _import_registry():
    """Return the ``visionlab.datasets`` module, or None if not installed."""
    try:
        from visionlab import datasets as vd  # type: ignore
    except Exception:
        return None
    return vd


def _owner_group(st: os.stat_result) -> tuple[str, str]:
    owner, group = str(st.st_uid), str(st.st_gid)
    try:
        import grp
        import pwd

        owner = pwd.getpwuid(st.st_uid).pw_name
        group = grp.getgrgid(st.st_gid).gr_name
    except Exception:
        pass
    return owner, group


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class DirAccess:
    path: str
    exists: bool = False
    is_dir: bool = False
    is_symlink: bool = False
    resolved: str | None = None
    readable: bool = False
    writable: bool = False
    traversable: bool = False
    owner: str | None = None
    group: str | None = None
    mode: str | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None
    can_create: bool = False  # when missing: nearest existing parent is writable
    error: str | None = None


def inspect_dir(path: Path) -> DirAccess:
    info = DirAccess(path=str(path))
    try:
        info.is_symlink = path.is_symlink()
        info.exists = path.exists()
        if info.is_symlink:
            info.resolved = str(path.resolve())
        if info.exists:
            st = path.stat()
            info.is_dir = stat.S_ISDIR(st.st_mode)
            info.owner, info.group = _owner_group(st)
            info.mode = stat.filemode(st.st_mode)
            info.readable = os.access(path, os.R_OK)
            info.writable = os.access(path, os.W_OK)
            info.traversable = os.access(path, os.X_OK)
            du = shutil.disk_usage(path)
            info.free_bytes, info.total_bytes = du.free, du.total
        else:
            parent = path
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            info.can_create = os.access(parent, os.W_OK)
            if parent.exists():
                du = shutil.disk_usage(parent)
                info.free_bytes, info.total_bytes = du.free, du.total
    except Exception as exc:  # pragma: no cover - defensive
        info.error = f"{type(exc).__name__}: {exc}"
    return info


@dataclass
class CacheDirInfo:
    path: str
    source: str  # where the setting came from
    env_var: str | None
    platform: str | None
    slipstream_default: str  # what plain slipstream would use (get_cache_base)
    mismatch: bool  # True if plain slipstream and visionlab-datasets disagree
    access: DirAccess
    platform_dirs: dict[str, str] = field(default_factory=dict)  # all platform defaults


def resolve_cache_dir(vd: Any) -> CacheDirInfo:
    env_val = os.environ.get(CACHE_DIR_ENV_VAR)
    platform_name: str | None = None
    if env_val:
        path = Path(env_val).expanduser()
        source = f"{CACHE_DIR_ENV_VAR} environment variable"
    elif vd is not None:
        try:
            plat = vd.detect_platform()
            platform_name = getattr(plat, "value", str(plat))
            path = Path(vd.get_platform_cache_dir(plat)).expanduser()
            source = f"visionlab-datasets platform default ({platform_name})"
        except Exception as exc:
            path = DEFAULT_CACHE_DIR
            source = f"slipstream default (visionlab-datasets platform detection failed: {exc})"
    else:
        path = DEFAULT_CACHE_DIR
        source = "slipstream default (~/.slipstream)"

    slip_default = get_cache_base()
    try:
        mismatch = path.resolve() != slip_default.resolve()
    except Exception:
        mismatch = str(path) != str(slip_default)

    platform_dirs: dict[str, str] = {}
    if vd is not None:
        try:
            from visionlab.datasets.runtime_platform import PLATFORM_CACHE_DIRS  # type: ignore

            platform_dirs = {getattr(k, "value", str(k)): str(v) for k, v in PLATFORM_CACHE_DIRS.items()}
        except Exception:
            pass

    return CacheDirInfo(
        path=str(path),
        source=source,
        env_var=env_val,
        platform=platform_name,
        slipstream_default=str(slip_default),
        mismatch=mismatch,
        access=inspect_dir(path),
        platform_dirs=platform_dirs,
    )


@dataclass
class S3Info:
    s5cmd_path: str | None = None
    s5cmd_version: str | None = None
    s5cmd_error: str | None = None
    credentials_found: bool = False
    credentials_method: str | None = None
    profile: str | None = None
    region: str | None = None
    identity_arn: str | None = None
    identity_error: str | None = None
    bucket_url: str | None = None
    bucket_readable: bool | None = None
    bucket_error: str | None = None
    checked: bool = True


def _boto_config():
    from botocore.config import Config

    return Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2})


def _boto_session(profile: str | None = None):
    import boto3

    return boto3.session.Session(profile_name=profile)


def check_s3(
    remote_base: str, *, check_remote: bool = True, endpoint_url: str | None = None
) -> S3Info:
    info = S3Info(bucket_url=remote_base, checked=check_remote)

    # s5cmd (required for sync/download)
    from slipstream.s3_sync import _check_s5cmd

    try:
        info.s5cmd_path = _check_s5cmd()
        import subprocess

        out = subprocess.run(
            [info.s5cmd_path, "version"], capture_output=True, text=True, timeout=5
        )
        info.s5cmd_version = out.stdout.strip() or None
    except RuntimeError as exc:
        info.s5cmd_error = str(exc).splitlines()[0]
    except Exception as exc:  # pragma: no cover
        info.s5cmd_error = f"{type(exc).__name__}: {exc}"

    # Credentials (boto3 default chain; same chain s5cmd uses)
    info.profile = os.environ.get("AWS_PROFILE")
    try:
        session = _boto_session(info.profile)
        info.region = session.region_name
        creds = session.get_credentials()
        if creds is not None:
            info.credentials_found = True
            info.credentials_method = getattr(creds, "method", None)
    except Exception as exc:
        info.identity_error = f"{type(exc).__name__}: {exc}"

    if not check_remote:
        return info

    if not info.credentials_found:
        info.identity_error = info.identity_error or "no AWS credentials found"
        info.bucket_readable = False
        info.bucket_error = "no credentials"
        return info

    try:
        sts = session.client("sts", config=_boto_config(), endpoint_url=None)
        info.identity_arn = _mask_account(sts.get_caller_identity()["Arn"])
    except Exception as exc:
        info.identity_error = f"{type(exc).__name__}: {exc}"

    try:
        bucket, prefix = _split_s3(remote_base)
        s3 = session.client("s3", config=_boto_config(), endpoint_url=endpoint_url)
        s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        info.bucket_readable = True
    except Exception as exc:
        info.bucket_readable = False
        info.bucket_error = f"{type(exc).__name__}: {exc}"
    return info


@dataclass
class DatasetEntry:
    dataset: str
    split: str
    fmt: str
    cache_name: str
    remote: str
    local_path: str
    local_status: str = "missing"  # ok | incomplete | missing | unreadable
    local_problems: list[str] = field(default_factory=list)
    local_bytes: int | None = None
    expected_samples: int | None = None  # from registry metadata num_{split}
    num_samples: int | None = None  # from local manifest
    remote_status: str = "unchecked"  # ok | missing | denied | error | unchecked
    remote_bytes: int | None = None
    remote_files: int | None = None
    remote_error: str | None = None


def registry_entries(vd: Any, cache_base: Path) -> list[DatasetEntry]:
    entries: list[DatasetEntry] = []
    for name in vd.list_datasets():
        cfg = vd.get_config(name)
        meta = getattr(cfg, "metadata", None) or {}
        for (split, fmt), remote in cfg.remote_cache.items():
            cache_name = remote.rstrip("/").rsplit("/", 1)[-1]
            expected = meta.get(f"num_{split}")
            entries.append(
                DatasetEntry(
                    dataset=name,
                    split=split,
                    fmt=fmt,
                    cache_name=cache_name,
                    remote=remote.rstrip("/") + "/",
                    local_path=str(cache_base / cache_name),
                    expected_samples=int(expected) if isinstance(expected, int) else None,
                )
            )
    return entries


def datasets_without_caches(vd: Any) -> list[str]:
    """Registered dataset names that have no remote caches yet."""
    return [n for n in vd.list_datasets() if not getattr(vd.get_config(n), "remote_cache", None)]


def _dir_bytes(path: Path) -> int:
    """Recursive on-disk size (matches what ``s5cmd cp prefix/*`` downloads)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def check_local(entry: DatasetEntry) -> None:
    path = Path(entry.local_path)
    manifest = path / MANIFEST_FILE
    if not manifest.exists():
        entry.local_status = "missing"
        return
    if not os.access(manifest, os.R_OK) or not os.access(path, os.R_OK | os.X_OK):
        entry.local_status = "unreadable"
        entry.local_problems = ["no read permission"]
        return
    ok, problems = OptimizedCache.check_integrity(path)
    entry.local_status = "ok" if ok else "incomplete"
    entry.local_problems = problems
    try:
        with open(manifest) as f:
            entry.num_samples = int(json.load(f).get("num_samples"))
    except Exception:
        entry.num_samples = None
    if entry.expected_samples is not None and entry.num_samples is not None:
        if entry.num_samples != entry.expected_samples:
            entry.local_problems.append(
                f"sample count {entry.num_samples:,} != registry num_{entry.split} {entry.expected_samples:,}"
            )
    try:
        entry.local_bytes = _dir_bytes(path)
    except OSError:
        pass
    if ok:
        # Files exist with right sizes, but can we actually open them?
        for p in path.iterdir():
            if p.is_file() and not os.access(p, os.R_OK):
                entry.local_status = "unreadable"
                entry.local_problems = [f"no read permission: {p.name}"]
                break


def remote_listing(
    remote: str, *, endpoint_url: str | None = None, profile: str | None = None
) -> tuple[int, int]:
    """Return (num_files, total_bytes) for an S3 prefix. Raises on error."""
    bucket, prefix = _split_s3(remote.rstrip("/") + "/")
    s3 = _boto_session(profile).client("s3", config=_boto_config(), endpoint_url=endpoint_url)
    n, total = 0, 0
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            n += 1
            total += obj["Size"]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return n, total


def check_remote(
    entry: DatasetEntry, *, endpoint_url: str | None = None, profile: str | None = None
) -> None:
    try:
        n, total = remote_listing(entry.remote, endpoint_url=endpoint_url, profile=profile)
    except Exception as exc:
        msg = str(exc)
        entry.remote_status = "denied" if "AccessDenied" in msg or "Forbidden" in msg else "error"
        entry.remote_error = f"{type(exc).__name__}: {msg}"
        return
    entry.remote_files, entry.remote_bytes = n, total
    entry.remote_status = "ok" if n > 0 else "missing"


def find_other_caches(cache_base: Path, known: set[str]) -> list[tuple[str, int]]:
    """Slipstream caches on disk that aren't in the registry: [(name, bytes)]."""
    out: list[tuple[str, int]] = []
    if not cache_base.is_dir() or not os.access(cache_base, os.R_OK | os.X_OK):
        return out
    for p in sorted(cache_base.iterdir()):
        if p.name in known or not p.is_dir():
            continue
        if (p / MANIFEST_FILE).exists():
            try:
                out.append((p.name, _dir_bytes(p)))
            except OSError:
                out.append((p.name, 0))
    return out


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def collect_status(
    *, check_remote_access: bool = True, endpoint_url: str | None = None
) -> dict[str, Any]:
    vd = _import_registry()
    cache = resolve_cache_dir(vd)
    cache_base = Path(cache.path)

    entries = registry_entries(vd, cache_base) if vd is not None else []
    no_caches = datasets_without_caches(vd) if vd is not None else []
    remote_base = DEFAULT_REMOTE_CACHE_BASE
    if entries:
        # common S3 base of registry entries (bucket + first path component)
        bucket, key = _split_s3(entries[0].remote)
        remote_base = f"s3://{bucket}/{key.split('/', 1)[0]}/"

    s3 = check_s3(remote_base, check_remote=check_remote_access, endpoint_url=endpoint_url)

    for e in entries:
        check_local(e)
    if check_remote_access and s3.credentials_found and entries:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda e: check_remote(e, endpoint_url=endpoint_url, profile=s3.profile),
                    entries,
                )
            )

    others = find_other_caches(cache_base, {e.cache_name for e in entries})

    return {
        "slipstream_version": __version__,
        "visionlab_datasets_version": getattr(vd, "__version__", None) if vd else None,
        "python": _platform.python_version(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "cache": asdict(cache),
        "s3": asdict(s3),
        "datasets": [asdict(e) for e in entries],
        "datasets_without_caches": no_caches,
        "other_caches": [{"name": n, "bytes": b} for n, b in others],
    }


def _problems(status: dict[str, Any]) -> list[str]:
    """Human-readable list of things that will block training."""
    out: list[str] = []
    acc = status["cache"]["access"]
    cache_path = status["cache"]["path"]
    if not acc["exists"]:
        if acc["can_create"]:
            out.append(
                f"Cache dir {cache_path} does not exist yet (will be created on first sync)."
            )
        else:
            out.append(
                f"Cache dir {cache_path} does not exist and cannot be created (parent not writable)."
            )
    else:
        if not acc["readable"] or not acc["traversable"]:
            out.append(
                f"No read access to cache dir {cache_path} (owner {acc['owner']}, mode {acc['mode']})."
            )
        if not acc["writable"]:
            out.append(
                f"No write access to cache dir {cache_path}; you can use existing caches but not sync new ones."
            )
    s3 = status["s3"]
    if s3["s5cmd_error"]:
        out.append(f"s5cmd not usable: {s3['s5cmd_error']}  (fix: uv tool install s5cmd)")
    if not s3["credentials_found"]:
        out.append(
            "No AWS credentials found (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or configure ~/.aws/credentials)."
        )
    elif s3["checked"] and s3["bucket_readable"] is False:
        out.append(f"Cannot list {s3['bucket_url']}: {s3['bucket_error']}")
    if status["visionlab_datasets_version"] is None:
        out.append(
            "visionlab-datasets not installed; lab dataset registry unavailable (pip install git+https://github.com/harvard-visionlab/datasets.git)."
        )
    return out


def _mark(ok: bool | None) -> str:
    if ok is None:
        return SKIP
    return OK if ok else BAD


def print_status(status: dict[str, Any]) -> None:
    p = _print
    c = status["cache"]
    a = c["access"]
    s3 = status["s3"]

    p(
        f"slipstream {status['slipstream_version']}"
        + (
            f"  ·  visionlab-datasets {status['visionlab_datasets_version']}"
            if status["visionlab_datasets_version"]
            else "  ·  visionlab-datasets: not installed"
        )
        + f"  ·  python {status['python']}"
    )
    p(f"{status['user']}@{status['host']}")
    p()

    p("Cache directory")
    p(f"  path        {c['path']}" + (f"  -> {a['resolved']}" if a["is_symlink"] else ""))
    p(f"  source      {c['source']}")
    p(f"  {CACHE_DIR_ENV_VAR}  {c['env_var'] or '(not set)'}")
    if c.get("platform_dirs"):
        items = ", ".join(f"{k}={v}" for k, v in c["platform_dirs"].items())
        p(f"  platforms   {items}")
    if a["exists"]:
        p(f"  exists      {OK}  owner {a['owner']}:{a['group']}  mode {a['mode']}")
        p(f"  read        {_mark(a['readable'] and a['traversable'])}")
        p(f"  write       {_mark(a['writable'])}")
    else:
        p(f"  exists      {BAD}  (not created yet; can create: {_mark(a['can_create'])})")
    if a["free_bytes"] is not None:
        p(f"  disk free   {_fmt_bytes(a['free_bytes'])} of {_fmt_bytes(a['total_bytes'])}")
    if c["mismatch"]:
        p(
            f"  {WARN} plain slipstream (without visionlab-datasets) would use {c['slipstream_default']}."
        )
        p(f"    Set {CACHE_DIR_ENV_VAR}={c['path']} to make every code path agree.")
    p()

    p("S3 access")
    if s3["s5cmd_path"]:
        p(f"  s5cmd       {OK}  {s3['s5cmd_path']} ({s3['s5cmd_version']})")
    else:
        p(f"  s5cmd       {BAD}  {s3['s5cmd_error']}")
    if s3["credentials_found"]:
        extra = f" via {s3['credentials_method']}" if s3["credentials_method"] else ""
        extra += f", profile {s3['profile']}" if s3["profile"] else ""
        extra += f", region {s3['region']}" if s3["region"] else ""
        p(f"  credentials {OK} {extra.strip()}")
    else:
        p(f"  credentials {BAD}  none found")
    if not s3["checked"]:
        p(f"  identity    {SKIP}  (remote checks skipped)")
    elif s3["identity_arn"]:
        p(f"  identity    {OK}  {s3['identity_arn']}")
    elif s3["identity_error"]:
        p(f"  identity    {BAD}  {s3['identity_error']}")
    if s3["checked"]:
        if s3["bucket_readable"]:
            p(f"  read        {OK}  {s3['bucket_url']}")
        elif s3["bucket_readable"] is False:
            p(f"  read        {BAD}  {s3['bucket_url']}: {s3['bucket_error']}")
    p()

    entries = status["datasets"]
    if entries:
        p("Lab datasets  (local = in cache dir, remote = readable on S3)")
        w_ds = max(len("dataset"), *(len(e["dataset"]) for e in entries)) + 2
        w_split = max(len("split"), *(len(e["split"]) for e in entries)) + 2
        w_fmt = max(len("fmt"), *(len(e["fmt"]) for e in entries)) + 2
        p(
            f"  {'dataset':<{w_ds}}{'split':<{w_split}}{'fmt':<{w_fmt}}{'local':<22}{'remote':<14}cache name"
        )
        for e in entries:
            ls = e["local_status"]
            if ls == "ok":
                local = f"{OK} {_fmt_bytes(e['local_bytes']):>9}"
            elif ls == "incomplete":
                local = f"{WARN} incomplete"
            elif ls == "unreadable":
                local = f"{BAD} unreadable"
            else:
                local = f"{BAD} missing"
            rs = e["remote_status"]
            if rs == "ok":
                remote = f"{OK} {_fmt_bytes(e['remote_bytes']):>9}"
            elif rs == "unchecked":
                remote = f"{SKIP}"
            elif rs == "missing":
                remote = f"{BAD} not found"
            elif rs == "denied":
                remote = f"{BAD} denied"
            else:
                remote = f"{BAD} error"
            p(
                f"  {e['dataset']:<{w_ds}}{e['split']:<{w_split}}{e['fmt']:<{w_fmt}}{local:<22}{remote:<14}{e['cache_name']}"
            )
        p(f"  local root: {c['path']}")
        problems = [e for e in entries if e["local_problems"]]
        for e in problems:
            p(f"  {WARN} {e['cache_name']}: {'; '.join(e['local_problems'][:3])}")
        errs = [e for e in entries if e["remote_status"] in ("denied", "error")]
        if errs:
            p(f"  {WARN} remote error example ({errs[0]['cache_name']}): {errs[0]['remote_error']}")
        missing = [e for e in entries if e["local_status"] != "ok"]
        if missing:
            ex = missing[0]
            p(
                f"  to fetch:   slipstream sync {ex['dataset']} --split {ex['split']} --fmt {ex['fmt']}"
            )
        if status.get("datasets_without_caches"):
            p(f"  registered but no remote caches yet: {', '.join(status['datasets_without_caches'])}")
        p()

    if status["other_caches"]:
        p("Other slipstream caches in cache dir")
        for o in status["other_caches"]:
            p(f"  {o['name']:<45}{_fmt_bytes(o['bytes']):>10}")
        p()

    probs = _problems(status)
    if probs:
        p("Problems")
        for msg in probs:
            p(f"  {BAD} {msg}")
    else:
        p(f"{OK} Everything looks good.")


def cmd_status(args: argparse.Namespace) -> int:
    status = collect_status(check_remote_access=not args.no_remote, endpoint_url=args.endpoint_url)
    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        print_status(status)
    probs = _problems(status)
    hard = [m for m in probs if "will be created" not in m and "not installed" not in m]
    return 1 if hard else 0


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


def cmd_datasets(args: argparse.Namespace) -> int:
    vd = _import_registry()
    if vd is None:
        print("visionlab-datasets is not installed; no lab dataset registry available.")
        print("  pip install git+https://github.com/harvard-visionlab/datasets.git")
        return 1
    if args.json:
        out = []
        for name in vd.list_datasets():
            cfg = vd.get_config(name)
            out.append(
                {
                    "name": name,
                    "num_classes": cfg.num_classes,
                    "remote_cache": {f"{s}/{f}": r for (s, f), r in cfg.remote_cache.items()},
                }
            )
        print(json.dumps(out, indent=2))
        return 0
    for name in vd.list_datasets():
        cfg = vd.get_config(name)
        print(f"{name}  ({cfg.num_classes} classes)")
        if not cfg.remote_cache:
            print("  (no remote caches registered)")
        for (split, fmt), remote in cfg.remote_cache.items():
            print(f"  {split:<6}{fmt:<8}{remote}")
    return 0


# --------------------------------------------------------------------------- #
# sync
# --------------------------------------------------------------------------- #


def _select_entries(
    vd: Any, names: list[str], split: str, fmt: str, cache_base: Path
) -> list[DatasetEntry]:
    all_entries = registry_entries(vd, cache_base)
    known = set(vd.list_datasets())
    selected: list[DatasetEntry] = []
    for name in names:
        if name not in known:
            raise SystemExit(f"Unknown dataset {name!r}. Available: {', '.join(sorted(known))}")
        for e in all_entries:
            if e.dataset != name:
                continue
            if split != "all" and e.split != split:
                continue
            if fmt != "all" and e.fmt != fmt:
                continue
            selected.append(e)
        if not any(e.dataset == name for e in selected):
            avail = sorted(
                {f"--split {e.split} --fmt {e.fmt}" for e in all_entries if e.dataset == name}
            )
            raise SystemExit(
                f"{name!r} has no cache for split={split!r} fmt={fmt!r}. Available: {avail}"
            )
    return selected


def cmd_sync(args: argparse.Namespace) -> int:
    from slipstream.s3_sync import download_s3_cache

    vd = _import_registry()
    cache = resolve_cache_dir(vd)
    cache_base = Path(args.dest) if args.dest else Path(cache.path)

    entries: list[DatasetEntry] = []
    for target in args.targets:
        if target.startswith("s3://"):
            cache_name = target.rstrip("/").rsplit("/", 1)[-1]
            entries.append(
                DatasetEntry(
                    dataset=cache_name,
                    split="-",
                    fmt="-",
                    cache_name=cache_name,
                    remote=target.rstrip("/") + "/",
                    local_path=str(cache_base / cache_name),
                )
            )
        else:
            if vd is None:
                raise SystemExit(
                    f"Cannot resolve dataset name {target!r}: visionlab-datasets is not installed.\n"
                    "Pass an s3:// URL instead, or: pip install git+https://github.com/harvard-visionlab/datasets.git"
                )
            entries.extend(_select_entries(vd, [target], args.split, args.fmt, cache_base))

    # De-duplicate while preserving order
    seen: set[str] = set()
    entries = [e for e in entries if not (e.remote in seen or seen.add(e.remote))]

    _print(f"Cache dir: {cache_base}")
    for e in entries:
        check_local(e)

    todo: list[DatasetEntry] = []
    for e in entries:
        if e.local_status == "ok" and not args.force:
            _print(
                f"  {OK} {e.cache_name}: already present ({_fmt_bytes(e.local_bytes)}), skipping (use --force to re-download)"
            )
            continue
        todo.append(e)
    if not todo:
        return 0

    # Remote sizes + disk check
    total_needed = 0
    for e in todo:
        try:
            e.remote_files, e.remote_bytes = remote_listing(
                e.remote, endpoint_url=args.endpoint_url
            )
            e.remote_status = "ok" if e.remote_files else "missing"
        except Exception as exc:
            e.remote_status = "error"
            e.remote_error = f"{type(exc).__name__}: {exc}"
        state = {
            "missing": "not present locally",
            "incomplete": "incomplete locally",
            "unreadable": "unreadable locally",
        }.get(e.local_status, "re-download")
        if e.remote_status == "ok":
            _print(
                f"  {e.cache_name}: {state}; remote {e.remote_files} files, {_fmt_bytes(e.remote_bytes)}  <- {e.remote}"
            )
            total_needed += e.remote_bytes or 0
        else:
            _print(
                f"  {BAD} {e.cache_name}: remote {e.remote_status} ({e.remote_error or 'no files at ' + e.remote})"
            )

    todo = [e for e in todo if e.remote_status == "ok"]
    if not todo:
        return 1

    acc = inspect_dir(cache_base)
    if acc.free_bytes is not None:
        _print(
            f"  need ~{_fmt_bytes(total_needed)}, free {_fmt_bytes(acc.free_bytes)} at {cache_base}"
        )
        if acc.free_bytes < total_needed:
            _print(f"  {BAD} not enough free disk space")
            if not args.force:
                return 1
    if acc.exists and not acc.writable:
        _print(f"  {BAD} cache dir {cache_base} is not writable")
        return 1
    if not acc.exists and not acc.can_create:
        _print(f"  {BAD} cache dir {cache_base} cannot be created")
        return 1

    if args.dry_run:
        _print("[dry-run] nothing downloaded")
        return 0

    failed = 0
    for e in todo:
        _print()
        ok = download_s3_cache(
            e.remote,
            Path(e.local_path),
            endpoint_url=args.endpoint_url,
            numworkers=args.numworkers,
            verbose=True,
        )
        if ok:
            check_local(e)
            ok = e.local_status == "ok"
            if not ok:
                _print(
                    f"  {BAD} {e.cache_name}: downloaded but integrity check failed: {'; '.join(e.local_problems[:3])}"
                )
        if ok:
            _print(f"  {OK} {e.cache_name} -> {e.local_path}")
        else:
            failed += 1
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slipstream",
        description="slipstream: cache location, dataset availability, S3 access, and sync.",
    )
    parser.add_argument("--version", action="version", version=f"slipstream {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("status", "config"):
        sp = sub.add_parser(
            name,
            help="Show cache dir, permissions, S3 access, and lab dataset availability"
            + (" (alias for status)" if name == "config" else ""),
        )
        sp.add_argument(
            "--no-remote", action="store_true", help="Skip network checks (S3 identity/listing)"
        )
        sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
        sp.add_argument("--json", action="store_true", help="Machine-readable output")
        sp.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
        sp.set_defaults(func=cmd_status)

    sp = sub.add_parser(
        "datasets", aliases=["list"], help="List registered lab datasets (needs visionlab-datasets)"
    )
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_datasets)

    sp = sub.add_parser(
        "sync",
        help="Download lab dataset cache(s) from S3 into the cache dir",
        description=(
            "Examples:\n"
            "  slipstream sync imagenet100                       # val+train, jpeg\n"
            "  slipstream sync imagenet100 --fmt yuv420          # val+train, yuv420\n"
            "  slipstream sync imagenet1k --split val --fmt all  # both formats\n"
            "  slipstream sync s3://visionlab-datasets/slipstream-cache/imagenet10/imagenet10-s256_l512-jpeg-val\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument("targets", nargs="+", help="Registered dataset name(s) or s3:// cache URL(s)")
    sp.add_argument("--split", default="all", help="train | val | all (default: all)")
    sp.add_argument("--fmt", default="jpeg", help="jpeg | yuv420 | all (default: jpeg)")
    sp.add_argument("--dest", default=None, help="Override cache dir (default: resolved cache dir)")
    sp.add_argument("--force", action="store_true", help="Re-download even if present and intact")
    sp.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    sp.add_argument(
        "--numworkers", type=int, default=32, help="s5cmd parallel workers (default: 32)"
    )
    sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
    sp.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    sp.set_defaults(func=cmd_sync)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_color(getattr(args, "no_color", False))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
