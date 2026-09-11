"""Command-line interface for slipstream: plumbing checks.

    slipstream status     # cache dir + permissions, s5cmd, AWS credentials/identity,
                          # S3 bucket read, slipstream caches found on disk
    slipstream config     # alias for status

Also runnable as ``python -m slipstream status``.

This command is registry-agnostic. Lab members who want to see which lab
datasets are present locally, or fetch them from S3, should use the
``visionlab-datasets`` CLI (``visionlab-datasets status|sync``), which is
built on the public helpers in this module:

    inspect_dir, check_s3, remote_listing, find_other_caches, fmt_bytes,
    dir_bytes, mask_account, configure_color, print_line, OK/BAD/WARN/SKIP,
    DirAccess, S3Info, MANIFEST_FILE
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slipstream.cache import MANIFEST_FILE
from slipstream.utils.cache_dir import CACHE_DIR_ENV_VAR, DEFAULT_CACHE_DIR
from slipstream.version import __version__

# S3 location of the lab caches; used for the bucket read test.
DEFAULT_REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/"

OK = "✓"
BAD = "✗"
WARN = "⚠"
SKIP = "-"

# ANSI colors, enabled in main() when stdout is a TTY (or FORCE_COLOR is set),
# disabled by --no-color or NO_COLOR.
_COLOR = False
_GREEN, _RED, _YELLOW, _BOLD = "\033[32m", "\033[31m", "\033[33m", "\033[1m"


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _paint(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _COLOR else text


def _colorize(line: str) -> str:
    """Color status glyphs and a few key phrases in a line of output."""
    if not _COLOR:
        return line
    line = line.replace(OK, _paint(OK, _GREEN)).replace(BAD, _paint(BAD, _RED))
    line = line.replace(WARN, _paint(WARN, _YELLOW))
    for word in ("missing", "incomplete", "unreadable", "denied", "not found", "error"):
        line = re.sub(
            rf"(?<=[{BAD}{WARN}]\033\[0m )({word})", lambda m: _paint(m.group(1), _RED), line
        )
    if line.startswith("Problems"):
        line = _paint(line, _RED + _BOLD)
    elif "Everything looks good" in line:
        line = _paint(line, _GREEN + _BOLD)
    return line


def print_line(line: str = "") -> None:
    """Print one line, colorizing ✓/✗/⚠ when colors are enabled."""
    print(_colorize(line))


def configure_color(no_color: bool = False) -> None:
    """Enable ANSI colors for print_line: TTY or FORCE_COLOR, unless NO_COLOR/no_color."""
    global _COLOR
    if no_color or os.environ.get("NO_COLOR"):
        _COLOR = False
    elif os.environ.get("FORCE_COLOR"):
        _COLOR = True
    else:
        _COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)())


def fmt_bytes(n: int | None) -> str:
    """Human-readable size: 1536 -> '1.5 KB'."""
    if n is None:
        return "?"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def mask_account(arn: str) -> str:
    """Hide all but the last 4 digits of the AWS account id in an ARN."""
    return re.sub(r"::(\d{12}):", lambda m: f"::********{m.group(1)[-4:]}:", arn)


def split_s3(url: str) -> tuple[str, str]:
    """'s3://bucket/a/b/' -> ('bucket', 'a/b/')."""
    if not url.startswith("s3://"):
        raise ValueError(f"Not an S3 URL: {url}")
    rest = url[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


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
# Local filesystem checks
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
    """Existence, ownership, mode, read/write/traverse access, and disk free for a dir."""
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


def dir_bytes(path: Path) -> int:
    """Recursive on-disk size (matches what ``s5cmd cp prefix/*`` downloads)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def find_other_caches(cache_base: Path, known: set[str]) -> list[tuple[str, int]]:
    """Manifest-bearing slipstream caches under cache_base not in ``known``: [(name, bytes)]."""
    out: list[tuple[str, int]] = []
    if not cache_base.is_dir() or not os.access(cache_base, os.R_OK | os.X_OK):
        return out
    for p in sorted(cache_base.iterdir()):
        if p.name in known or not p.is_dir():
            continue
        if (p / MANIFEST_FILE).exists():
            try:
                out.append((p.name, dir_bytes(p)))
            except OSError:
                out.append((p.name, 0))
    return out


@dataclass
class CacheDirInfo:
    path: str
    source: str  # where the setting came from
    env_var: str | None
    access: DirAccess


def resolve_cache_dir() -> CacheDirInfo:
    """SLIPSTREAM_CACHE_DIR if set, else slipstream's default (~/.slipstream)."""
    env_val = os.environ.get(CACHE_DIR_ENV_VAR)
    if env_val:
        path = Path(env_val).expanduser()
        source = f"{CACHE_DIR_ENV_VAR} environment variable"
    else:
        path = DEFAULT_CACHE_DIR
        source = "slipstream default (~/.slipstream)"
    return CacheDirInfo(path=str(path), source=source, env_var=env_val, access=inspect_dir(path))


# --------------------------------------------------------------------------- #
# S3 checks
# --------------------------------------------------------------------------- #


@dataclass
class S3Info:
    s5cmd_path: str | None = None
    s5cmd_version: str | None = None
    s5cmd_error: str | None = None
    credentials_found: bool = False
    credentials_method: str | None = None
    profile: str | None = None
    region: str | None = None
    identity_arn: str | None = None  # account id masked
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
    """s5cmd availability, AWS credentials, caller identity, and a list on remote_base."""
    info = S3Info(bucket_url=remote_base, checked=check_remote)

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
    session = None
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

    if not info.credentials_found or session is None:
        info.identity_error = info.identity_error or "no AWS credentials found"
        info.bucket_readable = False
        info.bucket_error = "no credentials"
        return info

    try:
        sts = session.client("sts", config=_boto_config())
        info.identity_arn = mask_account(sts.get_caller_identity()["Arn"])
    except Exception as exc:
        info.identity_error = f"{type(exc).__name__}: {exc}"

    try:
        bucket, prefix = split_s3(remote_base)
        s3 = session.client("s3", config=_boto_config(), endpoint_url=endpoint_url)
        s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        info.bucket_readable = True
    except Exception as exc:
        info.bucket_readable = False
        info.bucket_error = f"{type(exc).__name__}: {exc}"
    return info


def remote_listing(
    remote: str, *, endpoint_url: str | None = None, profile: str | None = None
) -> tuple[int, int]:
    """Return (num_files, total_bytes) for an S3 prefix. Raises on error."""
    bucket, prefix = split_s3(remote.rstrip("/") + "/")
    s3 = _boto_session(profile).client("s3", config=_boto_config(), endpoint_url=endpoint_url)
    n, total = 0, 0
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
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


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def collect_status(
    *, check_remote_access: bool = True, endpoint_url: str | None = None
) -> dict[str, Any]:
    cache = resolve_cache_dir()
    s3 = check_s3(
        DEFAULT_REMOTE_CACHE_BASE, check_remote=check_remote_access, endpoint_url=endpoint_url
    )
    caches = find_other_caches(Path(cache.path), set())
    return {
        "slipstream_version": __version__,
        "python": _platform.python_version(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "cache": asdict(cache),
        "s3": asdict(s3),
        "caches": [{"name": n, "bytes": b} for n, b in caches],
    }


def problems(status: dict[str, Any]) -> list[str]:
    """Human-readable list of things that will block training."""
    out: list[str] = []
    acc = status["cache"]["access"]
    cache_path = status["cache"]["path"]
    if not acc["exists"]:
        if acc["can_create"]:
            out.append(f"Cache dir {cache_path} does not exist yet (created on first use).")
        else:
            out.append(
                f"Cache dir {cache_path} does not exist and cannot be created "
                "(parent not writable)."
            )
    else:
        if not acc["readable"] or not acc["traversable"]:
            out.append(
                f"No read access to cache dir {cache_path} "
                f"(owner {acc['owner']}, mode {acc['mode']})."
            )
        if not acc["writable"]:
            out.append(
                f"No write access to cache dir {cache_path}; "
                "existing caches are usable but new ones cannot be written."
            )
    s3 = status["s3"]
    if s3["s5cmd_error"]:
        out.append(f"s5cmd not usable: {s3['s5cmd_error']}  (fix: uv tool install s5cmd)")
    if not s3["credentials_found"]:
        out.append(
            "No AWS credentials found (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            "or configure ~/.aws/credentials)."
        )
    elif s3["checked"] and s3["bucket_readable"] is False:
        out.append(f"Cannot list {s3['bucket_url']}: {s3['bucket_error']}")
    return out


def _hard_problems(status: dict[str, Any]) -> list[str]:
    return [m for m in problems(status) if "created on first use" not in m]


def _mark(ok: bool | None) -> str:
    if ok is None:
        return SKIP
    return OK if ok else BAD


def print_status(status: dict[str, Any]) -> None:
    p = print_line
    c = status["cache"]
    a = c["access"]
    s3 = status["s3"]

    p(f"slipstream {status['slipstream_version']}  ·  python {status['python']}")
    p(f"{status['user']}@{status['host']}")
    p()

    p("Cache directory")
    p(f"  path        {c['path']}" + (f"  -> {a['resolved']}" if a["is_symlink"] else ""))
    p(f"  source      {c['source']}")
    p(f"  {CACHE_DIR_ENV_VAR}  {c['env_var'] or '(not set)'}")
    if a["exists"]:
        p(f"  exists      {OK}  owner {a['owner']}:{a['group']}  mode {a['mode']}")
        p(f"  read        {_mark(a['readable'] and a['traversable'])}")
        p(f"  write       {_mark(a['writable'])}")
    else:
        p(f"  exists      {BAD}  (not created yet; can create: {_mark(a['can_create'])})")
    if a["free_bytes"] is not None:
        p(f"  disk free   {fmt_bytes(a['free_bytes'])} of {fmt_bytes(a['total_bytes'])}")
    if not c["env_var"]:
        p(
            "  lab users:  run `visionlab-datasets status` "
            "(sets the per-platform cache dir and shows lab datasets)"
        )
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

    if status["caches"]:
        p("Slipstream caches in cache dir")
        w = max(len(o["name"]) for o in status["caches"]) + 2
        for o in status["caches"]:
            p(f"  {o['name']:<{w}}{fmt_bytes(o['bytes']):>10}")
    else:
        p("Slipstream caches in cache dir: none")
    p()

    probs = problems(status)
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
    return 1 if _hard_problems(status) else 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slipstream",
        description=(
            "slipstream plumbing checks: cache dir, permissions, S3 access. "
            "For lab datasets use `visionlab-datasets status|sync`."
        ),
    )
    parser.add_argument("--version", action="version", version=f"slipstream {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("status", "config"):
        sp = sub.add_parser(
            name,
            help="Show cache dir, permissions, S3 access, and caches on disk"
            + (" (alias for status)" if name == "config" else ""),
        )
        sp.add_argument(
            "--no-remote", action="store_true", help="Skip network checks (S3 identity/listing)"
        )
        sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
        sp.add_argument("--json", action="store_true", help="Machine-readable output")
        sp.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
        sp.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_color(getattr(args, "no_color", False))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


# Backward-compatible private aliases (visionlab-datasets <= 0.8.0 used these).
_fmt_bytes = fmt_bytes
_dir_bytes = dir_bytes
_mask_account = mask_account
_split_s3 = split_s3
_configure_color = configure_color
_print = print_line
_problems = problems

__all__ = [
    "OK",
    "BAD",
    "WARN",
    "SKIP",
    "MANIFEST_FILE",
    "DEFAULT_REMOTE_CACHE_BASE",
    "DirAccess",
    "S3Info",
    "CacheDirInfo",
    "inspect_dir",
    "dir_bytes",
    "find_other_caches",
    "resolve_cache_dir",
    "check_s3",
    "remote_listing",
    "fmt_bytes",
    "mask_account",
    "split_s3",
    "configure_color",
    "print_line",
    "collect_status",
    "problems",
    "print_status",
    "cmd_status",
    "build_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
