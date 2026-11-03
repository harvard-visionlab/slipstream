"""Which tests to run for a change, and whether the full (slow) tier is needed.

    uv run python scripts/affected_tests.py [BASE]      # BASE defaults to the latest tag

Lists the changed files (BASE..HEAD plus uncommitted), the test files that reference a changed module
(by dotted path or by a public name it defines), and the tier to run before tagging:

- core  (always): ``pytest -m "not slow"``                       ~50 s on machina
- full  (FULL_TRIGGERS touched): ``pytest``                       ~4.5 min on machina
- the speed gate runs before every tag regardless (CLAUDE.md "Release gate").
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Code the slow tier exercises (cache build / conversion / integrity / sync), native code, dependencies.
FULL_TRIGGERS = (
    "slipstream/cache.py", "slipstream/dataset.py", "slipstream/readers/", "slipstream/s3_sync.py",
    "slipstream/identity.py", "slipstream/backends/", "libslipstream/", "hatch_build.py",
    "pyproject.toml", "uv.lock", "tests/conftest.py",
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def changed_files(base: str) -> list[str]:
    out = _git("diff", "--name-only", f"{base}...HEAD") + _git("diff", "--name-only", "HEAD")
    out += _git("ls-files", "--others", "--exclude-standard")
    return sorted({f for f in out.split() if f})


def _needles(path: str) -> set[str]:
    """Strings whose presence in a test file means it uses the module at ``path``."""
    mod = path[:-3].replace("/", ".").removesuffix(".__init__")
    needles = {mod}
    try:
        tree = ast.parse((ROOT / path).read_text())
    except (OSError, SyntaxError):
        return needles
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("__"):
            needles.add(node.name)
    return needles


def slow_files() -> set[str]:
    """SLOW_FILES from tests/conftest.py (parsed, so this script needs no pytest)."""
    tree = ast.parse((ROOT / "tests" / "conftest.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SLOW_FILES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    return set()


def affected_tests(files: list[str], include_slow: bool = False) -> list[str]:
    """Test files referencing a changed module, plus changed test files. Slow-tier files only when
    ``include_slow`` or when the slow file itself changed."""
    tests = {f for f in files if f.startswith("tests/test_") and (ROOT / f).exists()}
    slow = set() if include_slow else slow_files()
    needles: set[str] = set()
    for f in files:
        if f.startswith("slipstream/") and f.endswith(".py") and (ROOT / f).exists():
            needles |= _needles(f)
    if needles:
        pat = re.compile("|".join(rf"\b{re.escape(n)}\b" for n in sorted(needles, key=len, reverse=True)))
        for t in sorted((ROOT / "tests").glob("test_*.py")):
            if t.name not in slow and pat.search(t.read_text()):
                tests.add(str(t.relative_to(ROOT)))
    return sorted(tests)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else _git("describe", "--tags", "--abbrev=0").strip()
    files = changed_files(base)
    full = [f for f in files if f.startswith(FULL_TRIGGERS)]
    tests = affected_tests(files, include_slow=bool(full))
    print(f"changes since {base}: {len(files)} file(s)")
    for f in files:
        print(f"  {f}{'   [full-tier trigger]' if f in full else ''}")
    print("\ntargeted:" + ("\n  uv run pytest " + " ".join(tests) if tests else " (no test references a changed module)"))
    print("\nbefore tagging:")
    print('  core tier:  uv run pytest tests/ -m "not slow"')
    if full:
        print("  full tier:  uv run pytest tests/          (required: full-tier triggers changed)")
    print("  speed gate: uv run python -m benchmarks.release_gate --host machina --append-md --wait 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
