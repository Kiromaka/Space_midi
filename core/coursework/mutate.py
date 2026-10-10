"""A small mutation testing tool, written for the coursework (no third-party mutation library).

It parses a Python module with ``ast``, finds every place where one of the
mutation operators below applies, and makes one mutant per place: a copy of the
module with exactly one change. The test suite is run once per mutant:

* the tests fail       -> the mutant is KILLED (the tests noticed the change);
* the tests pass       -> the mutant SURVIVED (a gap in the tests, or an equivalent mutant);
* the tests time out   -> counted as killed (e.g. the mutant loops forever).

Mutation operators:

* AOR  arithmetic:    + <-> -,  * -> /,  / -> *,  // -> *,  % -> //
* BOR  bitwise/shift: << <-> >>,  & <-> |,  ^ -> |   (also inside augmented assignments: >>= -> <<=)
* ROR  relational:    == <-> !=,  < -> <= / >,  <= -> < / >=,  > -> >= / <,  >= -> > / <=,  in <-> not in
* LCR  logical:       and <-> or
* UOD  "not" deleted: not x -> x
* CRP  int constant:  n -> n + 1

The module under test is never modified in place. Every worker gets its own copy of
the package in a temporary folder and the tests run there, so an interrupted run
cannot leave a mutant behind in the repository.

Usage (from the core folder):

    uv run python coursework/mutate.py                 # all mutants of smf.py vs tests/test_smf.py
    uv run python coursework/mutate.py --list          # only print the mutants
    uv run python coursework/mutate.py --only 12,40    # re-run selected mutants
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]

BINARY = {  # ast operator -> (source token, replacement tokens)
    ast.Add: ("+", ["-"]),
    ast.Sub: ("-", ["+"]),
    ast.Mult: ("*", ["/"]),
    ast.Div: ("/", ["*"]),
    ast.FloorDiv: ("//", ["*"]),
    ast.Mod: ("%", ["//"]),
    ast.LShift: ("<<", [">>"]),
    ast.RShift: (">>", ["<<"]),
    ast.BitAnd: ("&", ["|"]),
    ast.BitOr: ("|", ["&"]),
    ast.BitXor: ("^", ["|"]),
}
COMPARE = {
    ast.Eq: ("==", ["!="]),
    ast.NotEq: ("!=", ["=="]),
    ast.Lt: ("<", ["<=", ">"]),
    ast.LtE: ("<=", ["<", ">="]),
    ast.Gt: (">", [">=", "<"]),
    ast.GtE: (">=", [">", "<="]),
    ast.In: ("in", ["not in"]),
    ast.NotIn: ("not in", ["in"]),
}
BOOLEAN = {ast.And: ("and", ["or"]), ast.Or: ("or", ["and"])}
OPERATOR_GROUP = {
    **{op: "AOR" for op in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)},
    **{op: "BOR" for op in (ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)},
}


@dataclass
class Mutant:
    id: int
    line: int
    operator: str  # AOR, BOR, ROR, LCR, UOD, CRP
    original: str  # the source text that is replaced
    replacement: str
    start: int  # byte offsets in the module source
    end: int
    before: str = ""  # whole source line, before and after the change
    after: str = ""
    status: str = ""  # killed / survived / timeout / error
    killed_by: str = ""
    seconds: float = 0.0


# ---------------------------------------------------------------- finding mutants


class _Finder(ast.NodeVisitor):
    """Collects (start, end, original, replacement, operator) for every mutation point."""

    def __init__(self, source: bytes):
        self.source = source
        self.line_starts = [0]
        for i, byte in enumerate(source):
            if byte == 0x0A:
                self.line_starts.append(i + 1)
        self.found: list[tuple[int, int, str, str, str]] = []

    # -- positions

    def offset(self, line: int, col: int) -> int:
        return self.line_starts[line - 1] + col  # ast columns are UTF-8 byte offsets

    def start(self, node: ast.AST) -> int:
        return self.offset(node.lineno, node.col_offset)

    def end(self, node: ast.AST) -> int:
        return self.offset(node.end_lineno, node.end_col_offset)

    def token_between(self, left: ast.AST, right: ast.AST, token: str, group: str, replacements: list[str]) -> None:
        """Mutate the operator token written between two nodes (skipped if it can't be found exactly)."""
        lo, hi = self.end(left), self.start(right)
        segment = self.source[lo:hi].decode("utf-8")
        stripped = segment.strip(" \t\r\n()\\")
        if " ".join(stripped.split()) != token:
            return
        at = lo + len(segment[: segment.index(stripped)].encode("utf-8"))
        for new in replacements:
            self.found.append((at, at + len(stripped.encode("utf-8")), stripped, new, group))

    # -- skip what is not code: docstrings, annotations, f-strings

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(default)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for stmt in node.body:
            self.visit(stmt)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        pass

    # -- mutation points

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) in BINARY:
            token, replacements = BINARY[type(node.op)]
            self.token_between(node.left, node.right, token, OPERATOR_GROUP[type(node.op)], replacements)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if type(node.op) in BINARY:
            token, replacements = BINARY[type(node.op)]
            group = OPERATOR_GROUP[type(node.op)]
            self.token_between(node.target, node.value, token + "=", group, [r + "=" for r in replacements])
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        for i, op in enumerate(node.ops):
            if type(op) in COMPARE:
                token, replacements = COMPARE[type(op)]
                self.token_between(operands[i], operands[i + 1], token, "ROR", replacements)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        token, replacements = BOOLEAN[type(node.op)]
        for left, right in zip(node.values, node.values[1:]):
            self.token_between(left, right, token, "LCR", replacements)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            lo, hi = self.start(node), self.start(node.operand)
            text = self.source[lo:hi].decode("utf-8")
            if text.split() == ["not"]:
                self.found.append((lo, hi, text, "", "UOD"))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if type(node.value) is not int:  # bool is a subclass of int: skip it, and strings/bytes
            return
        lo, hi = self.start(node), self.end(node)
        text = self.source[lo:hi].decode("utf-8")
        self.found.append((lo, hi, text, _format_like(text, node.value + 1), "CRP"))


def _format_like(text: str, value: int) -> str:
    """Write an int the way the original literal was written (0x7F stays hex, same width)."""
    if text[:2].lower() == "0x":
        digits = text[2:].replace("_", "")
        out = f"{value:0{len(digits)}x}"
        return text[:2] + (out.upper() if digits.upper() == digits else out)
    return str(value)


def find_mutants(source: bytes) -> list[Mutant]:
    finder = _Finder(source)
    finder.visit(ast.parse(source))
    mutants = []
    for i, (lo, hi, original, replacement, group) in enumerate(sorted(set(finder.found)), start=1):
        line = source.count(b"\n", 0, lo) + 1
        line_lo = source.rfind(b"\n", 0, lo) + 1
        line_hi = source.find(b"\n", hi)
        line_hi = len(source) if line_hi < 0 else line_hi
        before = source[line_lo:line_hi].decode("utf-8").rstrip()
        after = (source[line_lo:lo] + replacement.encode("utf-8") + source[hi:line_hi]).decode("utf-8").rstrip()
        mutants.append(Mutant(i, line, group, original, replacement, lo, hi, before.strip(), after.strip()))
    return mutants


def apply(source: bytes, mutant: Mutant) -> bytes:
    return source[: mutant.start] + mutant.replacement.encode("utf-8") + source[mutant.end :]


# ---------------------------------------------------------------- running the tests


def make_sandbox(root: Path) -> Path:
    """A copy of the package, the tests and the pytest config in a fresh temporary folder."""
    box = Path(tempfile.mkdtemp(prefix="mutant-", dir=root))
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(CORE / "spacemidi", box / "spacemidi", ignore=ignore)
    shutil.copytree(CORE / "tests", box / "tests", ignore=ignore)
    shutil.copy2(CORE / "pyproject.toml", box / "pyproject.toml")
    return box


def run_tests(box: Path, tests: str, timeout: float) -> tuple[str, str, float]:
    """Run pytest inside a sandbox; returns (status, first failing test, seconds)."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")  # never reuse a stale .pyc of another mutant
    cmd = [sys.executable, "-m", "pytest", tests, "-x", "-q", "-rfE", "-p", "no:cacheprovider", "--no-header"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=box, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "", time.perf_counter() - t0
    seconds = time.perf_counter() - t0
    if proc.returncode == 0:
        return "survived", "", seconds
    failed = re.search(r"^(?:FAILED|ERROR) (.+?)(?: - .*)?$", proc.stdout, re.MULTILINE)
    name = failed.group(1).split("::", 1)[-1] if failed else ""
    return ("killed" if proc.returncode == 1 else "error"), name, seconds


def check_isolation(box: Path, module: str) -> None:
    """Make sure the tests in a sandbox import the sandbox copy, not the repository."""
    dotted = module.removesuffix(".py").replace("/", ".")
    out = subprocess.run(
        [sys.executable, "-c", f"import {dotted} as m; print(m.__file__)"],
        cwd=box, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    where = Path(out.stdout.strip())
    where = where if where.is_absolute() else box / where
    if out.returncode != 0 or not where.resolve().is_relative_to(box.resolve()):
        sys.exit(f"sandbox is not isolated: {dotted} imported from {out.stdout.strip() or out.stderr}")


# ---------------------------------------------------------------- report


def write_report(path: Path, args, mutants: list[Mutant], baseline: float, wall: float) -> None:
    counts = {s: sum(m.status == s for m in mutants) for s in ("killed", "timeout", "error", "survived")}
    detected = counts["killed"] + counts["timeout"] + counts["error"]
    score = detected / len(mutants) if mutants else 0.0
    lines = [
        f"# Mutation testing: {args.module}",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Tests: `{args.tests}` (baseline run {baseline:.2f} s, all passed)",
        f"- Mutants: {len(mutants)}; killed {counts['killed']}, timeout {counts['timeout']}, "
        f"error {counts['error']}, survived {counts['survived']}",
        f"- Mutation score: {detected}/{len(mutants)} = {score:.1%}",
        f"- Wall time: {wall:.1f} s",
        "",
        "## Survived mutants",
        "",
        "| # | Line | Op | Original | Mutant |",
        "|---|---|---|---|---|",
    ]
    for m in mutants:
        if m.status == "survived":
            lines.append(f"| {m.id} | {m.line} | {m.operator} | `{m.before}` | `{m.after}` |")
    lines += ["", "## All mutants", "", "| # | Line | Op | Change | Status | Killed by |", "|---|---|---|---|---|---|"]
    for m in mutants:
        change = f"`{m.original}` → `{m.replacement}`" if m.replacement else f"`{m.original.strip()}` deleted"
        lines.append(f"| {m.id} | {m.line} | {m.operator} | {change} | {m.status} | {m.killed_by} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps({"module": args.module, "tests": args.tests, "score": score, "counts": counts,
                    "mutants": [asdict(m) for m in mutants]}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--module", default="spacemidi/midi/smf.py", help="module to mutate, relative to core/")
    ap.add_argument("--tests", default="tests/test_smf.py", help="tests to run, relative to core/")
    ap.add_argument("--list", action="store_true", help="print the mutants and exit")
    ap.add_argument("--only", help="comma-separated mutant numbers to run")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel workers")
    ap.add_argument("--out", default="coursework/runs", help="folder for the report, relative to core/")
    args = ap.parse_args(argv)

    source = (CORE / args.module).read_bytes()
    mutants = find_mutants(source)
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        mutants = [m for m in mutants if m.id in wanted]
    if args.list:
        for m in mutants:
            print(f"{m.id:4} line {m.line:4}  {m.operator}  {m.before}\n{'':20}->  {m.after}")
        print(f"{len(mutants)} mutants")
        return 0

    tmp_root = Path(tempfile.mkdtemp(prefix="mutation-"))
    try:
        boxes: queue.Queue[Path] = queue.Queue()
        for _ in range(max(1, args.jobs)):
            boxes.put(make_sandbox(tmp_root))
        first = boxes.get()
        check_isolation(first, args.module)
        status, failed, baseline = run_tests(first, args.tests, timeout=600)
        boxes.put(first)
        if status != "survived":
            sys.exit(f"the tests must pass on the original code first ({status}: {failed})")
        timeout = max(10.0, baseline * 10)
        print(f"{len(mutants)} mutants, baseline {baseline:.2f} s, timeout {timeout:.0f} s, {args.jobs} jobs")

        def run(m: Mutant) -> Mutant:
            box = boxes.get()
            try:
                (box / args.module).write_bytes(apply(source, m))
                m.status, m.killed_by, m.seconds = run_tests(box, args.tests, timeout)
                (box / args.module).write_bytes(source)
            finally:
                boxes.put(box)
            mark = "SURVIVED" if m.status == "survived" else m.status
            print(f"[{m.id:3}/{len(mutants)}] line {m.line:4} {m.operator}  {m.after:60.60}  {mark} {m.killed_by}")
            return m

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            mutants = list(pool.map(run, mutants))
        wall = time.perf_counter() - t0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_dir = CORE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"mutation-{datetime.now():%Y%m%d-%H%M%S}.md"
    write_report(report, args, mutants, baseline, wall)
    survived = [m for m in mutants if m.status == "survived"]
    detected = len(mutants) - len(survived)
    print(f"\nkilled {detected} of {len(mutants)} mutants, survived {len(survived)}; "
          f"mutation score {detected / max(1, len(mutants)):.1%}")
    print(f"report: {report.relative_to(CORE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
