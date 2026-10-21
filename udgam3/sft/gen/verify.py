"""Verifiers: answer matching (numeric / math-verify / exact strings), sandboxed Python execution, JSON-schema checks.

math-verify is not in the harness env; it is in the data factory's venv, so `MathVerify` keeps one helper process
(`datafactory/.venv/bin/python -I mv_worker.py`) and talks to it line by line. If that venv is missing it falls back
to numeric / normalised-string comparison only.
"""
from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from fractions import Fraction
from pathlib import Path

from common import GEN, REPO

FACTORY_PY = REPO / "datafactory" / ".venv" / "bin" / "python"

# ------------------------------------------------------------------------------------------------ answers
NUM_RX = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+")


def to_number(s: str):
    """Parse '1,234.5', '$12', '45%', '3/4', '-7', '2.50 dollars' -> float, else None."""
    if s is None:
        return None
    t = str(s).strip().lower()
    t = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "").replace("\\!", "")
    t = re.sub(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", t)
    t = re.sub(r"^(?:x|y|n|a|answer)\s*=\s*", "", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Sc")          # any currency symbol
    t = re.sub(r"[a-z°\s]+$", "", t).strip()
    t = re.sub(r"^[a-z\s]+", "", t).strip()
    if not re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+", t):
        return None
    t = t.replace(",", "")
    try:
        return float(Fraction(t)) if "/" in t else float(t)
    except (ValueError, ZeroDivisionError):
        return None


def num_equal(a, b, rel=1e-6) -> bool:
    x, y = to_number(a), to_number(b)
    if x is None or y is None:
        return False
    return math.isclose(x, y, rel_tol=rel, abs_tol=1e-9)


def norm_ans(s: str) -> str:
    t = str(s or "").strip().lower()
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\boxed\{(.*)\}", r"\1", t)
    t = t.replace("$", "").replace("\\,", "").replace("\\!", "").replace("\\ ", "").replace("^\\circ", "").replace("°", "")
    t = re.sub(r"\s+", "", t)
    return t.rstrip(".")


def logic_equal(pred: str, gold: str) -> bool:
    f = lambda s: [w for w in re.split(r"[\s,;]+(?:and\s+)?", str(s).strip().lower().rstrip(".")) if w]
    return f(pred) == f(gold)


def exact_output_equal(pred: str, gold: str) -> bool:
    f = lambda s: "\n".join(" ".join(l.split()) for l in str(s).strip().splitlines())
    return f(pred) == f(gold)


def answer_in_text(ans: str, text: str) -> bool:
    """The final answer is stated in the text (normalised substring, or the same number appears)."""
    if not ans or not text:
        return False
    a, t = norm_ans(ans), norm_ans(text)
    if a and a in t:
        return True
    x = to_number(ans)
    if x is not None:
        for m in NUM_RX.finditer(text.replace("\\frac{", "").replace("}{", "/")):
            y = to_number(m.group(0))
            if y is not None and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9):
                return True
    return False


def last_number(text: str):
    nums = [to_number(m.group(0)) for m in NUM_RX.finditer(text or "")]
    nums = [n for n in nums if n is not None]
    return nums[-1] if nums else None


class MathVerify:
    """Persistent math-verify helper process (factory venv). verify(pred, gold) -> bool | None (None = unavailable)."""

    def __init__(self):
        self.proc = None
        self.lock = asyncio.Lock()
        self.available = FACTORY_PY.exists()

    async def _start(self):
        self.proc = await asyncio.create_subprocess_exec(
            str(FACTORY_PY), "-I", str(GEN / "mv_worker.py"), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def verify(self, pred: str, gold: str):
        if num_equal(pred, gold) or (norm_ans(pred) and norm_ans(pred) == norm_ans(gold)):
            return True
        if not self.available:
            return None
        async with self.lock:
            try:
                if self.proc is None or self.proc.returncode is not None:
                    await self._start()
                self.proc.stdin.write((json.dumps({"pred": str(pred), "gold": str(gold)}) + "\n").encode())
                await self.proc.stdin.drain()
                line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=40)
                return bool(json.loads(line)["ok"])
            except Exception:                      # noqa: BLE001
                if self.proc:
                    self.proc.kill()
                self.proc = None
                return False

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()


# ------------------------------------------------------------------------------------------------ python sandbox
ALLOWED_IMPORTS = {"math", "fractions", "decimal", "itertools", "functools", "statistics", "datetime", "calendar",
                   "collections", "string", "re", "heapq", "bisect", "operator", "cmath", "textwrap", "json"}
BANNED_NAMES = {"exec", "eval", "compile", "open", "input", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "breakpoint", "help", "exit", "quit", "memoryview", "os", "sys", "subprocess",
                "socket", "shutil", "pathlib", "importlib", "builtins"}
SANDBOX = '(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n(allow file-write* (literal "/dev/null"))\n'
LIMITS = r'''
import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
for lim, val in (("RLIMIT_AS", 1 << 30), ("RLIMIT_FSIZE", 0), ("RLIMIT_NPROC", 0)):
    try:
        resource.setrlimit(getattr(resource, lim), (val, val))
    except (ValueError, OSError, AttributeError):
        pass
src = sys.stdin.read()
exec(compile(src, "<tool>", "exec"), {"__name__": "__main__"})
'''


def check_code(code: str, allowed_imports=ALLOWED_IMPORTS) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in allowed_imports:
                    return False, f"import not allowed: {a.name}"
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in allowed_imports:
                return False, f"import not allowed: {n.module}"
        elif isinstance(n, ast.Name) and n.id in BANNED_NAMES:
            return False, f"name not allowed: {n.id}"
        elif isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return False, "dunder attribute"
    return True, ""


def run_python(code: str, timeout: float = 6.0, stdin_text: str = "") -> dict:
    """Run code in a fresh, limited process (macOS: sandbox-exec, no network, no file writes). Code is passed on
    stdin to a tiny runner; any stdin for the program itself is not supported (input() is banned)."""
    ok, why = check_code(code)
    if not ok:
        return {"ok": False, "stdout": "", "stderr": why, "rejected": True}
    argv = [sys.executable, "-I", "-S", "-c", LIMITS]
    if sys.platform == "darwin" and os.path.exists("/usr/bin/sandbox-exec"):
        argv = ["/usr/bin/sandbox-exec", "-p", SANDBOX] + argv
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run(argv, input=code, capture_output=True, text=True, timeout=timeout, cwd=d,
                               env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "timeout", "rejected": False}
    return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr[-600:], "rejected": False}


async def arun_python(code: str, timeout: float = 6.0) -> dict:
    return await asyncio.to_thread(run_python, code, timeout)


# ------------------------------------------------------------------------------------------------ JSON schema
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict,
         "null": type(None)}


def schema_ok(schema: dict) -> tuple[bool, str]:
    """A tool's `parameters` schema: object with properties; every property has a known type; required ⊆ properties."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return False, "parameters must be an object schema"
    props = schema.get("properties")
    if not isinstance(props, dict):
        return False, "no properties"
    for k, v in props.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", k):
            return False, f"bad property name {k}"
        if not isinstance(v, dict) or v.get("type") not in TYPES:
            return False, f"property {k} has no valid type"
        if not v.get("description"):
            return False, f"property {k} has no description"
        if v["type"] == "array" and not isinstance(v.get("items"), dict):
            return False, f"array {k} has no items"
        if "enum" in v and (not isinstance(v["enum"], list) or not v["enum"]):
            return False, f"bad enum on {k}"
    req = schema.get("required", [])
    if not isinstance(req, list) or any(r not in props for r in req):
        return False, "required lists unknown properties"
    return True, ""


def value_ok(v, s: dict, path: str = "") -> tuple[bool, str]:
    t = s.get("type")
    py = TYPES.get(t)
    if py is None:
        return False, f"{path}: unknown type"
    if t in ("integer", "number") and isinstance(v, bool):
        return False, f"{path}: bool for {t}"
    if t == "integer" and isinstance(v, float) and v.is_integer():
        v = int(v)
    if not isinstance(v, py):
        return False, f"{path}: expected {t}"
    if "enum" in s and v not in s["enum"]:
        return False, f"{path}: not in enum"
    if t == "array":
        for i, x in enumerate(v):
            ok, why = value_ok(x, s.get("items") or {"type": "string"}, f"{path}[{i}]")
            if not ok:
                return False, why
    if t == "object" and isinstance(s.get("properties"), dict):
        return args_ok(v, s)
    return True, ""


def args_ok(args, schema: dict) -> tuple[bool, str]:
    if not isinstance(args, dict):
        return False, "arguments not an object"
    props = schema.get("properties") or {}
    for r in schema.get("required", []):
        if r not in args:
            return False, f"missing required {r}"
    for k, v in args.items():
        if k not in props:
            return False, f"unknown argument {k}"
        ok, why = value_ok(v, props[k], k)
        if not ok:
            return False, why
    return True, ""


def flat_values(x) -> list:
    if isinstance(x, dict):
        return [y for v in x.values() for y in flat_values(v)]
    if isinstance(x, list):
        return [y for v in x for y in flat_values(v)]
    return [x]
