"""Run the Python a model writes in tool mode (Laghu RL rollouts and evals), safely. Reuses Lilavati's structure check
(release/lilavati/code/lilavati/safe_exec.py: plain arithmetic, loops, lists/dicts, `math` only; no other imports, no
open/exec/eval/getattr, no private attributes) but for print-style scripts, as in the SFT tool rows: the tool returns
what the program prints. Runs in a fresh Python process with CPU (2 s), memory (512 MB), file and process limits and a
3 s wall clock. run(code) -> (ok, text): text = stdout (first 400 chars) or a short "Error: ..." line.
"""
import importlib.util, json, os, subprocess, sys, tempfile

_P = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "release", "lilavati", "code", "lilavati", "safe_exec.py"))   # lexical: the VM links udgam3/rl
_spec = importlib.util.spec_from_file_location("lilavati_safe_exec", _P)
_le = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_le)

RUNNER = r'''
import io, json, math, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    for lim, val in (("RLIMIT_AS", 512 << 20), ("RLIMIT_FSIZE", 0), ("RLIMIT_NOFILE", 16), ("RLIMIT_NPROC", 0)):
        if hasattr(resource, lim):
            try:
                resource.setrlimit(getattr(resource, lim), (val, val))
            except (ValueError, OSError):
                pass
except ImportError:
    pass
code = sys.stdin.read()
import builtins
SAFE = {k: getattr(builtins, k) for k in ("abs", "round", "min", "max", "sum", "int", "float", "len", "range", "pow",
        "divmod", "sorted", "list", "tuple", "dict", "set", "enumerate", "zip", "any", "all", "str", "bool", "reversed",
        "print", "True", "False", "None", "ZeroDivisionError", "ValueError", "AssertionError")}
SAFE["__import__"] = lambda name, *a, **k: math if name == "math" else (_ for _ in ()).throw(ImportError(name))
buf = io.StringIO(); real = sys.stdout; sys.stdout = buf
res = {"ok": True}
try:
    exec(compile(code, "<tool>", "exec"), {"__builtins__": SAFE, "math": math})
except BaseException as e:
    res = {"ok": False, "err": type(e).__name__}
sys.stdout = real
res["out"] = buf.getvalue()[:400]
real.write("\n" + json.dumps(res))
'''


def check(code):
    ok, why = _le.check_program(code)
    return (True, "") if (ok or why == "no function defined") else (False, why)


def run(code, timeout=3.0):
    ok, why = check(code)
    if not ok:
        return False, f"Error: {why}"
    argv = [sys.executable, "-I", "-S", "-c", RUNNER]
    if sys.platform == "darwin" and os.path.exists("/usr/bin/sandbox-exec"):
        argv = ["/usr/bin/sandbox-exec", "-p", _le.SANDBOX] + argv
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run(argv, input=code, capture_output=True, text=True, timeout=timeout, cwd=d, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False, "Error: timeout"
    try:
        res = json.loads((r.stdout.strip().splitlines() or ["null"])[-1])
    except (json.JSONDecodeError, IndexError):
        return False, "Error: crashed"
    if not res or not res.get("ok"):
        return False, f"Error: {(res or {}).get('err', 'crashed')}"
    return True, res["out"].strip() or "(no output)"


if __name__ == "__main__":
    tests = [("print(200 * 142 + 15000)", True), ("x = [i*i for i in range(5)]\nprint(sum(x))", True),
             ("import os\nprint(os.listdir('/'))", False), ("open('/etc/passwd').read()", False),
             ("print(1/0)", False), ("while True:\n    pass", False), ("print(10**10**10)", False),
             ("import math\nprint(math.sqrt(144))", True), ("print(().__class__)", False)]
    for code, want in tests:
        ok, out = run(code)
        print("PASS" if ok == want else "FAIL", repr(code[:40]), "->", ok, out[:60])
