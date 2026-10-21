"""Mac-side self-delete for ONE Shadeform instance, mirroring what Runpod pods do themselves.

pod_train.sh (and our bootstrap) write "self-delete: <reason>" to the run log at every point where a Runpod pod deletes
itself: finished (after the final push is verified and DONE is written), deadline, env/CUDA/data/preflight failure,
start or bootstrap failure. It never writes it on "FINAL PUSH FAILED - HOLDING POD", so a pod holding the only copy of
the weights is NOT deleted (Shadeform's auto_delete at the deadline still applies). Deletes only the recorded instance
id, only if its name matches, once; the Shadeform key never leaves the Mac.

  python3 selfdel_mirror.py STATE_NAME RUN_NAME [--every 60]
Exit 0 = deleted (or already gone), 1 = gave up after the auto-delete time (Shadeform's own cap applies).
"""
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from sf_api import api, info

ap = argparse.ArgumentParser()
ap.add_argument("state"); ap.add_argument("run"); ap.add_argument("--every", type=int, default=60)
a = ap.parse_args()
rec = json.load(open(os.path.join(here, "state", f"{a.state}_sf.json")))
until = rec["deadline_epoch"] + 900
say = lambda m: print(f"{datetime.now():%H:%M} {m}", flush=True)
say(f"mirror for {rec['name']} ({rec['id']}), until {datetime.fromtimestamp(until):%H:%M}")


def self_delete_line(rc, out):
    """The pod's own self-delete request, or None. Only a successful remote grep counts, and only a line from the run
    log ("[<UTC time>] self-delete: <reason>"); helper messages such as "no ip yet (status creating)" never do
    (2026-10-18 bug: the first version treated any output as a request and would have deleted a pending instance)."""
    if rc != 0:
        return None
    for l in out.splitlines():
        if re.match(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\] self-delete: ", l.strip()):
            return l.strip()
    return None


def status():
    try:
        return info(rec["id"]).get("status")
    except RuntimeError as e:   # the info endpoint only serves non-deleted instances
        return "deleted" if "HTTP 404" in str(e) else f"?({str(e)[:60]})"
    except Exception as e:
        return f"?({str(e)[:60]})"


while time.time() < until:
    st = status()
    if st in ("deleting", "deleted"):
        say(f"instance already {st}"); sys.exit(0)
    try:
        r = subprocess.run(["bash", os.path.join(here, "sf_ssh.sh"), a.state,
                            f"grep -m1 '] self-delete:' /udgam_ws/logs/{a.run}.log 2>/dev/null"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=90)
    except subprocess.TimeoutExpired:  # a slow SSH answer is a retry: it crashed this backstop on 2026-10-18
        say("slow SSH answer; retrying"); time.sleep(30); continue
    line = self_delete_line(r.returncode, r.stdout)
    if line:
        say(f"pod asked to self-delete: {line[:160]}")
        time.sleep(45)   # pod_train uploads its log before this line; give any last upload a moment
        cur = info(rec["id"])
        if cur.get("name") != rec["name"]:
            say(f"name mismatch ({cur.get('name')} != {rec['name']}); not deleting"); sys.exit(1)
        api("POST", f"/instances/{rec['id']}/delete")
        say(f"delete requested; cost so far {cur.get('cost_estimate')}")
        for _ in range(10):
            time.sleep(30)
            s2 = status()
            if s2 in ("deleting", "deleted"):
                say(f"confirmed {s2}"); sys.exit(0)
        say("delete not confirmed after 5 min; auto_delete still applies"); sys.exit(1)
    time.sleep(a.every)
say("past the auto-delete time; Shadeform's auto_delete should have removed it"); sys.exit(1)
