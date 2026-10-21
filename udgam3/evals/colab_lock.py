"""One Colab CLI call at a time, across every session and script (2026-10-19: a long `colab exec` on one session hung
twice when another CLI process touched a second session). Shared with udgam3/runs/ctx32k/colab_ssh.sh: same lock dir.
Keep every call short: start long jobs in the background on the VM and poll them."""
import os, subprocess, time

LOCK = "/tmp/colab_cli.lock"
COLAB = [os.path.expanduser("~/.local/bin/colab"), "--auth=adc"]


def colab(*args, stdin=None, timeout=600):
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK); break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > 900:   # stale (a crashed holder)
                    os.rmdir(LOCK); continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > 1800:
                raise TimeoutError("colab lock busy for 30 min")
            time.sleep(1)
    try:
        return subprocess.run(COLAB + list(args), input=stdin, capture_output=True, text=True, timeout=timeout)
    finally:
        try:
            os.rmdir(LOCK)
        except FileNotFoundError:
            pass
