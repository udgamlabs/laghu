"""Create ONE Shadeform instance for a Udgam training run (called by create_sf.sh, which sources the run config).

Same run config as Runpod (common.sh + a .conf), plus Shadeform fields:
  SF_CLOUD SF_REGION SF_TYPE SF_OS   which offer (from shadeform_offers.py)
  SF_SPEND_CAP                       dollars: Shadeform auto-deletes the instance at this spend (hard cap, server-side)
  MAX_MIN                            minutes: DEADLINE_EPOCH for pod_train.sh's watchdog; Shadeform auto-deletes at +10 min
Secrets read from files, never printed: the Mac's active HF token (~/.cache/huggingface/token, the fine-grained
"claudewrite" token; user's choice 2026-10-18) or HF_TOKEN_FILE. SSH: our public key ~/.ssh/udgam_shadeform_ed25519.pub
is installed by the startup script (nothing is registered in the Shadeform account). W&B runs offline (no W&B key on the VM).

  python3 create_sf_instance.py [--dry-run]   (--dry-run prints the request with secrets redacted; creates nothing)
Refuses if a non-deleted instance with the same name exists. Writes $OUT/<RUN_NAME>_sf.json on success.
"""
import base64, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_api import api

E = os.environ
dry = "--dry-run" in sys.argv
need = ["RUN_NAME", "TRAIN_ARGS", "DATA_REPO", "CKPT_REPO", "EXPORT_REPO", "SAVE_EVERY", "CODE_REV", "IMAGE", "MAX_MIN",
        "SF_CLOUD", "SF_REGION", "SF_TYPE", "SF_SPEND_CAP"]
miss = [k for k in need if not E.get(k)]
if miss:
    sys.exit(f"missing config: {miss}")
here = os.path.dirname(os.path.abspath(__file__))
name = "udgam-" + E["RUN_NAME"]


def secret(path, what):
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return open(p).read().strip()
    if dry:
        return f"<{what} missing: {path}>"
    sys.exit(f"{what} not found at {path}")


hf_token = secret(E.get("HF_TOKEN_FILE", "~/.cache/huggingface/token"), "HF token")
ssh_pub = secret("~/.ssh/udgam_shadeform_ed25519.pub", "SSH public key")
if not ssh_pub.startswith("ssh-ed25519 ") or "\n" in ssh_pub:
    sys.exit("unexpected SSH public key format")
now = int(time.time())
deadline = now + int(E["MAX_MIN"]) * 60
env = {"RUN_NAME": E["RUN_NAME"], "CODE_REPO": E.get("CODE_REPO", "YOUR_HF_NAMESPACE/laghu-build-code"), "CODE_REV": E["CODE_REV"],
       "DATA_REPO": E["DATA_REPO"], "CKPT_REPO": E["CKPT_REPO"], "EXPORT_REPO": E["EXPORT_REPO"], "TRAIN_ARGS": E["TRAIN_ARGS"],
       "SAVE_EVERY": E["SAVE_EVERY"], "DEADLINE_EPOCH": str(deadline), "HF_TOKEN": hf_token, "WANDB_MODE": "offline",
       "WANDB_PROJECT": E.get("WANDB_PROJECT", "udgam3-main"), "WANDB_ENTITY": "udgamlabs",
       "WANDB_RUN_ID": E.get("WANDB_RUN_ID", E["RUN_NAME"]), "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
for k in ("DATA_PATTERNS", "SKIP_EXPORT", "FALLBACK_REPO", "EXTRA_DATA_JSON", "RESUME_FROM_REPO", "NPROC"):
    if E.get(k):
        env[k] = E[k]
for k, v in env.items():
    if "\n" in v:
        sys.exit(f"env {k} contains a newline (not allowed in a docker env file)")
start = ("set -u\n"
         "pip install -q --break-system-packages --root-user-action=ignore 'huggingface_hub[hf_xet]==1.33.0' \\\n"
         "  && hf download \"$CODE_REPO\" --revision \"${CODE_REV:-main}\" --include 'train-udgam3/*' --local-dir /workspace/code_dl --quiet \\\n"
         "  && mkdir -p /workspace/code && rm -rf /workspace/code/train && mv /workspace/code_dl/train-udgam3 /workspace/code/train \\\n"
         "  && exec bash /workspace/code/train/ops/pod_train.sh\n"
         "echo \"[$(date -u +%FT%TZ)] START_FAILED (code download or pod_train.sh start)\" >> /workspace/logs/$RUN_NAME.log\n"
         "echo \"[$(date -u +%FT%TZ)] self-delete: start failed\" >> /workspace/logs/$RUN_NAME.log\n"
         "sleep infinity\n")
tmpl = open(os.path.join(here, "bootstrap.sh.tmpl")).read()
script = (tmpl.replace("__RUN_NAME__", E["RUN_NAME"]).replace("__IMAGE__", E["IMAGE"]).replace("__SSH_PUBKEY__", ssh_pub)
          .replace("__ENVFILE__", "\n".join(f"{k}={v}" for k, v in env.items())).replace("__START__", start.rstrip("\n")))
for marker in ("__RUN_NAME__", "__IMAGE__", "__ENVFILE__", "__START__", "__SSH_PUBKEY__"):
    assert marker not in script, marker
rfc = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
body = {"cloud": E["SF_CLOUD"], "region": E["SF_REGION"], "shade_instance_type": E["SF_TYPE"], "shade_cloud": True, "name": name,
        "launch_configuration": {"type": "script", "script_configuration": {"base64_script": base64.b64encode(script.encode()).decode()}},
        "volume_mount": {"auto": True}, "tags": ["udgam", E["RUN_NAME"]],
        "auto_delete": {"date_threshold": rfc(deadline + 600), "spend_threshold": f"{float(E['SF_SPEND_CAP']):.2f}"},
        "alert": {"spend_threshold": f"{0.8 * float(E['SF_SPEND_CAP']):.2f}"}}
if E.get("SF_OS"):
    body["os"] = E["SF_OS"]

if dry:
    shown = json.loads(json.dumps(body))
    shown["launch_configuration"]["script_configuration"]["base64_script"] = f"<{len(script)} chars, decoded below>"
    print(json.dumps(shown, indent=1))
    print("----- decoded startup script (HF token redacted) -----")
    print(script.replace(hf_token, "<HF_TOKEN redacted>"))
    sys.exit(0)

live = [i for i in api("GET", "/instances").get("instances", []) if i.get("name") == name and i.get("status") not in ("deleted",)]
if live:
    sys.exit(f"refusing: instance {name} already exists ({live[0].get('id')}, {live[0].get('status')})")
resp = api("POST", "/instances/create", body)
iid = resp.get("id")
if not iid:
    sys.exit(f"create failed: {json.dumps(resp)[:300]}")
out = os.environ.get("OUT", here)
rec = {"id": iid, "name": name, "cloud": E["SF_CLOUD"], "region": E["SF_REGION"], "type": E["SF_TYPE"], "created_epoch": now,
       "deadline_epoch": deadline, "auto_delete_at": rfc(deadline + 600), "spend_cap": E["SF_SPEND_CAP"]}
json.dump(rec, open(os.path.join(out, f"{E['RUN_NAME']}_sf.json"), "w"), indent=1)
print(f"CREATED {iid} {name} {E['SF_CLOUD']}/{E['SF_REGION']} {E['SF_TYPE']} deadline_epoch {deadline} spend_cap {E['SF_SPEND_CAP']}")
