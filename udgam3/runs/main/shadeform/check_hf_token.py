"""Check the fine-grained HF token for Shadeform runs ($0): it must READ the code bundle, data and val sets, and WRITE
to the checkpoint/export repos (8 MB upload + delete each). Token: HF_TOKEN_FILE or the Mac's active token
(~/.cache/huggingface/token), never printed.

  uv run --project evals/harness python udgam3/runs/main/shadeform/check_hf_token.py
"""
import os, sys, tempfile
from huggingface_hub import HfApi

tok = open(os.path.expanduser(os.environ.get("HF_TOKEN_FILE", "~/.cache/huggingface/token"))).read().strip()
api = HfApi(token=tok)
READ = [("YOUR_HF_NAMESPACE/laghu-build-code", "model"), ("YOUR_HF_NAMESPACE/laghu-pretrain-v3", "dataset"),
        ("YOUR_HF_NAMESPACE/udgam-val-sets", "dataset"), ("YOUR_HF_NAMESPACE/laghu-val-sets", "dataset")]
WRITE = ["YOUR_HF_NAMESPACE/laghu-ckpt-test", "YOUR_HF_NAMESPACE/laghu-base-ckpt", "YOUR_HF_NAMESPACE/laghu-base"]
ok = True
try:
    who = api.whoami()
    print("token user:", who.get("name"), "| type:", (who.get("auth") or {}).get("accessToken", {}).get("role"))
except Exception as e:
    sys.exit(f"whoami failed: {str(e)[:200]}")
for repo, kind in READ:
    try:
        n = len(api.list_repo_files(repo, repo_type=kind))
        print(f"read  OK   {repo} ({n} files)")
    except Exception as e:
        ok = False; print(f"read  FAIL {repo}: {str(e)[:160]}")
p = os.path.join(tempfile.mkdtemp(), "t.bin"); open(p, "wb").write(os.urandom(8 << 20))
for repo in WRITE:
    try:
        api.upload_file(path_or_fileobj=p, path_in_repo="token_check/t.bin", repo_id=repo, commit_message="token write check")
        api.delete_file("token_check/t.bin", repo_id=repo, commit_message="remove token write check")
        print(f"write OK   {repo}")
    except Exception as e:
        ok = False; print(f"write FAIL {repo}: {str(e)[:160]}")
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
