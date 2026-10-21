#!/usr/bin/env bash
# Stage nanochat (working tree; branch udgam3 commit recorded) + pod scripts + tokenizer_v1 and upload them to
# YOUR_HF_NAMESPACE/laghu-build-code under train-udgam3/ (Udgam-3's own copy; the shared train/ folder is not touched).
set -euo pipefail
R="${LAGHU_REPO:-.}"; S=$(mktemp -d)
mkdir -p "$S/train-udgam3/ops" "$S/train-udgam3/tokenizer"
rsync -a --exclude .venv --exclude .git --exclude __pycache__ --exclude 'dev/*.png' "$R/nanochat/" "$S/train-udgam3/nanochat/"
cp "$R/ops/pod_train.sh" "$R/ops/pod_hf.py" "$R/ops/pod_samples.py" "$R/ops/pod_build/pod_delete.py" "$S/train-udgam3/ops/"
cp "$R/artifacts/tokenizer_v1/"* "$S/train-udgam3/tokenizer/"
( cd "$R/nanochat" && git rev-parse HEAD > "$S/train-udgam3/NANOCHAT_COMMIT" && git status --short >> "$S/train-udgam3/NANOCHAT_COMMIT" )
cd "$R" && uv run --project evals/harness python - "$S" <<'PY'
import sys
from huggingface_hub import HfApi
c = HfApi().upload_folder(repo_id="YOUR_HF_NAMESPACE/laghu-build-code", folder_path=sys.argv[1], repo_type="model",
                          commit_message="udgam3 train code", delete_patterns=["train-udgam3/nanochat/**"])
print("pushed train-udgam3:", c.oid[:10] if hasattr(c, "oid") else c)
PY
rm -rf "$S"
