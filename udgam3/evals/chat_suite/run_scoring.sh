#!/bin/bash
# Score, check and tabulate the chat suite on the Mac (CPU only):  bash udgam3/evals/chat_suite/run_scoring.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/../../.."
TAG=${1:-full}
PY=(uv run -q --project evals/harness --with langdetect==1.0.9 --with immutabledict==4.2.1 --with math-verify==0.9.0 python)
"${PY[@]}" udgam3/evals/chat_suite/score.py --tag "$TAG" 2>&1 | grep -v -i -E "warn|nltk|punkt" || true
"${PY[@]}" udgam3/evals/chat_suite/validate.py --tag "$TAG" 2>&1 | grep -v -i -E "warn|nltk|punkt" || true
python3 udgam3/evals/chat_suite/eyeball.py --tag "$TAG"
python3 udgam3/evals/chat_suite/make_tables.py --tag "$TAG" > "udgam3/evals/chat_suite/TABLES_$TAG.md"
echo "tables: udgam3/evals/chat_suite/TABLES_$TAG.md"
