# Repository names used by the scripts

The scripts are published as they ran. Private Hugging Face repositories were renamed to placeholders under `YOUR_HF_NAMESPACE`, so a full re-run writes and reads only your own repositories. To start partway, copy the public artifact into the placeholder repo.

| In the scripts | Public artifact with the same content |
|---|---|
| `YOUR_HF_NAMESPACE/laghu-base-ckpt` | udgamlabs/Laghu-Checkpoints (pretrain/) |
| `YOUR_HF_NAMESPACE/laghu-ckpt-test` | none (dry-run scratch repo) |
| `YOUR_HF_NAMESPACE/laghu-ctx32k-data` | rebuild with udgam3/runs/ctx32k/build_ctx32k_data.py |
| `YOUR_HF_NAMESPACE/laghu-ctx32k` | udgamlabs/Laghu-Base-32K and udgamlabs/Laghu-Checkpoints (ctx32k/) |
| `YOUR_HF_NAMESPACE/laghu-pretrain-v3` | rebuild with udgam/build_mix.py --mix v3 (not re-hosted) |
| `YOUR_HF_NAMESPACE/laghu-val-sets` | udgamlabs/laghu-evals (val_sets/) |
| `YOUR_HF_NAMESPACE/udgam-val-sets` | udgamlabs/laghu-evals (val_sets/) |
| `YOUR_HF_NAMESPACE/laghu-sft-data` | udgamlabs/laghu-sft |
| `YOUR_HF_NAMESPACE/laghu-sft` | udgamlabs/Laghu (2 passes) and udgamlabs/Laghu-Checkpoints (sft/) |
| `YOUR_HF_NAMESPACE/laghu-rl` | udgamlabs/Laghu-Checkpoints (rl-r1-failed/) |
| `YOUR_HF_NAMESPACE/laghu-base` | udgamlabs/Laghu-Base |
| `YOUR_HF_NAMESPACE/udgam2-ctx32k-data` | rebuild with udgam2/ctx32k/build_long32k.py |
| `YOUR_HF_NAMESPACE/laghu-build-code` | this repository |
| `YOUR_HF_NAMESPACE/laghu-g1-v3` | none (data-mix gate pools; rebuild with udgam/build_mix.py) |
| `YOUR_HF_NAMESPACE/udgam2-base-ckpt` | udgamlabs/Gargi-Base (Gargi lineage; not needed for Laghu) |
| `YOUR_HF_NAMESPACE/udgam2-ctx32k` | none (Gargi's 32K run; not needed for Laghu) |
| `YOUR_HF_NAMESPACE/udgam-pretrain-v2b` | none (Gargi's pool; its replay rows are not used by Laghu) |
