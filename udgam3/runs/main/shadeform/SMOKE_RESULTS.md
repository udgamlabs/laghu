# Shadeform smoke tests (2026-10-18): the Shadeform path works end to end

**Evidence:** [MEASURED] two instances via Shadeform (user go), logs in `state/smoke-denvr/` and `state/smoke-lambda/`
(copied from HF `YOUR_HF_NAMESPACE/laghu-ckpt-test`), costs from Shadeform's `cost_estimate`. Real 150M shape and batch,
60 steps (TST 0-14, cool-down data from 48), checkpoints every 20.

| | Denvr 1× A100 80GB SXM (houston) | Lambda 1× H100 PCIe (saltlakecity) |
|---|---|---|
| Created → active | 6 min | ~9 min |
| Host driver | 550.144 (CUDA 12.4) | 570.148 (CUDA 12.8; the OS label said cuda12.4) |
| Workspace | auto-mounted 2 TB disk (`/mnt/vdc`) | root disk, 967 GB free |
| Docker | `--gpus all` works | works (docker info showed no nvidia runtime, but GPUs were visible) |
| Env ready after container start | ~30 s | ~4 min |
| Speed, TST / normal | 128K / 130K tok/s (MFU ~50%), device batch 16 | 143K / 190K tok/s (MFU ~30%), device batch 32 |
| Peak memory | 30.0 GB | 57.8 GB (same as Runpod H100 SXM) |
| Held-out bpb at step 60 (val / fresh) | 1.908 / 2.015 | 1.895 / 2.004 |
| Checkpoints 20/40/60, final push | yes | yes |
| Export parity (final, EMA) | PASS (9.5e-7, 4.0e-7) | PASS (6.5e-7, 4.1e-7) |
| DONE + log upload + Mac-side delete | yes; deleted 1 min after DONE | yes; deleted 1 min after DONE |
| Cost | **$0.36** | **$1.07** |

**Findings**
- Our torch 2.9.1+cu128 env runs on a CUDA 12.4 (550) driver: the cheap 8× H100 offers with 12.4 images are usable.
- OS image labels don't give the driver version (Lambda's "cuda12.4" image had driver 570); the bootstrap logs it.
- H100 PCIe is ~65% of H100 SXM for this model (190K vs 291K normal phase): use SXM/NVLink 8× nodes only.
- A100 80GB: ~129K tok/s per GPU in both phases (compute-bound, so TST costs nothing extra), ~45% of an H100 SXM.
- Bug caught: the first self-delete mirror took the SSH helper's "no ip yet" message as a delete request (killed
  before it acted; no instance lost). Fixed to accept only a real `[UTC] self-delete:` log line from a successful
  read, with 9 unit cases.
