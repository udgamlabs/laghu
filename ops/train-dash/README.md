# Udgam Labs private training dashboard

A Cloudflare Worker (`udgam-train-dash`, workers.dev) with a D1 table of run snapshots, plus a Mac-side pusher.
Private by an unguessable link (no Cloudflare Access on the account yet; enabling it in the dashboard would add an
email login on top). Live: the pusher sends a snapshot every 60 s; the page refreshes every 30 s.

- Link: `$(cat ~/.config/udgam/dash_url)/$(cat ~/.config/udgam/dash_view_token)/` (secret, keep it out of repos/chats).
- Secrets (mode 600, never committed): `~/.config/udgam/dash_view_token`, `dash_push_secret`, `dash_url`; stored in the
  Worker with `wrangler secret put VIEW_TOKEN|PUSH_SECRET`.
- Deploy: `cd ops/train-dash && wrangler deploy` (wrangler config in this folder). Schema: `schema.sql`.
- Access tests (2026-10-18): `/`, a wrong link and unauthorised pushes all return 404; pages are no-store, noindex,
  no-referrer, no framing.

## A new run
1. Write `push/configs/<run>.json` (copy `udgam3-base-v1.json`: model, data, recipe, hardware, steps, phase steps,
   cap, deadline, reference lines, manual events).
2. Start the pusher next to the run's watcher:
   `python3 ops/train-dash/push/dash_push.py --config ops/train-dash/push/configs/<run>.json --ssh <ssh helper>
   --state <state name> --remote-log <log path on the machine> [--sf-id <Shadeform instance>]`
   It reads only new log bytes over SSH (read-only), parses steps/evals/checkpoints/events and pushes; it stops when the
   run is done or the machine is gone. `--dry --once` prints the snapshot instead; `--log-file` parses a local log.

## Public version (TEST ONLY: needs the user's go before it is public)
- Preview, behind the same secret link: `<dash_url>/<view token>/public/`.
- The pusher builds the public snapshot from an ALLOWLIST (`public_snapshot()` in `push/dash_push.py`): progress %,
  tokens, speed, time left, GPU telemetry, checkpoint count, sanitized milestones/log, held-out (fresh text) and
  training loss from 25% on by % of training. Never: money, provider, recipe, steps/batch/shape, data, targets, W&B,
  monitoring. `push/check_public.py` scans every public snapshot and the pusher refuses to send one that fails.
- Stored as `<run>.public`; the private run list hides these.
- Going public later = set the Worker variable `PUBLIC_ENABLED=1` (enables `/live/`, which can only read `*.public`
  records), optionally a custom domain, and add caching. Do not do this without the user's explicit go.
