#!/usr/bin/env python3
"""(User-Agent set: Cloudflare 1010-blocks the default Python UA.) Delete (terminate) the Runpod pod this runs on, using the pod's own RUNPOD_API_KEY / RUNPOD_POD_ID env.
Tries the REST API first, then the GraphQL podTerminate mutation. Never raises."""
import json, os, sys, urllib.request

pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
if not pod or not key:
    print("pod_delete: RUNPOD_POD_ID / RUNPOD_API_KEY not set; cannot self-delete", flush=True); sys.exit(0)
try:
    r = urllib.request.urlopen(urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE",
                                                      headers={"Authorization": f"Bearer {key}", "User-Agent": "curl/8.7.1"}), timeout=30)
    print(f"pod_delete: REST DELETE -> {r.status}", flush=True)
except Exception as e:
    print(f"pod_delete: REST failed ({e}); trying GraphQL", flush=True)
    try:
        body = json.dumps({"query": f'mutation {{ podTerminate(input: {{podId: "{pod}"}}) }}'}).encode()
        r = urllib.request.urlopen(urllib.request.Request("https://api.runpod.io/graphql", data=body, method="POST",
                                                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "curl/8.7.1"}),
                                   timeout=30)
        print(f"pod_delete: GraphQL -> {r.status} {r.read()[:200]!r}", flush=True)
    except Exception as e2:
        print(f"pod_delete: GraphQL failed too ({e2})", flush=True)
