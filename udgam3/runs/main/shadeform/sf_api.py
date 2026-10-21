"""Minimal Shadeform API client for Udgam runs. The API key is read from ~/.config/shadeform/api_key and never printed.

  python3 sf_api.py info ID            -> status, ip, ssh_user, ssh_port, cost_estimate, hourly price (one JSON line)
  python3 sf_api.py list               -> name, id, status, cloud, type, cost of every non-deleted instance
  python3 sf_api.py delete ID NAME     -> delete ONE instance, only if its name is exactly NAME (recorded at creation)
  python3 sf_api.py sshkeys            -> list registered SSH keys (id, name)
Library use: from sf_api import api; api("GET", "/instances/types", query={...}); api("POST", "/instances/create", body)
"""
import json, os, sys, urllib.error, urllib.parse, urllib.request

BASE = "https://api.shadeform.ai/v1"


def _key():
    return open(os.path.expanduser("~/.config/shadeform/api_key")).read().strip()


def api(method, path, body=None, query=None, timeout=90):
    url = BASE + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-API-KEY": _key(), "Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {e.read()[:400].decode(errors='replace')}") from None


def info(iid):
    d = api("GET", f"/instances/{iid}/info")
    keep = ("id", "name", "status", "cloud", "region", "shade_instance_type", "ip", "ssh_user", "ssh_port", "cost_estimate",
            "hourly_price", "created_at", "active_at", "auto_delete")
    return {k: d.get(k) for k in keep}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "info":
        print(json.dumps(info(sys.argv[2])))
    elif cmd == "list":
        for i in api("GET", "/instances").get("instances", []):
            print(i.get("name"), i.get("id"), i.get("status"), i.get("cloud"), i.get("shade_instance_type"), "cost", i.get("cost_estimate"))
    elif cmd == "delete":
        iid, name = sys.argv[2], sys.argv[3]
        cur = info(iid)
        if cur.get("name") != name:
            sys.exit(f"refusing: instance {iid} is named {cur.get('name')!r}, not {name!r}")
        api("POST", f"/instances/{iid}/delete")
        print(f"delete requested for {name} ({iid}); status was {cur.get('status')}, cost so far {cur.get('cost_estimate')}")
    elif cmd == "sshkeys":
        for k in api("GET", "/sshkeys").get("ssh_keys", []):
            print(k.get("id"), k.get("name"), "default" if k.get("is_default") else "")
    else:
        sys.exit(__doc__)
