#!/usr/bin/env python3
"""HF Hub helper for the training pod (uses HF_TOKEN from the env). Never prints the token.
  pod_hf.py latest REPO RUN                 -> prints the step in REPO/RUN/LATEST ("" if none)
  pod_hf.py fetch  REPO RUN STEP DIR        -> downloads RUN/*_<STEP>* into DIR (flat)
  pod_hf.py push   REPO RUN STEP CKDIR [KEEP]-> uploads CKDIR/*_<STEP>* to RUN/, writes RUN/LATEST, prunes all but the
                                               newest KEEP (default 2) steps from the repo head
  pod_hf.py put    REPO LOCAL PATH_IN_REPO  -> uploads a file or folder
"""
import os, re, sys, glob, shutil, time
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

api = HfApi()


def ensure(repo):
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)


def retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"pod_hf: retry {i + 1} after {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(15 * (i + 1))


def latest(repo, run):
    try:
        p = hf_hub_download(repo, f"{run}/LATEST", repo_type="model")
        return open(p).read().strip()
    except Exception:
        return ""


def steps_in_repo(repo, run):
    files = api.list_repo_files(repo, repo_type="model")
    return files, sorted({int(m.group(1)) for f in files if f.startswith(run + "/") and "/" not in f[len(run) + 1:]
                          and (m := re.search(r"_(\d{6})(?:_rank\d+)?\.(?:pt|json|ok)$", f))})


def main():
    cmd = sys.argv[1]
    if cmd == "latest":
        print(latest(sys.argv[2], sys.argv[3]))
    elif cmd == "fetch":
        repo, run, step, d = sys.argv[2:6]
        s = f"{int(step):06d}"
        tmp = snapshot_download(repo, repo_type="model", allow_patterns=[f"{run}/*_{s}*"], local_dir=d + "/.pull")
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(f"{tmp}/{run}/*_{s}*"):
            shutil.move(f, os.path.join(d, os.path.basename(f)))
        shutil.rmtree(d + "/.pull", ignore_errors=True)
        print("\n".join(sorted(os.listdir(d))))
    elif cmd == "push":
        repo, run, step, ckdir = sys.argv[2:6]
        keep = int(sys.argv[6]) if len(sys.argv) > 6 else 2
        s = f"{int(step):06d}"
        ensure(repo)
        retry(lambda: api.upload_folder(repo_id=repo, repo_type="model", folder_path=ckdir, path_in_repo=run,
                                        allow_patterns=[f"*_{s}*"], commit_message=f"{run} step {int(step)}"))
        retry(lambda: api.upload_file(path_or_fileobj=str(int(step)).encode(), path_in_repo=f"{run}/LATEST",
                                      repo_id=repo, repo_type="model", commit_message=f"{run} LATEST={int(step)}"))
        files, steps = steps_in_repo(repo, run)
        old = steps[:-keep] if keep > 0 else []
        dels = [f for f in files if f.startswith(run + "/") and any(f"_{o:06d}" in f for o in old)]
        if dels:
            retry(lambda: api.delete_files(repo_id=repo, repo_type="model", delete_patterns=dels,
                                           commit_message=f"{run}: prune steps {old}"))
        print(f"pushed step {int(step)} to {repo}/{run}; repo steps now {[x for x in steps if x not in old]}")
    elif cmd == "put":
        repo, local, path = sys.argv[2:5]
        ensure(repo)
        if os.path.isdir(local):
            retry(lambda: api.upload_folder(repo_id=repo, repo_type="model", folder_path=local, path_in_repo=path))
        else:
            retry(lambda: api.upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=repo, repo_type="model"))
        print(f"put {local} -> {repo}/{path}")


if __name__ == "__main__":
    main()
