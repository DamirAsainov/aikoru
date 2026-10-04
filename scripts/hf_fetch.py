"""Скачивает файлы репозитория HuggingFace в локальную папку (докачивает пропущенные).

    python scripts/hf_fetch.py <repo> <revision> <dest> [--exclude .gguf,.bin]
"""
import json
import sys
import urllib.request
from pathlib import Path


def fetch_repo(repo: str, rev: str, dest: str | Path, exclude: list[str] = ()) -> None:
    dest = Path(dest)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1") as r:
        files = [f for f in json.load(r) if f["type"] == "file"]
    for f in files:
        path = f["path"]
        if any(path.endswith(e) for e in exclude):
            continue
        out = dest / path
        if out.exists() and out.stat().st_size == f["size"]:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"{repo}: {path} ({f['size'] / 1e6:.0f} MB)", flush=True)
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/{rev}/{path}", tmp)
        tmp.replace(out)
    print(f"{repo}: готово", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    exclude = args[args.index("--exclude") + 1].split(",") if "--exclude" in args else []
    fetch_repo(args[0], args[1], args[2], exclude)
