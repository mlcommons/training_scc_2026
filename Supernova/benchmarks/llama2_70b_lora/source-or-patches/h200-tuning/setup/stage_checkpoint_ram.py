#!/usr/bin/env python3
"""Stage and checksum a checkpoint in node-local tmpfs; reuse validated copies.

Use /tmp/mlperf-sc26-ram on coffeepot3/4: /tmp is tmpfs there and persists
across SSH sessions, unlike /dev/shm.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination
    files = sorted(p for p in source.rglob("*") if p.is_file())
    signature = {str(p.relative_to(source)): [p.stat().st_size, p.stat().st_mtime_ns]
                 for p in files}
    size = sum(v[0] for v in signature.values())
    marker = destination / ".ram-checkpoint.json"
    if destination.exists():
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["source_signature"] == signature and all(
                    (destination / name).is_file()
                    and (destination / name).stat().st_size == info[0]
                    for name, info in signature.items()):
                print(f"Reusing verified checkpoint: {destination}", flush=True)
                return
        raise RuntimeError(f"Destination exists without a matching verified copy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < size + 1024**3:
        raise RuntimeError("Insufficient RAM-disk space for checkpoint")
    stage = destination.with_name(destination.name + f".partial-{os.getpid()}")
    stage.mkdir()
    start = time.monotonic()
    print(f"Copying {size / 1024**3:.2f} GiB to {stage}", flush=True)
    try:
        # tmpfs here does not support the source filesystem's ACLs. Copy data
        # and timestamps, materializing symlinks, without ACL/ownership metadata.
        subprocess.run(["cp", "-RL", "--preserve=timestamps",
                        "--no-preserve=mode,ownership,xattr,context", "--reflink=never",
                        str(source) + "/.", str(stage)], check=True)
        print("Copy complete; verifying SHA256 of every file", flush=True)

        def verify(name):
            with (source / name).open("rb") as f:
                original = hashlib.file_digest(f, "sha256").hexdigest()
            with (stage / name).open("rb") as f:
                copied = hashlib.file_digest(f, "sha256").hexdigest()
            if original != copied:
                raise RuntimeError(f"Checksum mismatch: {name}")
            return name, copied

        with ThreadPoolExecutor(max_workers=2) as pool:
            hashes = dict(pool.map(verify, signature))
        (stage / marker.name).write_text(json.dumps({
            "source": str(source), "source_signature": signature,
            "sha256": hashes, "bytes": size,
        }, indent=2) + "\n")
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f"Verified {len(files)} files in {time.monotonic() - start:.1f}s: {destination}", flush=True)


if __name__ == "__main__":
    main()
