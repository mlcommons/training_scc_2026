#!/usr/bin/env python3
"""Évince du cache de pages les fichiers du benchmark (données + checkpoint), sans droits root.

Remplace `sudo sysctl vm.drop_caches=3` de nemo/run.sub (pas de sudo sur Helma) : posix_fadvise
DONTNEED sur chaque fichier retire ses pages du cache du nœud, ce qui garantit que chaque essai relit
réellement les données, comme l'exige la règle MLPerf de vidage des caches.
Usage : drop_page_cache.py <dossier> [<dossier> ...]
"""
import os
import socket
import sys


def cached_kib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return -1


def main(roots):
    before = cached_kib()
    n_files = n_bytes = 0
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n_files += 1
                    n_bytes += os.fstat(fd).st_size
                finally:
                    os.close(fd)
    after = cached_kib()
    print(f"[{socket.gethostname().split('.')[0]}] cache_evicted files={n_files} "
          f"size_GB={n_bytes / 1e9:.1f} node_cached_kB {before} -> {after}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["/data", "/ckpt"])
