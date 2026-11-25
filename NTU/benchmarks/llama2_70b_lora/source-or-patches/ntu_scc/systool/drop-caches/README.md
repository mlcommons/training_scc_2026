# drop-caches systool

Drops the kernel page cache, dentries and inodes (`vm.drop_caches=3`) as root, for unprivileged users. Same construction as the
existing `/mnt/opt/systools/set-cpu-freq` and `set-gpu-freq`: a tiny setuid-root C launcher that verifies the root-owned script,
scrubs the environment and execs it; all logic is in the bash script.

Purpose: MLPerf Training rules require the caches to be cleared before every run and logged as `cache_clear`. `run.sub` does both
when `CLEAR_CACHES=1`, using `DROPCACHE_CMD` (default `sudo /sbin/sysctl vm.drop_caches=3`). Without passwordless sudo that step is
skipped and the logs are rejected by the package checker. With this tool: `CLEAR_CACHES=1 DROPCACHE_CMD=/mnt/opt/systools/drop-caches`.

## Files

| file | role |
|---|---|
| `drop-caches.c` | setuid launcher: checks the script is a root-owned, non-writable regular file; `setgroups/setgid/setuid(0)`; clean env (`PATH`, `IFS`, caller uid); `execve /usr/bin/bash /mnt/opt/systools/drop-caches.sh [arg]`; forwards at most one `-`-prefixed argument |
| `drop-caches.sh` | runs as root: `sync`, `echo 3 > /proc/sys/vm/drop_caches`, prints Cached/Buffers/Dirty before → after, one syslog line (tag `drop-caches`, with the caller's uid). Non-interactive. `--quiet`, `--help` |
| `Makefile` | `make` (build), `make test` (unprivileged checks), `make install` (as root) |

## Build and test (as the user)

```
cd ~/systools/drop-caches
make            # -> ./drop-caches
make test       # launcher refuses without setuid, script refuses without root, bad args refused
```

## Install (as root, on every compute node or once if /mnt/opt is shared)

```
cd /mnt/home/unicorn/systools/drop-caches
make install
#  = install -o root -g root -m 0755 drop-caches.sh /mnt/opt/systools/drop-caches.sh
#    install -o root -g root -m 4755 drop-caches    /mnt/opt/systools/drop-caches
```

Resulting listing should match the siblings: `-rwsr-xr-x root root drop-caches`, `-rwxr-xr-x root root drop-caches.sh`.
`/mnt/opt` must be mounted without `nosuid` (it already is, or set-gpu-freq would not work).

Smoke test as the user afterwards:

```
/mnt/opt/systools/drop-caches
# ntu-scc01: dropped caches: Cached 123456789 kB -> 1234567 kB, Buffers ... (caller uid 10001)
journalctl -t drop-caches -n 1     # or: grep drop-caches /var/log/syslog
```

## Use in the benchmark

`ntu_scc/launcher/run_nemo_l2l.sh` enables cache clearing automatically when the tool is present
(`CLEAR_CACHES=1`, `DROPCACHE_CMD=/mnt/opt/systools/drop-caches`), so the next run writes the `cache_clear` record; verify with
`grep '"key": "cache_clear"' slurm-<job>.out` (one record per node) and `grep cache_cleared slurm-<job>.out` (before → after sizes).

## Security notes

- The launcher accepts at most one argument, which must start with `-` and be ≤ 16 chars, so it can never be made to pass a path.
- Environment is replaced entirely (`PATH=/usr/bin:/bin`, `IFS`, `DROP_CACHES_CALLER=<uid>`); the script additionally unsets
  `BASH_ENV ENV CDPATH GLOBIGNORE` and resets aliases, like the sibling tools.
- The script is checked for root ownership and no group/other write bit before it is executed; a test build
  (`-DALLOW_NONROOT_SCRIPT`) relaxes only the ownership check and is never installed.
- The action itself is non-destructive (drops clean cache only; the kernel re-reads from disk as needed). Dirty pages are synced
  first, so no data is lost.
