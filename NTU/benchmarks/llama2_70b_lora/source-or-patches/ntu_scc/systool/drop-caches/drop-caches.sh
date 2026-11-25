#!/bin/bash
#
# drop-caches.sh - drop the kernel page cache, dentries and inodes (vm.drop_caches=3).
#
# Like set-cpu-freq.sh / set-gpu-freq.sh this is run as root through the
# setuid launcher /mnt/opt/systools/drop-caches. It is non-interactive on
# purpose: MLPerf's run.sub calls it unattended before every experiment
# (DROPCACHE_CMD) and then logs the mandatory cache_clear record.
#
# It syncs first (so dirty pages become clean and can be dropped), writes 3 to
# /proc/sys/vm/drop_caches, and prints the Cached/Buffers/Dirty sizes before
# and after so the caller can see that the drop happened. One line goes to
# syslog (tag drop-caches) with the calling uid for auditability.
#
# Dropping caches is non-destructive: it only discards clean cached data; the
# next readers re-fetch it from disk. It does not touch running processes.
#
#   usage: drop-caches [--quiet] [--help]

set -u

# Hardening: this runs as root, so do not inherit anything that decides which
# code gets executed.
PATH=/usr/bin:/bin
export PATH
IFS=$' \t\n'
unset BASH_ENV ENV CDPATH GLOBIGNORE
\unalias -a 2>/dev/null

readonly CTL=/proc/sys/vm/drop_caches
readonly MEMINFO=/proc/meminfo

die() { echo "error: $*" >&2; exit 1; }

quiet=0
case "${1:-}" in
    "") ;;
    -q|--quiet) quiet=1 ;;
    -h|--help)
        cat <<'EOF'
drop-caches - drop the kernel page cache, dentries and inodes (vm.drop_caches=3)

  usage: drop-caches [--quiet]

  Runs sync, writes 3 to /proc/sys/vm/drop_caches and prints Cached, Buffers
  and Dirty from /proc/meminfo before and after. Exits 0 only if the write
  succeeded. Non-interactive; no setting is changed or held.
EOF
        exit 0 ;;
    *) die "unknown argument '$1' (usage: drop-caches [--quiet|--help])" ;;
esac
[ "$#" -le 1 ] || die "too many arguments"

[ "$(id -u)" -eq 0 ] || die "must run as root (use the setuid launcher /mnt/opt/systools/drop-caches)"
[ -w "$CTL" ] || die "$CTL is not writable"

# Cached / Buffers / Dirty in kB from /proc/meminfo, as "cached buffers dirty".
meminfo() {
    awk '/^Cached:/ {c=$2} /^Buffers:/ {b=$2} /^Dirty:/ {d=$2} END {printf "%d %d %d\n", c, b, d}' "$MEMINFO"
}

read -r c0 b0 d0 <<< "$(meminfo)"
sync || die "sync failed"
if ! echo 3 > "$CTL" 2>/dev/null; then
    die "write to $CTL failed"
fi
read -r c1 b1 d1 <<< "$(meminfo)"

host=$(hostname -s 2>/dev/null || echo unknown)
caller=${DROP_CACHES_CALLER:-unknown}
msg="$host: dropped caches: Cached ${c0} kB -> ${c1} kB, Buffers ${b0} kB -> ${b1} kB, Dirty ${d0} kB -> ${d1} kB (caller uid $caller)"
logger -t drop-caches -- "$msg" 2>/dev/null || true
[ "$quiet" -eq 1 ] || echo "$msg"
exit 0
