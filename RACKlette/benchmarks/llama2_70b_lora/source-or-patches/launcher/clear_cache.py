"""Evict files from the page cache (posix_fadvise DONTNEED, no root needed).
usage: clear_cache.py DIR [DIR...]    |    clear_cache.py --log-event  (emit the MLPerf cache_clear event)"""
import os, sys
if sys.argv[1:] == ["--log-event"]:
    sys.path.insert(0, os.environ["FTLLM"])
    from callback_logging import mllogger
    mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)
    sys.exit(0)
for d in sys.argv[1:]:
    total = 0
    for root, _, files in os.walk(d, followlinks=True):
        for f in files:
            p = os.path.join(root, f)
            try:
                fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
                total += os.path.getsize(p)
            except OSError:
                pass
    print(f"{os.uname().nodename}: evicted {total/2**30:.1f} GiB from page cache: {d}", flush=True)
