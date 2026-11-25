# drop-caches: setuid launcher + root script, same pattern as set-cpu-freq / set-gpu-freq.
#   make            build ./drop-caches (production: script path /mnt/opt/systools/drop-caches.sh)
#   make test       build a test launcher against ./drop-caches.sh and run the unprivileged checks
#   make install    copy both files into /mnt/opt/systools with the right owner/modes (run as root)
CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -Wformat=2 -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE
LDFLAGS ?= -pie -Wl,-z,relro,-z,now
PREFIX  ?= /mnt/opt/systools
HERE    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

all: drop-caches

drop-caches: drop-caches.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# Test launcher: points at the script in this directory and accepts a non-root owner (never install this one).
drop-caches.test: drop-caches.c drop-caches.sh
	$(CC) $(CFLAGS) $(LDFLAGS) -DALLOW_NONROOT_SCRIPT -DSCRIPT_PATH='"$(HERE)/drop-caches.sh"' -o $@ $<

test: drop-caches drop-caches.test
	bash -n drop-caches.sh
	@echo "-- launcher without setuid must refuse:"; ./drop-caches.test; test $$? -eq 1
	@echo "-- script as non-root must refuse:"; bash drop-caches.sh; test $$? -eq 1
	@echo "-- --help works:"; bash drop-caches.sh --help | head -2
	@echo "-- bad argument refused:"; ./drop-caches /etc/passwd; test $$? -eq 1
	@echo "tests OK"

install: drop-caches drop-caches.sh
	install -o root -g root -m 0755 drop-caches.sh $(PREFIX)/drop-caches.sh
	install -o root -g root -m 4755 drop-caches    $(PREFIX)/drop-caches
	ls -la $(PREFIX)/drop-caches $(PREFIX)/drop-caches.sh

clean:
	rm -f drop-caches drop-caches.test

.PHONY: all test install clean
