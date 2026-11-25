"""Print the name of the interface holding an IPv4 address in the given prefix.

    python ib_ifname.py 10.18.18.

The IPoIB interface is ibP2s2f0 on work1-8 but ibP4s4f0 on work9, and gloo needs
the exact name. The container has no `ip`, so this asks the kernel directly.
"""
import fcntl
import os
import socket
import struct
import sys

SIOCGIFADDR = 0x8915
prefix = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for name in sorted(os.listdir("/sys/class/net")):
    try:
        raw = fcntl.ioctl(s.fileno(), SIOCGIFADDR, struct.pack("256s", name[:15].encode()))
    except OSError:
        continue
    if socket.inet_ntoa(raw[20:24]).startswith(prefix):
        print(name)
        sys.exit(0)
sys.exit(f"no interface with an address in {prefix}")
