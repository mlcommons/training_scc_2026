_rt=$(env -u LD_LIBRARY_PATH "$V/bin/python" -c '
import ctypes
ctypes.CDLL("libstdc++.so.6")
for line in open("/proc/self/maps"):
    if "libstdc++.so.6" in line:
        print(line.split()[-1]); break
' 2>/dev/null)
_rtmax=""
[ -f "$_rt" ] && _rtmax=$(objdump -T "$_rt" 2>/dev/null \
    | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -V | tail -1)
_tc=""
if [ -n "$_rtmax" ]; then
    for _g in $(ls -d /home/apps/gcc/*/bin/gcc 2>/dev/null | sort -V); do
        [ -x "$_g" ] || continue
        _lib="$(dirname "$(dirname "$_g")")/lib64/libstdc++.so.6"
        _max=$(objdump -T "$_lib" 2>/dev/null | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -V | tail -1)
        [ -n "$_max" ] || continue
        [ "$(printf '%s\n%s\n' "$_max" "$_rtmax" | sort -V | tail -1)" = "$_rtmax" ] && _tc="$(dirname "$_g")"
    done
fi
if [ -n "$_tc" ]; then
    export PATH="$_tc:$PATH" CC="$_tc/gcc" CXX="$_tc/g++" CUDAHOSTCXX="$_tc/g++"
else
    export CC=gcc CXX=g++ CUDAHOSTCXX=g++
fi
echo "toolchain: CC=$CC (venv libstdc++ max: ${_rtmax:-unknown})"
