#!/bin/bash
# Run in Git Bash with MSYS make, Windows x64 GCC, pkg-config and NASM in PATH.
# Arguments: source directory (contains x264 and ffmpeg), output prefix.
set -euo pipefail
sources=$(cd "$1" && pwd)
mkdir -p "$2"
prefix=$(cd "$2" && pwd -W)
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
cd "$sources/x264-373697b"
bash ./configure --prefix="$prefix" --enable-static --disable-cli --disable-opencl
make -j8 SHELL=bash
make SHELL=bash install
cd "$sources/FFmpeg-db69d06eee"
# Archive builds must not inherit the enclosing application's Git description.
printf '7.1.1\n' > VERSION
bash ./configure --prefix="$prefix" --cc=gcc --ar=ar --nm=nm --ranlib=ranlib \
    --target-os=mingw32 --arch=x86_64 --disable-autodetect --enable-gpl --enable-libx264 \
    --disable-shared --enable-static --disable-network --disable-doc --disable-ffplay --disable-debug \
    --extra-cflags="-I$prefix/include" --extra-ldflags="-L$prefix/lib"
make -j8 V=1
make install
cp COPYING.GPLv2 "$prefix/LICENSE"
