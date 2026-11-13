#!/usr/bin/env bash
# Download the MDK SDK and its C++ ABI headers into ./mdk-sdk.
#
#   scripts/fetch-mdk-sdk.sh [windows|linux|macos] [nightly|release]
#
# A plugin must be built against the same MDK build it is loaded into: on an
# ABI mismatch libmdk only logs "MAY CRASH!" and loads the plugin anyway.
# Gyroflow ships the nightly (qml-video-rs builds with `mdk-nightly`), and the
# release ABI headers on SourceForge lag the release SDK, so nightly is the
# default.
set -euo pipefail

os="${1:-}"
channel="${2:-nightly}"
if [[ -z "$os" ]]; then
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) os=windows ;;
        Darwin) os=macos ;;
        *) os=linux ;;
    esac
fi

case "$os" in
    windows) sdk=mdk-sdk-windows-clang.7z ;;
    linux)   sdk=mdk-sdk-linux.tar.xz ;;
    macos)   sdk=mdk-sdk-macOS.tar.xz ;;
    *) echo "unknown OS: $os" >&2; exit 1 ;;
esac

case "$channel" in
    nightly) base="https://master.dl.sourceforge.net/project/mdk-sdk/nightly" ;;
    release) base="https://master.dl.sourceforge.net/project/mdk-sdk" ;;
    *) echo "unknown channel: $channel" >&2; exit 1 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$root/.mdk-download"
rm -rf "$work" "$root/mdk-sdk"
mkdir -p "$work"
cd "$work"

fetch() {
    curl -fsSL --retry 3 -o "$1" "$base/$1?viasf=1"
}

extract() {
    case "$1" in
        *.7z) 7z x -y "$1" > /dev/null ;;
        *.tar.xz) tar xf "$1" ;;
    esac
}

fetch "$sdk"
fetch mdk-abi-sdk.7z
extract "$sdk"
# The ABI archive holds mdk-sdk/include/abi/mdk/*.h, laid over the SDK.
extract mdk-abi-sdk.7z

mv mdk-sdk "$root/mdk-sdk"
cd "$root"
rm -rf "$work"

grep -E "define MDK_(ABI_VERSION|MAJOR|MINOR|MICRO)" mdk-sdk/include/abi/mdk/global.h
