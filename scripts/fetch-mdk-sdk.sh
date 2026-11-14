#!/usr/bin/env bash
# Download the MDK SDK and its C++ ABI headers into ./mdk-sdk.
#
#   scripts/fetch-mdk-sdk.sh [windows|linux|macos] [nightly|release] [output dir]
#
# A plugin must be built against the same MDK build it is loaded into: on an
# ABI mismatch libmdk only logs "MAY CRASH!" and loads the plugin anyway.
# Gyroflow ships the nightly (qml-video-rs builds with `mdk-nightly`), and the
# release ABI headers on SourceForge lag the release SDK, so nightly is the
# default.
#
# The SDK and the ABI headers are separate archives: each nightly build uploads
# the ABI archive first and the SDKs 8-30 minutes later, several builds a day,
# and neither carries a build id. The pair is taken as one build's only when the
# ABI archive does not change during the download and the SDK is no older than
# the headers and at most an hour newer, both by the server's Last-Modified and
# by the archives' file times (libmdk against the headers' global.h). While a
# build's SDK is still on its way the script waits up to 45 minutes; otherwise a
# mismatch fails.
set -euo pipefail

pair_gap=3600      # seconds
clock_slack=120    # the archives come from different CI runners
publish_wait=2700  # how long after an ABI upload its SDK may still be on its way

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
    windows) sdk=mdk-sdk-windows-clang.7z; libmdk=bin/x64/mdk.dll ;;
    linux)   sdk=mdk-sdk-linux.tar.xz;     libmdk=lib/amd64/libmdk.so.0 ;;
    macos)   sdk=mdk-sdk-macOS.tar.xz;     libmdk=lib/mdk.framework/Versions/A/mdk ;;
    *) echo "unknown OS: $os" >&2; exit 1 ;;
esac

case "$channel" in
    nightly) base="https://master.dl.sourceforge.net/project/mdk-sdk/nightly" ;;
    release) base="https://master.dl.sourceforge.net/project/mdk-sdk" ;;
    *) echo "unknown channel: $channel" >&2; exit 1 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${3:-$root/mdk-sdk}"
work="$root/.mdk-download"
rm -rf "$work"  # the old SDK stays until the new one is checked
mkdir -p "$work"
trap 'cd "$root"; rm -rf "$work"' EXIT
cd "$work"

# Downloads $1 to $2, with the response headers in $2.headers.
# curl is bounded, so that a stalled connection can't hold a build for ever: a transfer that stays under 1 kB/s for a
# minute is given up, and so is one (with its retries) that takes longer than the total.
fetch() {
    curl -fsSL --connect-timeout 30 --speed-limit 1024 --speed-time 60 --max-time 1800 \
        --retry 3 --retry-delay 5 --retry-max-time 2400 -D "$2.headers" -o "$2" "$base/$1?viasf=1"
}

# Epoch seconds of an HTTP date or of the Last-Modified of the last response in a header dump (SourceForge
# redirects to a mirror; the file is the final response's). Empty when there is none.
epoch_of() {
    date -u -d "$1" +%s 2>/dev/null || date -u -j -f "%a, %d %b %Y %H:%M:%S %Z" "$1" +%s 2>/dev/null || true
}
last_modified() {
    local lm
    lm=$(tr -d '\r' | awk 'toupper(substr($0,1,5))=="HTTP/" {lm=""} tolower($1)=="last-modified:" {sub(/^[^:]*:[ \t]*/, ""); lm=$0} END {print lm}')
    [[ -n "$lm" ]] && epoch_of "$lm"
    true
}
# Empty when the request fails: the caller asks again or gives up at its deadline.
remote_modified() {
    { curl -fsSIL --connect-timeout 30 --max-time 60 --retry 3 --retry-delay 5 --retry-max-time 180 "$base/$1?viasf=1" 2>/dev/null || true; } | last_modified
}
file_time() {
    [[ -e "$1" ]] && { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }
    true
}
utc() {
    date -u -d "@$1" "+%Y-%m-%d %H:%M:%S UT" 2>/dev/null || date -u -r "$1" "+%Y-%m-%d %H:%M:%S UT"
}

# Why ABI headers of time $2 and an SDK of time $3 can't be one build's (empty if they can), $1 naming the times.
pair_problem() {
    [[ -z "$2" || -z "$3" ]] && return 0
    local gap=$(( $3 - $2 ))
    if (( gap < -clock_slack )); then
        echo "$1: ABI headers $(utc "$2"), SDK $(utc "$3"), the SDK is older"
    elif (( gap > pair_gap )); then
        echo "$1: ABI headers $(utc "$2"), SDK $(utc "$3"), over $(( pair_gap / 60 )) minutes apart"
    fi
}

extract() {
    case "$1" in
        *.7z) 7z x -y "$1" > /dev/null ;;
        *.tar.xz) tar xf "$1" ;;
    esac
}

changes=0
deadline=$(( $(date +%s) + publish_wait ))
while :; do
    fetch mdk-abi-sdk.7z abi-before.7z
    fetch "$sdk" "$sdk"
    fetch mdk-abi-sdk.7z mdk-abi-sdk.7z
    if ! cmp -s abi-before.7z mdk-abi-sdk.7z; then
        changes=$(( changes + 1 ))
        (( changes < 3 )) || { echo "the MDK nightly kept changing during the download" >&2; exit 1; }
        echo "a new nightly was being published during the download; fetching again" >&2
        sleep 60
        continue
    fi
    abi_time=$(last_modified < abi-before.7z.headers)
    sdk_time=$(last_modified < "$sdk.headers")
    if [[ -n "$abi_time" && -n "$sdk_time" ]] && (( sdk_time < abi_time - clock_slack && $(date +%s) - abi_time < publish_wait )); then
        # The ABI archive of a new build is up and its SDK is still on its way
        echo "the nightly of $(utc "$abi_time") is being published; waiting for its SDK" >&2
        arrived=
        while (( $(date +%s) < deadline )); do
            sleep 120
            t=$(remote_modified "$sdk")
            if [[ -n "$t" ]] && (( t >= abi_time - clock_slack )); then arrived=1; break; fi
        done
        [[ -n "$arrived" ]] && continue
    fi
    break
done

extract "$sdk"
# The ABI archive holds mdk-sdk/include/abi/mdk/*.h, laid over the SDK.
extract mdk-abi-sdk.7z

problem=$(pair_problem "by upload time" "$abi_time" "$sdk_time")
[[ -n "$problem" ]] || problem=$(pair_problem "by build time" "$(file_time mdk-sdk/include/abi/mdk/global.h)" "$(file_time "mdk-sdk/$libmdk")")
if [[ -n "$problem" ]]; then
    if [[ "$channel" == release ]]; then
        echo "the MDK ABI headers and SDK are from different builds ($problem); the release ABI headers lag the release SDK, use nightly" >&2
    else
        echo "the MDK ABI headers and SDK are from different builds ($problem); try again later" >&2
    fi
    exit 1
fi

mkdir -p "$(dirname "$dest")"
rm -rf "$dest"
mv mdk-sdk "$dest"
cd "$root"
rm -rf "$work"

grep -E "define MDK_(ABI_VERSION|MAJOR|MINOR|MICRO)" "$dest/include/abi/mdk/global.h"
