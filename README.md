# Canon Cinema RAW Light plugin for libmdk

Plays `.CRM` clips in [MDK](https://github.com/wang-bin/mdk-sdk) through Canon's CRM SDK, developed on
the GPU:

- **CUDA** (Windows, Linux): frames stay in GPU memory and go to MDK as CUDA buffers (zero-copy).
- **Metal** (macOS): the SDK's output buffer is sampled by MDK as a texture over the same memory on
  Apple silicon (zero-copy); elsewhere a GPU blit copies it into a texture.
- **OpenCL** (Windows, Linux, for GPUs without CUDA): developed on the GPU, read back to host memory,
  since MDK takes no OpenCL memory.

The SDK decompresses a frame on the CPU and develops it on the GPU. The plugin keeps up to three
frames decompressing ahead while the GPU develops the current one: 2.2-2.5 times the frame rate of
one frame at a time. Measured on a 6000×3164 C80 clip:

| | RTX 5090, Windows | RTX 5060 Ti, Linux |
|---|---|---|
| full resolution, `depth=0` | 22 fps | 22.5 fps |
| full resolution | 49 fps | 53 fps |
| `scale=1/2` | 125 fps | 191 fps |
| `scale=1/4` | 232 fps | 381 fps |

Audio is not decoded here: MDK reads the clip's PCM tracks with its own FFmpeg reader, as it does for
BRAW.

## Using it

Put `mdk-crm.dll` / `libmdk-crm.so` / `libmdk-crm.dylib` beside libmdk, then:

```cpp
SetGlobalOption("plugins", "mdk-crm");
player.setDecoders(MediaType::Video, {"CRM:scale=1920x1080", "FFmpeg"});
```

The Canon SDK is not included; get it from Canon. The plugin looks for it in the global option or
environment variable `CRMSDK_DIR` (the library itself, the directory holding it, or the SDK's own
`CRMSDK` directory), then beside the plugin, then beside the executable. Ship, from the SDK:

| Platform | Files |
|---|---|
| Windows | `Win_Lib/x64/*` (`CrmSdk.dll`, its runtimes and `CrmRsc*.bin`) |
| Linux | `Linux_Lib/*` |
| macOS | `CrmSdk.framework` (universal), in the app's `Frameworks` |

plus the `CanonLUTs` folder beside the library (or name the directory containing it with
`CRMSDK_LUTS_DIR`). The LUTs are needed for HDR and colour-space conversions.

On Windows the library must keep the exact name `CrmSdk.dll`: the SDK aborts the process otherwise.

### Decoder options

| Option | Values | Default |
|---|---|---|
| `gpu` | `auto`, `cuda`, `opencl`, `metal` | `auto`: CUDA, then OpenCL; Metal on macOS |
| `device` | GPU index, or a part of its name | the first GPU the SDK supports |
| `copy` | `1`: hand MDK host memory instead of GPU memory | `0` |
| `scale` | `1/2`, `1/4`, `WxH`, `W` | full resolution |
| `size` | `WxH` (same as `scale=WxH`) | |
| `depth` | frames decompressed ahead, `0`-`3` | `3` |
| `gamma` | `rec709`, `clog2`, `clog3`, `clog`, `linear`, `pq`, `hlg`, `widedr`, `canon709`, `cmt709`, `cmtpq`, `cmthlg`, `cmtdci`, `dci` | `rec709` |
| `gamut` | `bt709`, `bt2020`, `cinema`, `dcip3`, `dcip3plus`, `aces`, `original` | `bt709` |

`scale=WxH` picks the smallest of full, half and quarter resolution that still covers the picture as
shown fitted into `WxH`, so `scale=1920x1080` decodes a 6K clip at 3000×1582. The levels are the SDK's
decode levels, which skip work in the decompression itself. Options can also change while playing,
through `player.setProperty("video.decoder", "scale=1/4")`.

Frames are RGBA, 16 bits per sample. The stream's colour space follows `gamma` and `gamut`; camera log
curves and Cinema Gamut have no MDK equivalent and are left unspecified.

### Constraints

- **The plugin must be built against the MDK build it is loaded into.** On an ABI mismatch libmdk only
  logs `Plugin '…' was built against mdk::abi …, does not match runtime version …. MAY CRASH!` and
  loads it anyway. The release ABI headers on SourceForge lag the release SDK, so build against the
  nightly unless both match.
- The Canon SDK is a process singleton bound to one GPU. The first clip opened decides the GPU for the
  process; a later clip asking for another lane is refused.
- The SDK supports only some GPUs. A lane whose GPUs it does not support is refused at initialisation
  (for example an AMD iGPU through Mesa's rusticl), and `gpu=auto` tries the next one.
- With CUDA the SDK forbids its calls from overlapping any other CUDA call in the process. The plugin
  takes the SDK's lock around its own CUDA work and around MDK's mapping of its frames; CUDA work that
  other parts of the application do outside MDK is not covered.

## Building

Needs CMake 3.20+, Ninja, Rust, and the MDK SDK with its C++ ABI headers:

```sh
scripts/fetch-mdk-sdk.sh windows|linux|macos [nightly|release]   # into ./mdk-sdk
```

The Rust core depends on [canon-raw](https://github.com/AdrianEddy/canon-raw) by git.

**Windows** (clang-cl: MSVC rejects a `constexpr` in MDK's headers; libmdk is built with clang-cl too),
from a VS developer prompt:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build
```

**Linux**: clang and libc++. libmdk uses libc++, and the plugin interface passes `std::` types, so the
plugin must too. libc++ is linked statically and kept private, because the `libc++.so.1` shipped with
libmdk dates from 2020 and lacks functions current libc++ headers call. Build on an old distribution
for an old glibc floor; CI uses Debian 10 (glibc 2.28).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

**macOS** (universal):

```sh
rustup target add aarch64-apple-darwin x86_64-apple-darwin
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"
cmake --build build
```

`-DCORE_CARGO_ARGS=...` passes extra arguments to cargo; `-DCORE_TARGET_DIR=...` moves its target dir.

## Testing

`-DMDK_CRM_TOOLS=ON` builds test programs. Run them from a directory holding libmdk (and its FFmpeg)
and the plugin:

| Program | What it checks |
|---|---|
| `crm_read <clip> [-d CRM:opts] [-n N] [-s ms] [-o f.ppm] [-i k]` | decode rate through MDK's FrameReader, timestamps, frame `k` written out |
| `crm_process <clip> [-d CRM:opts] [-w W -h H] [-s ms] [-n N]` | the way Gyroflow's processing player reads: no audio, no clock, every frame to host RGBA |
| `crm_render <clip> [-d CRM:opts] [-w W -h H] [-s ms] [-k steps] [-p s] [-r opts] [-o f.ppm]` | Windows: playback through MDK's Player into a D3D11 texture, frame stepping, options changed while playing |

## License

MIT OR Apache-2.0. The Canon CRM SDK is Canon's and is covered by its own licence.
