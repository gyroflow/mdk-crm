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

plus the `CanonLUTs` folder beside the library or the framework (or name the directory containing
it with `CRMSDK_LUTS_DIR`). Every develop recipe needs the LUTs, the default Rec.709 one too: without
them a clip fails to open with the SDK's "The necessary LUT is not available".

On Windows the library must keep the exact name `CrmSdk.dll`: the SDK aborts the process otherwise.

### Decoder options

| Option | Values | Default |
|---|---|---|
| `gpu` | `auto`, `cuda`, `opencl`, `metal` (`no` is ignored: the SDK develops on the GPU only) | `auto`: CUDA, then OpenCL; Metal on macOS |
| `device` | GPU index, or a part of its name | the first GPU the SDK supports |
| `copy` | `1`: hand MDK host memory instead of GPU memory | `0` |
| `scale` | `1/2`, `1/4`, `WxH`, `W` | full resolution |
| `size` | `WxH` (same as `scale=WxH`) | |
| `depth` | frames decompressed ahead, `0`-`3` | `3` |
| `gamma` | `rec709`, `clog2`, `clog3`, `clog`, `linear`, `pq`, `hlg`, `widedr`, `canon709`, `cmt709`, `cmtpq`, `cmthlg`, `cmtdci`, `dci` | `rec709` |
| `gamut` | `bt709`, `bt2020`, `cinema`, `dcip3`, `dcip3plus`, `aces`, `original` | `bt709` |

`scale=WxH` picks the smallest of full, half and quarter resolution that still covers the picture as
shown fitted into `WxH`, so `scale=1920x1080` decodes a 6K clip at 3000×1582. The levels are the SDK's
decode levels, which skip work in the decompression itself. MDK sets `scale` from the player's video
surface size too (`setVideoSurfaceSize`): a player with a small surface, such as a processing player
that reads frames to memory, gets quarter resolution unless its decoder string names a `scale` (a
`scale` in the decoder string wins when the clip opens).

`scale`, `size` and `copy` also change while playing, through
`player.setProperty("video.decoder", "scale=1/4")`. The others apply when a clip is opened.

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
  takes the SDK's lock around its own CUDA work and around MDK's creation, mapping and release of its
  frames. The application's own CUDA work (CUDA interop, FFmpeg decoding or encoding on CUDA, other
  CUDA libraries) is the application's to wrap, with the two functions the plugin exports (look them
  up in the loaded plugin, e.g. `GetProcAddress(GetModuleHandleW(L"mdk-crm.dll"), ...)`):

  ```c
  void mdk_crm_cuda_exclusive(void (*fn)(void* user), void* user); // fn(user) under the lock; reentrant
  int mdk_crm_cuda_in_use(void);                                   // 1 while the SDK develops on CUDA
  ```

  Take the lock whenever the plugin is loaded, not only once a clip develops on CUDA, so that work
  begun earlier cannot overlap the first develops; and hold it around the CUDA calls only, since the
  plugin cannot develop a frame, nor MDK map one, while it is held.

  Not covered by either: CUDA work MDK does for other media in the same process (its NVDEC decoder,
  the CUDA pools of other players).
- Memory per reader: the SDK's process refs, a develop ref on the GPU (about 1 GB of device memory at
  8K; refs are sized by the clip's RAW frame, 4K, 4.5K, 5.9K or 8K) and a decompress ref in host
  memory (about 3.5 GB committed with its threads); and the output frames, one buffer each (283 MB at
  8K full resolution), at most `depth` + 4 per reader (+ 2 on Metal). Process refs are kept for reuse
  until the process exits: releasing one soon after creating it can crash the SDK on Windows. So a
  process keeps the refs of the most readers it ever had open at once.
- A clip cut short (a recording that lost power, an interrupted copy) plays to the last frame whose
  data is in the file. A frame that fails to develop is skipped. When three in a row fail (a damaged
  stretch, the GPU out of memory), the stream ends: the reader sends the end of the stream, so that a
  player or a processing loop waiting for frames is released. A seek into such a stretch shows a
  black frame at the third frame tried, then ends the stream the same way: MDK takes the end of a
  stream only after a frame. A clip none of whose first three frames develops fails to open, with
  the reason (for example missing `CanonLUTs`); the player's media status then turns `Invalid`.
- With `continue_at_end` and a loop count, MDK does not restart the loop of a stream that has no
  audio being decoded beside it (a clip without audio tracks, or a player with audio decoders off):
  the player stays at the end of the clip or of its playback range, playing. Seeks still work there
  (the end of the stream and the frames at the end of the range are handed to MDK on a thread of
  their own, since MDK holds on to them), so loop by seeking to the start of the range when the
  position stops at its end.

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

`-DCORE_CARGO_ARGS=...` replaces the arguments passed to cargo (default `--locked`: the dependencies are
the ones `core/Cargo.lock` names); `-DCORE_TARGET_DIR=...` moves its target dir.

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
