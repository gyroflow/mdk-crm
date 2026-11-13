/*
 * Read a clip through the plugin the way Gyroflow's processing player does:
 * no audio, no clock (onSync), every frame converted to host RGBA at a given size.
 *
 *   crm_process <clip> [-d "CRM:opt=val:..."] [-w width] [-h height] [-s start_ms] [-n frames]
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "mdk/Player.h"
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

using namespace MDK_NS;
using namespace std;
using clk = chrono::steady_clock;

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: crm_process <clip> [-d decoder] [-w width] [-h height] [-s start_ms] [-n frames]\n");
        return 2;
    }
    string url = argv[1], decoder = "CRM";
    int width = 0, height = 0;
    int64_t startMs = 0, maxFrames = 100;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-d")) decoder = argv[i + 1];
        else if (!strcmp(argv[i], "-w")) width = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-h")) height = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) startMs = atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "-n")) maxFrames = atoll(argv[i + 1]);
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    setLogHandler([](LogLevel, const char* msg) { fputs(msg, stderr); });
    SetGlobalOption("plugins", "mdk-crm");

    Player p;
    p.setDecoders(MediaType::Video, {decoder});
    p.setMedia(url.c_str());
    p.setDecoders(MediaType::Audio, {});
    p.setMute(true);
    p.onSync([] { return DBL_MAX; });

    atomic<int64_t> n = 0, bad = 0, gaps = 0;
    atomic<bool> done = false;
    double prev = -1, first = -1;
    int outW = 0, outH = 0;
    uint64_t sum = 0;
    const auto t0 = clk::now();
    clk::time_point tFirst{};
    p.onFrame<VideoFrame>([&](VideoFrame& v, int) {
        if (done)
            return 0;
        if (!v || v.timestamp() == TimestampEOS) {
            done = true;
            return 0;
        }
        const auto rgba = v.to(PixelFormat::RGBA, width ? width : v.width(), height ? height : v.height());
        const auto data = rgba.bufferData();
        if (!data) {
            ++bad;
        } else {
            outW = rgba.width();
            outH = rgba.height();
            const auto mid = data + (size_t)rgba.height() / 2 * rgba.bytesPerLine() + (size_t)rgba.width() / 2 * 4;
            sum += mid[0] + mid[1] + mid[2];
        }
        if (n == 0) {
            tFirst = clk::now();
            first = v.timestamp();
        } else if (fabs(v.timestamp() - prev - 0.04) > 0.02) { // the test clip is 25 fps
            ++gaps;
        }
        prev = v.timestamp();
        if (++n >= maxFrames)
            done = true;
        return 0;
    });
    p.setVideoSurfaceSize(64, 64);
    p.prepare(startMs);
    p.set(State::Playing);
    while (!done && clk::now() - t0 < chrono::seconds(300))
        this_thread::sleep_for(chrono::milliseconds(10));
    const double secs = chrono::duration<double>(clk::now() - tFirst).count();
    p.onFrame<VideoFrame>(nullptr);
    p.set(State::Stopped);
    p.waitFor(State::Stopped, 10000);
    printf("frames: %lld to host RGBA %dx%d, first t=%.4f last t=%.4f, first after %.0f ms, %.1f fps, unmapped: %lld, "
           "timestamp gaps: %lld, mean centre value %.0f\n",
           (long long)n.load(), outW, outH, first, prev, chrono::duration<double, milli>(tFirst - t0).count(),
           n > 1 ? (n - 1) / secs : 0.0, (long long)bad.load(), (long long)gaps.load(), n ? sum / (3.0 * n) : 0.0);
    return n > 0 && bad == 0 ? 0 : 1;
}
