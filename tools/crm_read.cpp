/*
 * Decode a clip through the plugin with MDK's FrameReader, as fast as it delivers.
 *
 *   crm_read <clip> [-d "CRM:opt=val:..."] [-n frames] [-s seek_ms] [-o frame.ppm] [-i dump_index]
 *
 * Prints the stream, the time to the first frame, the decode rate, and checks
 * that timestamps advance one frame at a time. -o writes one frame (8-bit) as PPM.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "mdk/FrameReader.h"
#include "mdk/MediaInfo.h"
#include "mdk/VideoFrame.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>
#include <string>

using namespace MDK_NS;
using namespace std;
using clk = chrono::steady_clock;

static bool save_ppm(const VideoFrame& f, const char* path)
{
    const auto rgb = f.to(PixelFormat::RGB24, f.width(), f.height());
    if (!rgb || !rgb.buffer(0) || !rgb.buffer(0)->constData())
        return false;
    FILE* fp = fopen(path, "wb");
    if (!fp)
        return false;
    fprintf(fp, "P6\n%d %d\n255\n", rgb.width(), rgb.height());
    const auto b = rgb.buffer(0);
    for (int y = 0; y < rgb.height(); ++y)
        fwrite(b->constData() + (size_t)y * b->stride(), 1, (size_t)rgb.width() * 3, fp);
    fclose(fp);
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: crm_read <clip> [-d decoder] [-n frames] [-s seek_ms] [-o out.ppm] [-i index]\n");
        return 2;
    }
    string url = argv[1], decoder = "CRM", out;
    int64_t maxFrames = 100, seekMs = -1, dumpIndex = 0;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-d")) decoder = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) maxFrames = atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) seekMs = atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "-o")) out = argv[i + 1];
        else if (!strcmp(argv[i], "-i")) dumpIndex = atoll(argv[i + 1]);
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    setLogHandler([](LogLevel, const char* msg) { fputs(msg, stderr); });
    SetGlobalOption("plugins", "mdk-crm");

    const char* readerName = FrameReader::supportedBy(url, MediaType::Video);
    printf("reader for this url: %s\n", readerName ? readerName : "(none)");
    auto fr = FrameReader::create(readerName ? readerName : "CRM");
    fr->setMedia(url);
    fr->setDecoders(MediaType::Video, {decoder});
    fr->setDecoders(MediaType::Audio, {});
    fr->setActiveTracks(MediaType::Audio, {});
    fr->setBufferFrames(MediaType::Video, 2);

    const auto t0 = clk::now();
    clk::time_point tFirst{};
    atomic<int64_t> n = 0;
    double firstTs = -1, lastTs = -1, dt = 0;
    int jumps = 0;
    int w = 0, h = 0;
    string fmt;
    bool dumped = false;
    promise<void> done;
    atomic<bool> finished = false;
    auto finish = [&] {
        if (!finished.exchange(true))
            done.set_value();
    };

    fr->onRead<VideoFrame>([&](const VideoFrame& v, int) {
        if (!v) {
            if (v.timestamp() == TimestampEOS)
                finish();
            return true;
        }
        if (finished)
            return true;
        const auto k = n++;
        if (k == 0) {
            tFirst = clk::now();
            firstTs = v.timestamp();
            w = v.width();
            h = v.height();
            ostringstream os;
            os << v.format();
            fmt = os.str();
            if (v.duration() > 0)
                dt = v.duration();
        } else if (dt > 0 && fabs(v.timestamp() - lastTs - dt) > dt / 2) {
            ++jumps;
        }
        lastTs = v.timestamp();
        if (!out.empty() && !dumped && k == dumpIndex) {
            dumped = true;
            printf("dump of frame %lld (t=%.4f): %s\n", (long long)k, v.timestamp(), save_ppm(v, out.c_str()) ? out.c_str() : "FAILED");
        }
        if (k + 1 >= maxFrames)
            finish();
        return true;
    });

    MediaInfo info;
    bool opened = false;
    fr->start(seekMs > 0 ? seekMs : 0, [&](const MediaInfo* mi, int64_t pos, bool*) {
        if (mi) {
            info = *mi;
            opened = true;
            cout << "opened at " << pos << " ms: " << info << endl;
        } else {
            finish();
        }
        return true;
    });
    auto f = done.get_future();
    if (f.wait_for(chrono::seconds(300)) != future_status::ready)
        printf("TIMEOUT\n");
    const auto t1 = clk::now();
    fr->stop();
    fr->waitFor(State::Stopped, 10000);
    if (!opened || n == 0) {
        printf("FAILED: %s\n", opened ? "no frames" : "not opened");
        return 1;
    }
    const double first = chrono::duration<double>(tFirst - t0).count();
    const double total = chrono::duration<double>(t1 - tFirst).count();
    printf("frames: %lld  %dx%d %s  first frame after %.0f ms (t=%.4f)  last t=%.4f  rate %.1f fps  timestamp gaps: %d\n",
           (long long)n.load(), w, h, fmt.c_str(), first * 1000, firstTs, lastTs, n > 1 ? (n - 1) / total : 0.0, jumps);
    return 0;
}
