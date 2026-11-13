/*
 * Play a clip through the plugin with MDK's Player, rendering into an
 * offscreen D3D11 texture the way an application does.
 *
 *   crm_render <clip> [-d "CRM:opt=val:..."] [-w width] [-h height] [-s seek_ms] [-p play_seconds] [-o out.ppm]
 *
 * Renders the first frame (at -s), writes it to -o, then plays for -p seconds
 * and reports how many distinct frames were presented.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include <d3d11.h> // before RenderAPI.h, which declares D3D11RenderAPI only after it
#include "mdk/Player.h"
#include "mdk/RenderAPI.h"
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <thread>

using namespace MDK_NS;
using namespace std;
using clk = chrono::steady_clock;

template<typename T>
struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
    operator T*() const { return p; }
};

static bool save(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* tex, const char* path)
{
    D3D11_TEXTURE2D_DESC d{};
    tex->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    Com<ID3D11Texture2D> staging;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &staging)))
        return false;
    ctx->CopyResource(staging, tex);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m)))
        return false;
    FILE* fp = fopen(path, "wb");
    if (fp) {
        fprintf(fp, "P6\n%u %u\n255\n", d.Width, d.Height);
        string row(d.Width * 3, '\0');
        for (UINT y = 0; y < d.Height; ++y) {
            const auto src = (const uint8_t*)m.pData + (size_t)y * m.RowPitch;
            for (UINT x = 0; x < d.Width; ++x) {
                row[x * 3 + 0] = (char)src[x * 4 + 0];
                row[x * 3 + 1] = (char)src[x * 4 + 1];
                row[x * 3 + 2] = (char)src[x * 4 + 2];
            }
            fwrite(row.data(), 1, row.size(), fp);
        }
        fclose(fp);
    }
    ctx->Unmap(staging, 0);
    return fp != nullptr;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: crm_render <clip> [-d decoder] [-w width] [-h height] [-s seek_ms] [-p seconds] [-o out.ppm]\n");
        return 2;
    }
    string url = argv[1], decoder = "CRM", out, reopt;
    int width = 1920, height = 1080, steps = 0, audio = 0;
    int64_t seekMs = 0;
    double playSeconds = 0;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-d")) decoder = argv[i + 1];
        else if (!strcmp(argv[i], "-w")) width = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-h")) height = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) seekMs = atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "-p")) playSeconds = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-o")) out = argv[i + 1];
        else if (!strcmp(argv[i], "-k")) steps = atoi(argv[i + 1]);  // frame steps (negative: backwards)
        else if (!strcmp(argv[i], "-r")) reopt = argv[i + 1];         // decoder options applied mid-playback
        else if (!strcmp(argv[i], "-a")) audio = atoi(argv[i + 1]);   // 1: keep audio decoding on, as a player does
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    setLogHandler([](LogLevel, const char* msg) { fputs(msg, stderr); });
    SetGlobalOption("plugins", "mdk-crm");

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                 D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("FAILED: no D3D11 device\n");
        return 1;
    }
    D3D11_TEXTURE2D_DESC td{};
    td.Width = width;
    td.Height = height;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    Com<ID3D11Texture2D> tex;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex))) {
        printf("FAILED: render target\n");
        return 1;
    }

    Player p;
    p.setDecoders(MediaType::Video, {decoder});
    if (!audio)
        p.setDecoders(MediaType::Audio, {});
    p.setMute(true);
    p.setMedia(url.c_str());
    D3D11RenderAPI ra(ctx, tex);
    p.setRenderAPI(&ra);
    p.setVideoSurfaceSize(width, height);
    p.setBackgroundColor(1, 0, 1, 1); // magenta shows anything left unrendered

    atomic<int> loaded = 0;
    const auto t0 = clk::now();
    p.prepare(seekMs, [&](int64_t pos, bool*) {
        loaded = pos >= 0 ? 1 : -1;
        printf("prepared at %lld ms\n", (long long)pos);
        return true;
    });
    while (!loaded && clk::now() - t0 < chrono::seconds(60))
        this_thread::sleep_for(chrono::milliseconds(5));
    if (loaded <= 0) {
        printf("FAILED: prepare\n");
        return 1;
    }
    const auto& info = p.mediaInfo();
    if (!info.video.empty())
        printf("stream: %dx%d @ %.3f fps, %lld frames, %lld ms\n", info.video[0].codec.width, info.video[0].codec.height,
               info.video[0].codec.frame_rate, (long long)info.video[0].frames, (long long)info.duration);

    double ts = -1;
    while (ts < 0 && clk::now() - t0 < chrono::seconds(60)) {
        ts = p.renderVideo();
        if (ts < 0)
            this_thread::sleep_for(chrono::milliseconds(5));
    }
    if (ts < 0) {
        printf("FAILED: nothing rendered\n");
        return 1;
    }
    printf("first frame rendered after %.0f ms, t=%.4f\n", chrono::duration<double, milli>(clk::now() - t0).count(), ts);
    if (!out.empty())
        printf("%s %s\n", save(dev, ctx, tex, out.c_str()) ? "wrote" : "FAILED writing", out.c_str());

    // Frame sizes as they reach the renderer.
    atomic<int> lastW = 0, lastH = 0;
    p.onFrame<VideoFrame>([&](VideoFrame& f, int) {
        if (f && (f.width() != lastW || f.height() != lastH)) {
            lastW = f.width();
            lastH = f.height();
            printf("  frame size now %dx%d at t=%.4f\n", f.width(), f.height(), f.timestamp());
        }
        return 0;
    });

    // Step frame by frame while paused, as an editor does.
    if (steps != 0) {
        const int dir = steps > 0 ? 1 : -1;
        printf("stepping %d frames:", steps);
        double prev = ts;
        int bad = 0;
        for (int i = 0; i < abs(steps); ++i) {
            atomic<int64_t> done = -100;
            p.seek(dir, SeekFlag::FromNow | SeekFlag::Frame, [&](int64_t r) { done = r; });
            const auto s0 = clk::now();
            double t = -1;
            while (clk::now() - s0 < chrono::seconds(10)) {
                const double r = p.renderVideo();
                if (done != -100 && r >= 0 && fabs(r - prev) > 1e-6) {
                    t = r;
                    break;
                }
                this_thread::sleep_for(chrono::milliseconds(2));
            }
            printf(" %.4f", t);
            const double fps = info.video.empty() ? 25 : info.video[0].codec.frame_rate;
            if (t < 0 || fabs((t - prev) * fps - dir) > 0.01)
                ++bad;
            prev = t;
        }
        printf("\n  %s\n", bad ? "STEP ERRORS" : "every step moved exactly one frame");
    }

    if (playSeconds > 0) {
        set<double> shown;
        p.set(State::Playing);
        const auto start = clk::now();
        bool reapplied = reopt.empty();
        while (clk::now() - start < chrono::duration<double>(playSeconds)) {
            const double t = p.renderVideo();
            if (t >= 0)
                shown.insert(t);
            if (!reapplied && clk::now() - start > chrono::duration<double>(playSeconds / 2)) {
                reapplied = true;
                printf("  applying \"%s\" while playing\n", reopt.c_str());
                p.setProperty("video.decoder", reopt);
            }
            this_thread::sleep_for(chrono::milliseconds(4));
        }
        const double fps = info.video.empty() ? 0 : info.video[0].codec.frame_rate;
        printf("played %.1f s: %zu distinct frames presented (%.1f fps, clip %.3f fps), position %lld ms\n", playSeconds,
               shown.size(), shown.size() / playSeconds, fps, (long long)p.position());
        p.set(State::Stopped);
        p.waitFor(State::Stopped, 10000);
    }
    p.onFrame<VideoFrame>(nullptr);
    return 0;
}
