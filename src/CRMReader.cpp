/*
 * Canon Cinema RAW Light (.CRM) plugin for libmdk.
 *
 * Decoder options, e.g. player.setDecoders(MediaType::Video, {"CRM:scale=1920x1080", "FFmpeg"}):
 *   gpu=auto|cuda|opencl|metal  the GPU the SDK develops on (auto: CUDA, then OpenCL; Metal on macOS)
 *   device=N|name               the GPU's index or a part of its name
 *   copy=0|1                    1: hand MDK host memory instead of GPU memory
 *   scale=1/2|1/4|WxH|W         decode at a reduced level; WxH picks the smallest level that still
 *   size=WxH                    covers the picture fitted into WxH (W alone: WxW)
 *   depth=0..3                  frames decompressed ahead on the CPU; 0: one frame at a time
 *   gamma=rec709|clog2|clog3|...  develop recipe (default rec709)
 *   gamut=bt709|bt2020|cinema|...
 * scale, size and copy also change while playing; the others apply when a clip is opened.
 *
 * The Canon SDK is found through the global option or environment variable CRMSDK_DIR, then beside
 * this plugin, then beside the executable. CRMSDK_LUTS_DIR names the directory containing CanonLUTs
 * when they are not beside the SDK.
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "mdk/FrameReader.h"
#include "mdk/MediaInfo.h"
#include "mdk/VideoFrame.h"
#include "Gpu.h"
#include "crm_core.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#if (__APPLE__ + 0)
#include <mach-o/dyld.h>
#endif
#endif

using namespace std;

MDK_NS_BEGIN

namespace {

// ─────────────────────────────────────────────────────────────────────
// Where the SDK is
// ─────────────────────────────────────────────────────────────────────

string dir_of(string path)
{
    const auto p = path.find_last_of("/\\");
    return p == string::npos ? string() : path.substr(0, p);
}

#ifdef _WIN32
string to_utf8(const wchar_t* w)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1)
        return {};
    string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    s.resize(n - 1);
    return s;
}

string module_path(HMODULE h)
{
    vector<wchar_t> buf(32768);
    const auto n = GetModuleFileNameW(h, buf.data(), (DWORD)buf.size());
    return n ? to_utf8(buf.data()) : string();
}
#endif

string plugin_dir()
{
#ifdef _WIN32
    HMODULE h = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)(void*)&plugin_dir, &h);
    return dir_of(module_path(h));
#else
    Dl_info info{};
    if (dladdr((void*)&plugin_dir, &info) && info.dli_fname)
        return dir_of(info.dli_fname);
    return {};
#endif
}

string exe_dir()
{
#ifdef _WIN32
    return dir_of(module_path(nullptr));
#elif (__APPLE__ + 0)
    char buf[4096];
    uint32_t n = sizeof(buf);
    return _NSGetExecutablePath(buf, &n) == 0 ? dir_of(buf) : string();
#else
    char buf[4096];
    const auto n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? dir_of(string(buf, (size_t)n)) : string();
#endif
}

// An environment variable as UTF-8 (getenv gives the ANSI code page on Windows, which loses other characters).
string env_utf8(const char* key)
{
#ifdef _WIN32
    wchar_t wkey[128] = {};
    if (MultiByteToWideChar(CP_UTF8, 0, key, -1, wkey, 128) <= 0)
        return {};
    vector<wchar_t> buf(32768);
    const DWORD n = GetEnvironmentVariableW(wkey, buf.data(), (DWORD)buf.size());
    return n > 0 && n < buf.size() ? to_utf8(buf.data()) : string();
#else
    const auto e = getenv(key);
    return e ? e : "";
#endif
}

string option(const char* key)
{
    const auto& v = GetGlobalOption(key);
    if (const auto s = get_if<string>(&v); s && !s->empty())
        return *s;
    return env_utf8(key);
}

// The core's diagnostics go where the plugin's own do.
void core_log(int32_t level, const char* msg)
{
    clog << (level == 1 ? "CRM error: " : level == 2 ? "CRM warning: " : "CRM: ") << msg << endl;
}

// Run `f` under the Canon SDK's CUDA exclusion (Interface Spec 9.2.1).
template<typename F>
void cuda_exclusive(F&& f)
{
    using Fn = remove_reference_t<F>;
    crmc_cuda_exclusive([](void* user) {
        try {
            (*static_cast<Fn*>(user))();
        } catch (...) {
        }
    }, (void*)&f);
}

int32_t sync_gpu(void* gpu)
{
    return static_cast<Gpu*>(gpu)->sync();
}

// ─────────────────────────────────────────────────────────────────────
// The process-wide SDK binding: one GPU for every clip
// ─────────────────────────────────────────────────────────────────────

mutex g_engine_mtx;
shared_ptr<Gpu> g_gpu;
atomic<bool> g_cuda_in_use = false; // g_gpu is the CUDA lane's
// GPUs the SDK was handed and then refused. Kept: the SDK may hold on to their stream or queue.
vector<shared_ptr<Gpu>> g_refused;

shared_ptr<Gpu> engine(int lane, const string& device, string* err)
{
    const scoped_lock lock(g_engine_mtx);
    if (g_gpu) {
        if (lane && lane != g_gpu->lane()) {
            *err = string("the Canon SDK already develops on ") + GpuLaneName(g_gpu->lane())
                + " in this process and cannot switch to " + GpuLaneName(lane);
            return nullptr;
        }
        return g_gpu;
    }
    const vector<string> dirs{option("CRMSDK_DIR"), plugin_dir(), exe_dir()};
    vector<const char*> pdirs;
    for (const auto& d : dirs)
        pdirs.push_back(d.c_str());
    char buf[2048] = {};
    // The library first: without it no GPU is worth setting up. Not
    // remembered when it fails, so the SDK may be installed meanwhile.
    if (crmc_sdk_load(pdirs.data(), pdirs.size(), buf, sizeof(buf)) != 0) {
        *err = buf;
        return nullptr;
    }
    const auto luts = option("CRMSDK_LUTS_DIR");
    string errs;
    for (int l : lane ? vector<int>{lane} : GpuLanes()) {
        // Without a device named, every device of the lane in turn: the SDK
        // supports some GPUs and refuses others.
        for (int i = 0; i < (device.empty() ? 8 : 1); ++i) {
            string e;
            auto gpu = CreateGpu(l, device.empty() ? std::to_string(i) : device, &e);
            if (!gpu) {
                if (i == 0)
                    errs += string(errs.empty() ? "" : "; ") + GpuLaneName(l) + ": " + e;
                break;
            }
            const auto desc = gpu->desc();
            buf[0] = 0;
            const int r = crmc_sdk_init(pdirs.data(), pdirs.size(), luts.c_str(), &desc, buf, sizeof(buf));
            if (r == 0) {
                clog << "CRM: the Canon SDK develops on " << GpuLaneName(l) << " (" << gpu->deviceName() << ")" << endl;
                g_gpu = gpu;
                g_cuda_in_use = l == CRMC_LANE_CUDA;
                return gpu;
            }
            if (r == -2)
                g_refused.push_back(gpu);
            errs += string(errs.empty() ? "" : "; ") + GpuLaneName(l) + " " + gpu->deviceName() + ": " + buf;
        }
    }
    *err = "no GPU the Canon SDK can develop on: " + errs;
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────
// Options
// ─────────────────────────────────────────────────────────────────────

constexpr int kInvalid = -2;

string lower(string s)
{
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)tolower(c); });
    s.erase(remove_if(s.begin(), s.end(), [](char c) { return c == '_' || c == '-' || c == '.' || c == ' '; }), s.end());
    return s;
}

// CrmGamma values.
int gamma_value(const string& name)
{
    static const pair<const char*, int> names[] = {
        {"linear", 0x0}, {"canonlog", 0x1}, {"clog", 0x1}, {"rec709", 0x2}, {"bt709", 0x2},
        {"widedr", 0x3}, {"dci", 0x4}, {"canonlog2", 0x5}, {"clog2", 0x5}, {"canonlog3", 0x6},
        {"clog3", 0x6}, {"cmt709", 0x41}, {"canon709", 0x42}, {"cmtpq", 0x52}, {"cmthlg", 0x53},
        {"cmtdci", 0x54}, {"pq", 0x20002}, {"bt2100pq", 0x20002}, {"hlg", 0x20003}, {"bt2100hlg", 0x20003},
    };
    const auto n = lower(name);
    for (const auto& [k, v] : names)
        if (n == k)
            return v;
    return kInvalid;
}

// CrmColorSpace values.
int gamut_value(const string& name)
{
    static const pair<const char*, int> names[] = {
        {"bt709", 0x0}, {"rec709", 0x0}, {"dcip3plus", 0x1}, {"cinema", 0x2}, {"cinemagamut", 0x2},
        {"bt2020", 0x3}, {"rec2020", 0x3}, {"aces", 0x11}, {"dcip3", 0x10001}, {"p3", 0x10001},
        {"original", (int)0x99999999u},
    };
    const auto n = lower(name);
    for (const auto& [k, v] : names)
        if (n == k)
            return v;
    return kInvalid;
}

// How MDK should read what the SDK developed. Camera log curves and camera
// gamuts have no MDK value and stay unspecified.
ColorSpace color_space(int gamma, int gamut)
{
    ColorSpace cs{};
    cs.matrix = ColorSpace::Matrix::RGB;
    cs.range = ColorSpace::Range::Full;
    switch (gamut) {
    case 0x0: cs.primaries = ColorSpace::Primary::BT709; break;
    case 0x3: cs.primaries = ColorSpace::Primary::BT2020; break;
    case 0x1:
    case 0x10001: cs.primaries = ColorSpace::Primary::DCI_P3; break;
    default: cs.primaries = ColorSpace::Primary::UNSPECIFIED; break;
    }
    switch (gamma) {
    case 0x0: cs.transfer = ColorSpace::Transfer::LINEAR; break;
    case 0x2:
    case 0x3:
    case 0x41:
    case 0x42: cs.transfer = ColorSpace::Transfer::BT709; break;
    case 0x52:
    case 0x20002: cs.transfer = ColorSpace::Transfer::PQ; break;
    case 0x53:
    case 0x20003: cs.transfer = ColorSpace::Transfer::HLG; break;
    default: cs.transfer = ColorSpace::Transfer::UNSPECIFIED; break;
    }
    return cs;
}

string percent_decoded(const string& s)
{
    string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) {
            out += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

/*
  A file path from MDK's url: a plain path as it is; a file: URL (RFC 8089)
  with its authority taken into account: none or "localhost" is this machine,
  another host is a network share (\\host\share\... on Windows).
 */
string url_to_path(string url)
{
    if (url.rfind("file:", 0) != 0)
        return url;
    url.erase(0, 5);
    string host;
    if (url.rfind("//", 0) == 0) {
        url.erase(0, 2);
        const auto slash = url.find('/');
        host = url.substr(0, slash);
        url = slash == string::npos ? string() : url.substr(slash);
        string h = host;
        transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return (char)tolower(c); });
        if (h == "localhost")
            host.clear();
    }
    url = percent_decoded(url);
    host = percent_decoded(host);
#ifdef _WIN32
    if (!host.empty()) {
        replace(url.begin(), url.end(), '/', '\\');
        return "\\\\" + host + url;
    }
    if (url.size() > 2 && url[0] == '/' && url[2] == ':')
        url.erase(0, 1);
#else
    if (!host.empty())
        return "//" + host + url;
#endif
    return url;
}

} // namespace

class CRMReader final : public FrameReader
{
public:
    CRMReader() = default;
    ~CRMReader() override {
        stopThread();
        joinThreads();
        releaseMdkPool();
    }
    const char* name() const override { return "CRM"; }
    void setTimeout(int64_t, TimeoutCallback) override {}
    bool load() override;
    bool unload() override;
    bool seekTo(int64_t msec, SeekFlag flag, int id) override;
    bool pause(bool value = true) override;
    int64_t buffered(int64_t* bytes = nullptr, float* percent = nullptr) const override {
        if (bytes)
            *bytes = 0;
        if (percent)
            *percent = 1.0f;
        return 0;
    }

protected:
    void onPropertyChanged(const std::string& key, const std::string& value) override;

private:
    /*
      One read for the decode thread. Each seek starts a new generation, and a
      read of an older one is never queued again nor its frame delivered: the
      generation is compared under the queue's lock, where seeks bump it.
     */
    struct Request {
        uint64_t index = 0;
        int seekId = 0;          // > 0: the seek this read completes
        bool seekWaitFrame = true;
        uint64_t gen = 0;
        bool onlyIfPlaying = false; // the next frame of the stream: queued only while playing
    };

    /*
      What MDK may hold on to: the end of the stream, and with it the frames at
      or past the end of the playback range (see tailLoop). Handed over on a
      thread of its own, so the decode thread stays free for seeks.
     */
    struct Tail {
        VideoFrame frame;        // invalid: none
        bool eos = false;        // then the end of the stream
        optional<Request> next;  // or the stream's next read, once the frame is taken
        uint64_t gen = 0;        // dropped when a seek came since
        uint64_t resumes = 0;    // resumes_ when it was posted
    };

    void parseDecoderOptions();
    int chooseLevel() const;
    void applyScale();
    bool openClip(int resolution, string* err);
    bool developFirstFrames(string* err);
    // Queue a read. A seek drops the reads queued before it. False once unloaded,
    // or for a read of an older generation.
    bool push(Request r, bool seek);
    // Queue the stream's next read, or note where it stopped when paused.
    void continueChain(const Request& next, uint64_t resumesAtStart);
    void stopThread();
    void joinThreads();
    void decodeLoop();
    optional<Request> process(const Request& r);
    bool decode(uint64_t index, VideoFrame* frame, string* err);
    VideoFrame blackFrame() const;
    bool heldAtRangeEnd(double t);
    void endOfStream(uint64_t gen);
    bool postTail(Tail&& t);
    void tailLoop();
    void releaseMdkPool();

    double frameTime(uint64_t i) const { return double(i) * fpsDen_ / fpsNum_; }
    int64_t frameMs(uint64_t i) const { return llround(frameTime(i) * 1000.0); }
    /*
      The frame nearest to `ms`. MDK's accurate seek is documented as the last
      frame at or before the target, but hosts seek to a frame by its time in
      whole milliseconds, truncated (qml-video-rs: frame / fps * 1000 as a float
      into seek(int64_t)), which can be up to 1 ms before the frame: flooring
      would land one frame early. Rounding lands on the frame for any rate below
      500 fps. wang-bin's BRAW reader does not floor either.
     */
    uint64_t indexAt(int64_t ms) const {
        const auto i = llround(double(ms) * fpsNum_ / (1000.0 * fpsDen_));
        return (uint64_t)clamp<long long>(i, 0, (long long)frames_ - 1);
    }

    // options: written by onPropertyChanged on any thread
    atomic<int> lane_ = 0;
    mutable mutex optMtx_;
    string device_;
    atomic<bool> copy_ = false;
    atomic<int> depth_ = 3;
    atomic<int> gamma_ = -1;
    atomic<int> gamut_ = -1;
    mutable mutex scale_mtx_;
    int scaleLevel_ = 0;
    uint32_t scaleW_ = 0, scaleH_ = 0;
    atomic<bool> scaleChanged_ = false;

    // clip: set up by load(), then used on the decode thread only
    crmc_reader* reader_ = nullptr;
    atomic<bool> loaded_ = false;
    crmc_info info_{};
    crmc_output out_{};
    uint64_t frames_ = 0;
    uint32_t fpsNum_ = 25, fpsDen_ = 1;
    ColorSpace cs_{};
    shared_ptr<Gpu> gpu_;
    shared_ptr<GpuBufferPool> pool_;
    NativeVideoBufferPool::Ptr mdkPool_;

    // decode thread
    thread thread_;
    mutex mtx_;
    condition_variable cv_;
    deque<Request> requests_;
    bool running_ = false;
    bool ready_ = false;        // load() has finished: the decode thread may start on the queue
    atomic<uint64_t> gen_ = 0;  // written under mtx_
    uint64_t resumes_ = 0;      // counts pause(false): a read in flight then continues the stream
    bool chainStopped_ = false; // the stream stopped reading on a pause
    uint64_t stoppedAt_ = 0;    // the frame it would have read next
    atomic<uint64_t> index_ = 0;
    int errors_ = 0;
    bool failed_ = false;       // the stream was ended after errors, and no frame decoded since
    atomic<bool> bufferingPending_ = false;

    // the tail thread (see Tail)
    thread tailThread_;
    mutex tailMtx_;
    condition_variable tailCv_;
    bool tailStopped_ = true;   // no clip is running: nothing is taken, and the thread leaves
    bool tailBusy_ = false;     // an item waits or is being handed over
    Tail tail_;
};

void CRMReader::parseDecoderOptions()
{
    // decoder: name:key1=val1:key2=val2
    for (const auto& d : decoders(MediaType::Video)) {
        if (const auto colon = d.find(':'); colon != string::npos && string_view(d).substr(0, colon) == name()) {
            parse(d.data() + colon);
            return;
        }
    }
}

void CRMReader::onPropertyChanged(const std::string& key, const std::string& value)
{
    const auto k = lower(key);
    const bool open = loaded_;
    const auto atOpen = [&] {
        if (open)
            clog << "CRM: '" << key << "' applies when a clip is opened, not to the clip playing" << endl;
    };
    if (k == "gpu" || k == "lane") {
        const auto v = lower(value);
        if (v == "no" || v == "none" || v == "cpu" || v == "0")
            clog << "CRM: gpu=" << value << " is ignored: the Canon SDK develops on the GPU only" << endl;
        lane_ = v == "cuda" ? CRMC_LANE_CUDA : v == "opencl" ? CRMC_LANE_OPENCL : v == "metal" ? CRMC_LANE_METAL : 0;
        atOpen();
    } else if (k == "device") {
        {
            const scoped_lock lock(optMtx_);
            device_ = value;
        }
        atOpen();
    } else if (k == "copy") {
        copy_ = atoi(value.c_str()) != 0;
    } else if (k == "depth") {
        depth_ = clamp(atoi(value.c_str()), 0, 3);
        atOpen();
    } else if (k == "gamma") {
        int g = gamma_value(value);
        if (g == kInvalid) {
            clog << "CRM: unknown gamma '" << value << "', using rec709" << endl;
            g = -1;
        }
        gamma_ = g;
        atOpen();
    } else if (k == "gamut" || k == "colorspace") {
        int g = gamut_value(value);
        if (g == kInvalid) {
            clog << "CRM: unknown gamut '" << value << "', using bt709" << endl;
            g = -1;
        }
        gamut_ = g;
        atOpen();
    } else if (k == "scale" || k == "size") {
        // MDK also sets this from the player's video surface size
        // (setVideoSurfaceSize), so a host that wants another size than its
        // surface's passes scale in the decoder string.
        const scoped_lock lock(scale_mtx_);
        scaleLevel_ = 0;
        scaleW_ = scaleH_ = 0;
        if (value.rfind("1/", 0) == 0) {
            const int s = atoi(value.c_str() + 2);
            scaleLevel_ = s >= 3 ? 4 : s >= 2 ? 2 : 1;
        } else if (const auto x = value.find('x'); x != string::npos) {
            scaleW_ = (uint32_t)strtoul(value.c_str(), nullptr, 10);
            scaleH_ = (uint32_t)strtoul(value.c_str() + x + 1, nullptr, 10);
        } else {
            scaleW_ = scaleH_ = (uint32_t)strtoul(value.c_str(), nullptr, 10);
        }
        scaleChanged_ = true;
    } else if (k == "decoder" || k == "videodecoder") {
        parse(value.data());
    }
}

/*
  The smallest decode level whose output still covers the picture as it is
  shown fitted into the requested box. An explicit 1/N is taken as is.
 */
int CRMReader::chooseLevel() const
{
    const scoped_lock lock(scale_mtx_);
    if (scaleLevel_ > 0)
        return scaleLevel_;
    if (!scaleW_ && !scaleH_)
        return 1;
    const double w = double(info_.width) * max(info_.pixel_aspect, 0.01f);
    const double h = info_.height;
    const double tw = scaleW_ ? scaleW_ : scaleH_;
    const double th = scaleH_ ? scaleH_ : scaleW_;
    const double fit = min(1.0, min(tw / w, th / h));
    for (int level : {4, 2}) {
        crmc_output o{};
        if (crmc_output_for(reader_, level, &o) != 0)
            continue;
        // The SDK floors each edge to an even number, so allow for that.
        if (double(o.width) * max(info_.pixel_aspect, 0.01f) + 2 >= w * fit && double(o.height) + 2 >= h * fit)
            return level;
    }
    return 1;
}

// Between frames: on the decode thread, or in load() before it starts.
void CRMReader::applyScale()
{
    if (!scaleChanged_.exchange(false))
        return;
    const int level = chooseLevel();
    if (level == out_.level)
        return;
    crmc_output out{};
    char err[1024] = {};
    if (crmc_set_level(reader_, level, &out, err, sizeof(err)) != 0) {
        clog << "CRM: " << err << endl;
        return;
    }
    out_ = out;
    pool_->setSize(out_.size);
    // MDK's buffer pools assume one buffer size: a new pool for the new size.
    if (mdkPool_) {
        releaseMdkPool();
        cuda_exclusive([&] { mdkPool_ = NativeVideoBufferPool::create("CUDA"); });
    }
    clog << "CRM: decoding at 1/" << out_.level << ": " << out_.width << "x" << out_.height << endl;
}

// Dropping MDK's CUDA pool releases its stream, pinned memory and context: CUDA calls.
void CRMReader::releaseMdkPool()
{
    if (mdkPool_)
        cuda_exclusive([&] { mdkPool_.reset(); });
}

bool CRMReader::openClip(int resolution, string* err)
{
    const crmc_open_params params{gamma_, gamut_, depth_, resolution};
    char ebuf[2048] = {};
    reader_ = crmc_open(url_to_path(url()).c_str(), &params, ebuf, sizeof(ebuf));
    if (!reader_)
        *err = ebuf;
    return reader_ != nullptr;
}

/*
  Develop one of the clip's first frames while loading. A recipe the SDK
  refuses (no CanonLUTs, which every recipe needs), a GPU that cannot develop
  this clip, or process refs too small for it then fail the load with the
  reason, instead of every frame failing later. A damaged first frame is no
  such reason, so the next two are tried too, as playback skips to them.
 */
bool CRMReader::developFirstFrames(string* err)
{
    void* buf = pool_->acquire();
    if (!buf) {
        *err = "out of GPU memory for a " + std::to_string(out_.size) + " byte frame";
        return false;
    }
    const auto recycle = pool_->recycler(buf);
    bool ok = false;
    for (uint64_t i = 0; i < min<uint64_t>(3, frames_) && !ok; ++i) {
        char ebuf[2048] = {};
        ok = crmc_decode(reader_, i, buf, out_.size, &sync_gpu, gpu_.get(), 0, ebuf, sizeof(ebuf)) == 0;
        if (!ok && i == 0)
            *err = ebuf; // the first frame's reason is the one reported
    }
    recycle();
    return ok;
}

bool CRMReader::load()
{
    joinThreads();
    releaseMdkPool(); // the previous clip's
    parseDecoderOptions();

    string err;
    string device;
    {
        const scoped_lock lock(optMtx_);
        device = device_;
    }
    gpu_ = engine(lane_, device, &err);
    const auto fail = [&](const string& e) {
        clog << "CRM: " << e << endl;
        dispatchEvent({.error = -1, .category = "decoder.video", .detail = "CRM: " + e});
        if (reader_) {
            crmc_close(reader_);
            reader_ = nullptr;
        }
        pool_.reset();
        return false;
    };
    if (!gpu_)
        return fail(err);
    if (!openClip(0, &err))
        return fail(err);
    crmc_get_info(reader_, &info_);
    frames_ = info_.frame_count;
    fpsNum_ = info_.fps_num ? info_.fps_num : 25;
    fpsDen_ = info_.fps_den ? info_.fps_den : 1;

    crmc_get_output(reader_, &out_);
    // A frame decoding, the read-ahead and what MDK holds for rendering; past
    // that acquire waits for MDK to return one (see GpuBufferPool).
    pool_ = make_shared<GpuBufferPool>(gpu_, 4 + (size_t)max(info_.depth, 0) + gpu_->reuseDelay());
    scaleChanged_ = true;
    applyScale();
    pool_->setSize(out_.size);

    if (!developFirstFrames(&err)) {
        constexpr int kResolution8K = 4;
        if (info_.resolution == kResolution8K)
            return fail(err);
        // The process refs were sized for the clip; try the largest once.
        clog << "CRM: the first frames could not be developed (" << err << "); trying again with process refs for 8K" << endl;
        crmc_close(reader_);
        reader_ = nullptr;
        const int level = out_.level;
        if (!openClip(kResolution8K, &err))
            return fail(err);
        crmc_get_info(reader_, &info_);
        if (level != 1) {
            char e[1024] = {};
            if (crmc_set_level(reader_, level, &out_, e, sizeof(e)) != 0)
                return fail(e);
        }
        if (!developFirstFrames(&err))
            return fail(err);
    }
    cs_ = color_space(info_.gamma, info_.color_space);

    if (gpu_->lane() == CRMC_LANE_CUDA && !copy_ && !mdkPool_)
        cuda_exclusive([&] { mdkPool_ = NativeVideoBufferPool::create("CUDA"); });

    MediaInfo info;
    info.format = "crm";
    info.streams = 1;
    info.duration = frameMs(frames_);
    for (size_t i = 0;; ++i) {
        const char* k = nullptr;
        const char* v = nullptr;
        if (!crmc_metadata(reader_, i, &k, &v))
            break;
        info.metadata.emplace(k, v);
    }
    VideoStreamInfo vsi;
    vsi.index = 0;
    vsi.frames = (int64_t)frames_;
    vsi.duration = info.duration;
    vsi.codec.codec = "crm";
    vsi.codec.width = (int)info_.width;
    vsi.codec.height = (int)info_.height;
    vsi.codec.frame_rate = float(double(fpsNum_) / fpsDen_);
    vsi.codec.format = PixelFormat::RGBA64LE;
    vsi.codec.par = info_.pixel_aspect;
    vsi.codec.color_space = cs_;
    vsi.codec.bits_per_raw_sample = (int)info_.bit_depth;
    info.video.push_back(vsi);
    static const char* const kResolutions[] = {"", "4K", "4.5K", "5.9K", "8K"};
    clog << "CRM: " << info_.width << "x" << info_.height << " @ " << fpsNum_ << "/" << fpsDen_ << ", " << frames_
         << " frames, decoding at 1/" << out_.level << " (" << out_.width << "x" << out_.height << ") on "
         << GpuLaneName(gpu_->lane()) << (copy_ || gpu_->lane() == CRMC_LANE_OPENCL ? " to host memory" : " zero-copy")
         << ", read-ahead " << info_.depth << ", process refs for "
         << (info_.resolution >= 1 && info_.resolution <= 4 ? kResolutions[info_.resolution] : "?") << endl;

    uint64_t gen = 0;
    {
        const scoped_lock lock(mtx_);
        requests_.clear();
        running_ = true;
        ready_ = false;
        chainStopped_ = false;
        gen = gen_;
    }
    {
        const scoped_lock lock(tailMtx_);
        tailStopped_ = false;
        tailBusy_ = false;
        tail_ = {};
    }
    index_ = 0;
    errors_ = 0;
    failed_ = false;
    loaded_ = true;
    bufferingPending_ = true;
    thread_ = thread([this] { decodeLoop(); });

    MediaEvent e{};
    e.category = "decoder.video";
    e.detail = "crm";
    dispatchEvent(e);

    changed(info); // may seek, for prepare(position)
    update(MediaStatus::Loaded);
    updateBufferingProgress(0);
    if (state() == State::Stopped)
        update(State::Running);
    // From the start, unless a seek (prepare(position)) came meanwhile.
    push({0, 0, true, gen, false}, false);
    // Only now does the decode thread start on the queue: a seek that came
    // during the calls above must find the reader loaded and running when its
    // frame is out, or the stream would end there.
    {
        const scoped_lock lock(mtx_);
        ready_ = true;
    }
    cv_.notify_all();
    return true;
}

bool CRMReader::unload()
{
    const bool onDecodeThread = this_thread::get_id() == thread_.get_id();
    update(MediaStatus::Unloaded);
    if (!loaded_) {
        update(State::Stopped);
        return false;
    }
    stopThread(); // the decode thread closes the clip when it leaves
    if (!onDecodeThread)
        frameAvailable(VideoFrame().setTimestamp(TimestampEOS)); // clears frames queued for rendering
    update(State::Stopped);
    return true;
}

bool CRMReader::pause(bool value)
{
    const bool changed = FrameReader::pause(value);
    if (!value) {
        // A pause can stop the stream between a frame and the read of the
        // next; resuming reads on from there. A read in flight sees the count
        // move and continues by itself.
        const scoped_lock lock(mtx_);
        ++resumes_;
        if (chainStopped_ && running_ && requests_.empty()) {
            requests_.push_back({stoppedAt_, 0, true, gen_, false});
            cv_.notify_one();
        }
        chainStopped_ = false;
    }
    return changed;
}

void CRMReader::stopThread()
{
    vector<Request> dropped;
    {
        const scoped_lock lock(mtx_);
        running_ = false;
        chainStopped_ = false;
        dropped.assign(requests_.begin(), requests_.end());
        requests_.clear();
        cv_.notify_all();
    }
    {
        const scoped_lock lock(tailMtx_);
        tailStopped_ = true;
        tailCv_.notify_all();
    }
    for (const auto& r : dropped) {
        if (r.seekId > 0 && r.seekWaitFrame)
            seekComplete(int64_t(SeekError::Unloaded), r.seekId);
    }
}

void CRMReader::joinThreads()
{
    if (thread_.joinable())
        thread_.join();
    // After the decode thread, the only one that posts to it.
    if (tailThread_.joinable())
        tailThread_.join();
}

bool CRMReader::push(Request r, bool seek)
{
    vector<Request> dropped;
    {
        const scoped_lock lock(mtx_);
        if (!running_)
            return false;
        if (seek) {
            r.gen = gen_ + 1;
            gen_ = r.gen;
            chainStopped_ = false;
            dropped.assign(requests_.begin(), requests_.end());
            requests_.clear();
        } else if (r.gen != gen_) {
            return false;
        }
        requests_.push_back(r);
        cv_.notify_one();
    }
    // A seek superseded before it started completes as cancelled.
    for (const auto& d : dropped) {
        if (d.seekId > 0 && d.seekWaitFrame)
            seekComplete(int64_t(SeekError::Cancelled), d.seekId);
    }
    return true;
}

void CRMReader::continueChain(const Request& next, uint64_t resumesAtStart)
{
    // Read before taking the lock: MDK may hold its own while calling pause().
    const bool playing = state() == State::Running;
    bool stopped = false;
    {
        const scoped_lock lock(mtx_);
        if (!running_ || next.gen != gen_)
            return;
        if (!next.onlyIfPlaying || playing || resumes_ != resumesAtStart) {
            requests_.push_back(next);
            cv_.notify_one();
        } else {
            chainStopped_ = stopped = true;
            stoppedAt_ = next.index;
        }
    }
    if (stopped)
        pool_->trim(2); // paused: keep only what resuming needs
}

bool CRMReader::seekTo(int64_t msec, SeekFlag flag, int id)
{
    if (frames_ == 0)
        return false;
    uint64_t index = 0;
    if (test_flag(flag, SeekFlag::Frame)) { // FromNow|Frame: step by frames
        if (msec == 0) {
            seekComplete(frameMs(index_), id);
            return true;
        }
        index = (uint64_t)clamp<long long>((long long)index_ + msec, 0, (long long)frames_ - 1);
    } else if (test_flag(flag, SeekFlag::FromNow)) {
        index = indexAt(frameMs(index_) + msec);
    } else {
        index = indexAt(min<int64_t>(msec, frameMs(frames_))); // msec can be INT64_MAX
    }
    const bool waitFrame = !test_flag(flag, SeekFlag::IOCompleteCallback);
    bufferingPending_ = true;
    if (!push({index, id, waitFrame, 0, false}, true))
        return false;
    updateBufferingProgress(0);
    if (!waitFrame)
        seekComplete(frameMs(index), id);
    return true;
}

void CRMReader::decodeLoop()
{
    for (;;) {
        Request r;
        uint64_t resumes = 0;
        {
            unique_lock lock(mtx_);
            cv_.wait(lock, [this] { return !running_ || (ready_ && !requests_.empty()); });
            if (!running_)
                break;
            r = requests_.front();
            requests_.pop_front();
            resumes = resumes_;
        }
        optional<Request> next;
        RunInAutoreleasePool([&] { next = process(r); });
        if (next)
            continueChain(*next, resumes);
    }
    // The clip is only ever used on this thread, so it is closed here.
    if (pool_)
        clog << "CRM: at most " << pool_->highWater() << " output buffers were in use" << endl;
    loaded_ = false;
    if (reader_) {
        crmc_close(reader_);
        reader_ = nullptr;
    }
}

bool CRMReader::decode(uint64_t index, VideoFrame* frame, string* err)
{
    void* buf = pool_->acquire();
    if (!buf) {
        *err = "out of GPU memory for a " + std::to_string(out_.size) + " byte frame";
        return false;
    }
    auto recycle = pool_->recycler(buf);
    char ebuf[2048] = {};
    if (crmc_decode(reader_, index, buf, out_.size, &sync_gpu, gpu_.get(), info_.depth > 0, ebuf, sizeof(ebuf)) != 0) {
        recycle();
        *err = ebuf;
        return false;
    }
    const bool copy = copy_;
    if (!copy && !mdkPool_ && gpu_->lane() == CRMC_LANE_CUDA) // copy was switched off while playing
        cuda_exclusive([&] { mdkPool_ = NativeVideoBufferPool::create("CUDA"); });
    *frame = gpu_->frame(buf, out_, copy, &mdkPool_, std::move(recycle), err);
    if (!*frame) {
        *err = "frame " + std::to_string(index) + " could not be handed over: " + *err;
        return false;
    }
    frame->setColorSpace(cs_, false);
    if (info_.pixel_aspect != 1.0f)
        frame->pixelAspectRatio(info_.pixel_aspect);
    return true;
}

// An opaque black frame of the output size, in host memory. Invalid when it cannot be allocated.
VideoFrame CRMReader::blackFrame() const
{
    VideoFrame f((int)out_.width, (int)out_.height, PixelFormat::RGBA64LE);
    int strides[4] = {(int)out_.bytes_per_line};
    f.setBuffers(nullptr, strides);
    const auto b = f.buffer(0);
    if (!b || !b->data())
        return VideoFrame();
    const auto stride = (size_t)b->stride();
    for (uint32_t y = 0; y < out_.height; ++y) {
        const auto row = reinterpret_cast<uint16_t*>(b->data() + y * stride);
        for (uint32_t x = 0; x < out_.width; ++x) {
            row[x * 4] = row[x * 4 + 1] = row[x * 4 + 2] = 0;
            row[x * 4 + 3] = 0xFFFF;
        }
    }
    f.setColorSpace(cs_, false);
    if (info_.pixel_aspect != 1.0f)
        f.pixelAspectRatio(info_.pixel_aspect);
    return f;
}

/*
  Whether MDK may hold on to the frame at `t` seconds: with ContinueAtEnd (a
  looping player) a frame at or past the end of the playback range is where MDK
  restarts the loop, and for a stream with no audio decoded beside it, it keeps
  the frameAvailable call until the player stops instead.
 */
bool CRMReader::heldAtRangeEnd(double t)
{
    if (!test_flag(options() & Options::ContinueAtEnd))
        return false;
    const auto end = range().second;
    return end > 0 && end < frameMs(frames_) && llround(t * 1000.0) >= end;
}

/*
  The end of the stream. Without ContinueAtEnd MDK reads nothing after it, and
  the reader unloads. With it (a looping player) MDK can hold this call until
  the player stops, so it goes out on the tail thread and the decode thread
  keeps serving the seeks that loop back or that the user makes.
 */
void CRMReader::endOfStream(uint64_t gen)
{
    if (!test_flag(options() & Options::ContinueAtEnd)) {
        if (frameAvailable(VideoFrame().setTimestamp(TimestampEOS)))
            unload();
        return;
    }
    Tail t;
    t.eos = true;
    t.gen = gen;
    postTail(std::move(t));
}

// Hand `t` to the tail thread. False, and nothing is delivered, when no clip
// runs any more or the thread still holds an earlier item (MDK has not let go
// of it): the stream stays where it is until the next seek.
bool CRMReader::postTail(Tail&& t)
{
    {
        const scoped_lock lock(tailMtx_);
        if (tailStopped_ || tailBusy_)
            return false;
        tail_ = std::move(t);
        tailBusy_ = true;
        if (!tailThread_.joinable())
            tailThread_ = thread([this] { tailLoop(); });
    }
    tailCv_.notify_one();
    return true;
}

void CRMReader::tailLoop()
{
    unique_lock lock(tailMtx_);
    for (;;) {
        tailCv_.wait(lock, [this] { return tailStopped_ || tailBusy_; });
        if (tailStopped_)
            break;
        Tail t = std::move(tail_);
        tail_ = {};
        lock.unlock();
        // A seek since it was posted restarted the stream: nothing of it goes out.
        if (t.gen == gen_) {
            bool accepted = true;
            if (t.frame)
                accepted = frameAvailable(t.frame);
            t.frame = VideoFrame();
            if (accepted && t.gen == gen_) {
                if (t.eos)
                    frameAvailable(VideoFrame().setTimestamp(TimestampEOS));
                else if (t.next && test_flag(mediaStatus() & MediaStatus::Loaded) && state() != State::Stopped)
                    continueChain(*t.next, t.resumes);
            }
        }
        lock.lock();
        tailBusy_ = false;
    }
    tail_ = {};
    tailBusy_ = false;
}

// The decode thread's part of one read. Returns the read to queue next, if any.
optional<CRMReader::Request> CRMReader::process(const Request& r)
{
    applyScale();
    VideoFrame frame;
    string err;
    bool ok = decode(r.index, &frame, &err);
    const bool current = r.gen == gen_;
    const bool seekOwed = r.seekId > 0 && r.seekWaitFrame; // its seekComplete has not been called yet
    bool givingUp = false;
    if (!ok) {
        clog << "CRM: " << err << endl;
        if (!failed_)
            dispatchEvent({.error = -1, .category = "decoder.video", .detail = "CRM: " + err});
        if (!current) { // a newer seek takes over
            if (seekOwed)
                seekComplete(int64_t(SeekError::Cancelled), r.seekId);
            return nullopt;
        }
        // Skip a bad frame, but do not spin on a clip that fails throughout. A
        // seek stays open: it completes at the first frame that decodes.
        if (++errors_ < 3 && r.index + 1 < frames_)
            return Request{r.index + 1, r.seekId, r.seekWaitFrame, r.gen, false};
        givingUp = true;
        // MDK takes the end of a stream only after a frame, and whoever waits
        // for the seek's frame (a processing player) would wait for ever: a
        // black frame stands in for a seek target that cannot be decoded.
        if (r.seekId > 0) {
            frame = blackFrame();
            ok = bool(frame);
        }
        if (!ok) {
            index_ = r.index;
            if (seekOwed)
                seekComplete(int64_t(SeekError::EosReached), r.seekId);
        }
    } else {
        errors_ = 0;
        failed_ = false;
    }
    bool accepted = false;
    if (ok) {
        if (current)
            index_ = r.index; // before seekComplete, which may start the next seek
        if (seekOwed)
            seekComplete(current ? frameMs(r.index) : int64_t(SeekError::Cancelled), r.seekId);
        if (r.gen != gen_) // a newer seek is queued, maybe started by seekComplete above
            return nullopt;
        const bool last = r.index == frames_ - 1 && !givingUp;
        if (last)
            update(MediaStatus::Loaded | MediaStatus::End);
        const double t = frameTime(r.index);
        frame.setTimestamp(t);
        frame.setDuration(double(fpsDen_) / fpsNum_);
        if (r.seekId > 0)
            frameAvailable(VideoFrame(frame.format()).setTimestamp(t));
        // Before the frame: MDK holds that call while paused.
        if (bufferingPending_.exchange(false))
            updateBufferingProgress(100);
        const Request next{r.index + 1, 0, true, r.gen, true};
        if (!givingUp && heldAtRangeEnd(t)) {
            Tail tail;
            tail.frame = std::move(frame);
            tail.eos = last;
            if (!last)
                tail.next = next;
            tail.gen = r.gen;
            {
                const scoped_lock lock(mtx_);
                tail.resumes = resumes_;
            }
            postTail(std::move(tail));
            return nullopt;
        }
        // Blocks while paused; returns early on a seek.
        accepted = frameAvailable(frame);
        if (!givingUp) {
            if (!accepted || r.gen != gen_)
                return nullopt;
            if (last) {
                endOfStream(r.gen);
                return nullopt;
            }
            if (!test_flag(mediaStatus() & MediaStatus::Loaded) || state() == State::Stopped)
                return nullopt;
            return next;
        }
    }
    // Giving up: end the stream, so that whoever waits for frames (a processing
    // player, the renderer) is released. Once per run of failures, so that a
    // looping player does not loop on frames that keep failing.
    if (!failed_ && r.gen == gen_) {
        failed_ = true;
        clog << "CRM: " << errors_ << " frames in a row failed; ending the stream" << endl;
        update(MediaStatus::Loaded | MediaStatus::End);
        endOfStream(r.gen);
    }
    return nullopt;
}

MDK_NS_END

#if defined(_WIN32)
#define MDK_CRM_EXPORT extern "C" __declspec(dllexport)
#else
#define MDK_CRM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/*
  For the host application: run fn(user) under the Canon SDK's CUDA exclusion
  (Interface Spec 9.2.1: no CUDA call of the process may overlap the SDK's
  process-ref, develop and release calls). Wrap the application's own CUDA work
  in it, e.g. by looking the function up in the loaded plugin. Reentrant on the
  calling thread; safe to call when the SDK does not use CUDA (it then only
  serialises with the plugin's own calls).
 */
MDK_CRM_EXPORT void mdk_crm_cuda_exclusive(void (*fn)(void* user), void* user)
{
    crmc_cuda_exclusive(fn, user);
}

/* 1 when the Canon SDK develops on CUDA in this process (so the host's CUDA work needs the exclusion), else 0. */
MDK_CRM_EXPORT int mdk_crm_cuda_in_use(void)
{
    return MDK_NS::g_cuda_in_use ? 1 : 0;
}

// The plugin file must be named mdk-crm.
MDK_PLUGIN(crm) {
    using namespace MDK_NS;
    crmc_set_log(&core_log);
    FrameReader::registerOnce("CRM", [] { return new CRMReader(); }, {{"crm", "CRM"}});
    return MDK_ABI_VERSION;
}
