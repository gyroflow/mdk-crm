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

string option(const char* key)
{
    const auto& v = GetGlobalOption(key);
    if (const auto s = get_if<string>(&v); s && !s->empty())
        return *s;
    if (const auto e = getenv(key); e && *e)
        return e;
    return {};
}

// ─────────────────────────────────────────────────────────────────────
// The process-wide SDK binding: one GPU for every clip
// ─────────────────────────────────────────────────────────────────────

mutex g_engine_mtx;
shared_ptr<Gpu> g_gpu;
// GPUs the SDK refused. Kept: the SDK may hold on to their stream or queue.
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
            char buf[2048] = {};
            if (crmc_sdk_init(pdirs.data(), pdirs.size(), luts.c_str(), &desc, buf, sizeof(buf)) == 0) {
                clog << "CRM: the Canon SDK develops on " << GpuLaneName(l) << " (" << gpu->deviceName() << ")" << endl;
                g_gpu = gpu;
                return gpu;
            }
            g_refused.push_back(gpu);
            errs += string(errs.empty() ? "" : "; ") + GpuLaneName(l) + " " + gpu->deviceName() + ": " + buf;
        }
    }
    *err = errs;
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

string url_to_path(string url)
{
    if (url.rfind("file:", 0) != 0)
        return url;
    url.erase(0, 5);
    if (url.rfind("//", 0) == 0)
        url.erase(0, 2);
#ifdef _WIN32
    if (url.size() > 2 && url[0] == '/' && url[2] == ':')
        url.erase(0, 1);
#endif
    string out;
    for (size_t i = 0; i < url.size(); ++i) {
        if (url[i] == '%' && i + 2 < url.size() && isxdigit((unsigned char)url[i + 1]) && isxdigit((unsigned char)url[i + 2])) {
            out += (char)strtol(url.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else {
            out += url[i];
        }
    }
    return out;
}

} // namespace

class CRMReader final : public FrameReader
{
public:
    CRMReader() = default;
    ~CRMReader() override {
        stopThread();
        if (thread_.joinable())
            thread_.join();
    }
    const char* name() const override { return "CRM"; }
    void setTimeout(int64_t, TimeoutCallback) override {}
    bool load() override;
    bool unload() override;
    bool seekTo(int64_t msec, SeekFlag flag, int id) override;
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
    struct Request {
        uint64_t index = 0;
        int seekId = 0;
        bool seekWaitFrame = true;
    };

    void parseDecoderOptions();
    int chooseLevel() const;
    void applyScale();
    // Queue a read. A seek drops the reads queued before it. False once unloaded.
    bool push(const Request& r, bool seek);
    void stopThread();
    void decodeLoop();
    void process(const Request& r);
    bool decode(uint64_t index, VideoFrame* frame, string* err);

    double frameTime(uint64_t i) const { return double(i) * fpsDen_ / fpsNum_; }
    int64_t frameMs(uint64_t i) const { return llround(frameTime(i) * 1000.0); }
    uint64_t indexAt(int64_t ms) const {
        const auto i = llround(double(ms) * fpsNum_ / (1000.0 * fpsDen_));
        return (uint64_t)clamp<long long>(i, 0, (long long)frames_ - 1);
    }

    // options
    int lane_ = 0;
    string device_;
    bool copy_ = false;
    int depth_ = 3;
    int gamma_ = -1;
    int gamut_ = -1;
    mutable mutex scale_mtx_;
    int scaleLevel_ = 0;
    uint32_t scaleW_ = 0, scaleH_ = 0;
    atomic<bool> scaleChanged_ = false;

    // clip
    crmc_reader* reader_ = nullptr;
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
    atomic<int> seeking_ = 0;
    atomic<uint64_t> index_ = 0;
    int errors_ = 0;
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
    if (k == "gpu" || k == "lane") {
        const auto v = lower(value);
        lane_ = v == "cuda" ? CRMC_LANE_CUDA : v == "opencl" ? CRMC_LANE_OPENCL : v == "metal" ? CRMC_LANE_METAL : 0;
    } else if (k == "device") {
        device_ = value;
    } else if (k == "copy") {
        copy_ = atoi(value.c_str()) != 0;
    } else if (k == "depth") {
        depth_ = clamp(atoi(value.c_str()), 0, 3);
    } else if (k == "gamma") {
        gamma_ = gamma_value(value);
        if (gamma_ == kInvalid) {
            clog << "CRM: unknown gamma '" << value << "', using rec709" << endl;
            gamma_ = -1;
        }
    } else if (k == "gamut" || k == "colorspace") {
        gamut_ = gamut_value(value);
        if (gamut_ == kInvalid) {
            clog << "CRM: unknown gamut '" << value << "', using bt709" << endl;
            gamut_ = -1;
        }
    } else if (k == "scale" || k == "size") {
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

// On the decode thread, between frames.
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
    clog << "CRM: decoding at 1/" << out_.level << ": " << out_.width << "x" << out_.height << endl;
}

bool CRMReader::load()
{
    if (thread_.joinable())
        thread_.join();
    parseDecoderOptions();

    string err;
    gpu_ = engine(lane_, device_, &err);
    if (!gpu_) {
        clog << "CRM: no GPU to develop on: " << err << endl;
        dispatchEvent({.error = -1, .category = "decoder.video", .detail = "CRM: " + err});
        return false;
    }
    const crmc_open_params params{gamma_, gamut_, depth_};
    char ebuf[2048] = {};
    auto reader = crmc_open(url_to_path(url()).c_str(), &params, ebuf, sizeof(ebuf));
    if (!reader) {
        clog << "CRM: " << ebuf << endl;
        dispatchEvent({.error = -1, .category = "decoder.video", .detail = string("CRM: ") + ebuf});
        return false;
    }
    reader_ = reader;
    crmc_get_info(reader_, &info_);
    frames_ = info_.frame_count;
    fpsNum_ = info_.fps_num ? info_.fps_num : 25;
    fpsDen_ = info_.fps_den ? info_.fps_den : 1;
    cs_ = color_space(info_.gamma, info_.color_space);

    crmc_get_output(reader_, &out_);
    pool_ = make_shared<GpuBufferPool>(gpu_);
    scaleChanged_ = true;
    applyScale();
    pool_->setSize(out_.size);
    if (gpu_->lane() == CRMC_LANE_CUDA && !copy_)
        mdkPool_ = NativeVideoBufferPool::create("CUDA");

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
    clog << "CRM: " << info_.width << "x" << info_.height << " @ " << fpsNum_ << "/" << fpsDen_ << ", " << frames_
         << " frames, decoding at 1/" << out_.level << " (" << out_.width << "x" << out_.height << ") on "
         << GpuLaneName(gpu_->lane()) << (copy_ || gpu_->lane() == CRMC_LANE_OPENCL ? " to host memory" : " zero-copy")
         << ", read-ahead " << info_.depth << endl;

    {
        const scoped_lock lock(mtx_);
        requests_.clear();
        running_ = true;
    }
    seeking_ = 0;
    index_ = 0;
    errors_ = 0;
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
    if (seeking_ == 0)
        push({0, 0, true}, false);
    return true;
}

bool CRMReader::unload()
{
    const bool onDecodeThread = this_thread::get_id() == thread_.get_id();
    update(MediaStatus::Unloaded);
    if (!reader_) {
        update(State::Stopped);
        return false;
    }
    stopThread(); // the decode thread closes the clip when it leaves
    if (!onDecodeThread)
        frameAvailable(VideoFrame().setTimestamp(TimestampEOS)); // clears frames queued for rendering
    update(State::Stopped);
    return true;
}

void CRMReader::stopThread()
{
    vector<Request> dropped;
    {
        const scoped_lock lock(mtx_);
        running_ = false;
        dropped.assign(requests_.begin(), requests_.end());
        requests_.clear();
        cv_.notify_all();
    }
    for (const auto& r : dropped) {
        if (r.seekId > 0 && r.seekWaitFrame) {
            seeking_--;
            seekComplete(frameMs(r.index), r.seekId);
        }
    }
}

bool CRMReader::push(const Request& r, bool seek)
{
    vector<Request> dropped;
    {
        const scoped_lock lock(mtx_);
        if (!running_)
            return false;
        if (seek) {
            dropped.assign(requests_.begin(), requests_.end());
            requests_.clear();
        }
        requests_.push_back(r);
        cv_.notify_one();
    }
    // A seek superseded before it started still completes, at its own target.
    for (const auto& d : dropped) {
        if (d.seekId > 0 && d.seekWaitFrame) {
            seeking_--;
            seekComplete(frameMs(d.index), d.seekId);
        }
    }
    return true;
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
    seeking_++;
    if (!push({index, id, waitFrame}, true)) {
        seeking_--;
        return false;
    }
    updateBufferingProgress(0);
    if (!waitFrame) {
        seeking_--;
        seekComplete(frameMs(index), id);
    }
    return true;
}

void CRMReader::decodeLoop()
{
    for (;;) {
        Request r;
        {
            unique_lock lock(mtx_);
            cv_.wait(lock, [this] { return !running_ || !requests_.empty(); });
            if (!running_)
                break;
            r = requests_.front();
            requests_.pop_front();
        }
        process(r);
    }
    // The clip is only ever used on this thread, so it is closed here.
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
    const auto sync = [](void* gpu) { static_cast<Gpu*>(gpu)->sync(); };
    if (crmc_decode(reader_, index, buf, out_.size, sync, gpu_.get(), depth_ > 0, ebuf, sizeof(ebuf)) != 0) {
        recycle();
        *err = ebuf;
        return false;
    }
    *frame = gpu_->frame(buf, out_, copy_, &mdkPool_, std::move(recycle));
    if (!*frame) {
        *err = "frame " + std::to_string(index) + " could not be handed over";
        return false;
    }
    frame->setColorSpace(cs_, false);
    if (info_.pixel_aspect != 1.0f)
        frame->pixelAspectRatio(info_.pixel_aspect);
    return true;
}

void CRMReader::process(const Request& r)
{
    applyScale();
    VideoFrame frame;
    string err;
    const bool ok = decode(r.index, &frame, &err);
    index_ = r.index; // before seekComplete, which may start the next seek
    if (r.seekId > 0 && r.seekWaitFrame) {
        seeking_--;
        seekComplete(frameMs(r.index), r.seekId);
    }
    const bool reading = test_flag(mediaStatus() & MediaStatus::Loaded) && state() == State::Running;
    if (!ok) {
        clog << "CRM: " << err << endl;
        dispatchEvent({.error = -1, .category = "decoder.video", .detail = "CRM: " + err});
        // Skip a bad frame, but do not spin on a clip that fails throughout.
        if (++errors_ < 3 && seeking_ == 0 && reading && r.index + 1 < frames_)
            push({r.index + 1, 0, true}, false);
        return;
    }
    errors_ = 0;
    if (seeking_ > 0) // a newer seek is queued
        return;
    if (r.index == frames_ - 1)
        update(MediaStatus::Loaded | MediaStatus::End);
    const double t = frameTime(r.index);
    frame.setTimestamp(t);
    frame.setDuration(double(fpsDen_) / fpsNum_);
    if (r.seekId > 0)
        frameAvailable(VideoFrame(frame.format()).setTimestamp(t));
    // Blocks while paused; returns early on a seek.
    bool accepted = frameAvailable(frame);
    if (r.index == frames_ - 1 && seeking_ == 0 && accepted) {
        accepted = frameAvailable(VideoFrame().setTimestamp(TimestampEOS));
        if (accepted && !test_flag(options() & Options::ContinueAtEnd))
            unload();
        return;
    }
    if (accepted && seeking_ == 0 && state() == State::Running && test_flag(mediaStatus() & MediaStatus::Loaded))
        push({r.index + 1, 0, true}, false);
}

MDK_NS_END

// The plugin file must be named mdk-crm.
MDK_PLUGIN(crm) {
    using namespace MDK_NS;
    FrameReader::registerOnce("CRM", [] { return new CRMReader(); }, {{"crm", "CRM"}});
    return MDK_ABI_VERSION;
}
