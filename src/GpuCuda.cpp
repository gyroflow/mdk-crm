/*
 * CUDA: the SDK develops on a stream of the device's primary context, and the
 * frames go to MDK as CUDA device memory (zero-copy), or are copied to host
 * memory with copy=1.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "Gpu.h"
#include "DynLib.h"
#include <cstring>
#include <iostream>
#include <string>

MDK_NS_BEGIN

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = unsigned long long;
using CUarray = struct CUarray_st*;

enum { CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2 };

struct CUDA_MEMCPY2D {
    size_t srcXInBytes, srcY;
    int srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    int dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
};

struct CudaApi {
    CUresult (*init)(unsigned) = nullptr;
    CUresult (*deviceGetCount)(int*) = nullptr;
    CUresult (*deviceGet)(CUdevice*, int) = nullptr;
    CUresult (*deviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*primaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*ctxPush)(CUcontext) = nullptr;
    CUresult (*ctxPop)(CUcontext*) = nullptr;
    CUresult (*streamCreate)(CUstream*, unsigned) = nullptr;
    CUresult (*streamSynchronize)(CUstream) = nullptr;
    CUresult (*memAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*memFree)(CUdeviceptr) = nullptr;
    CUresult (*memcpy2D)(const CUDA_MEMCPY2D*) = nullptr;
    CUresult (*getErrorString)(CUresult, const char**) = nullptr;

    bool load(std::string* err) {
        static DynLib lib;
        if (!lib.open({
#ifdef _WIN32
                "nvcuda.dll"
#else
                "libcuda.so.1", "libcuda.so"
#endif
            })) {
            *err = "no CUDA driver";
            return false;
        }
        bool ok = lib.sym(init, "cuInit") && lib.sym(deviceGetCount, "cuDeviceGetCount")
            && lib.sym(deviceGet, "cuDeviceGet") && lib.sym(deviceGetName, "cuDeviceGetName")
            && lib.sym(primaryCtxRetain, "cuDevicePrimaryCtxRetain") && lib.sym(ctxPush, "cuCtxPushCurrent_v2")
            && lib.sym(ctxPop, "cuCtxPopCurrent_v2") && lib.sym(streamCreate, "cuStreamCreate")
            && lib.sym(streamSynchronize, "cuStreamSynchronize") && lib.sym(memAlloc, "cuMemAlloc_v2")
            && lib.sym(memFree, "cuMemFree_v2") && lib.sym(memcpy2D, "cuMemcpy2D_v2");
        lib.sym(getErrorString, "cuGetErrorString");
        if (!ok)
            *err = "the CUDA driver lacks a required function";
        return ok;
    }
    std::string error(CUresult r) const {
        const char* s = nullptr;
        if (getErrorString)
            getErrorString(r, &s);
        return s ? std::string(s) : "CUresult " + std::to_string(r);
    }
};

// Run `f` under the Canon SDK's CUDA exclusion (Interface Spec 9.2.1).
template<typename F>
void exclusive(F&& f)
{
    using Fn = std::remove_reference_t<F>;
    crmc_cuda_exclusive([](void* user) {
        try {
            (*static_cast<Fn*>(user))();
        } catch (...) {
        }
    }, (void*)&f);
}

/*
  The SDK forbids its CUDA calls from overlapping any other CUDA call of the
  process. MDK maps a CUDA frame for rendering with CUDA calls of its own, on
  its render thread, so the mapping runs under the same exclusion.
 */
class ExclusiveBuffer final : public NativeVideoBuffer
{
public:
    explicit ExclusiveBuffer(NativeVideoBufferRef b) : b_(std::move(b)) {}
    void* map(Type type, MapParameter* mp) override {
        void* r = nullptr;
        exclusive([&] { r = b_->map(type, mp); });
        return r;
    }
    void unmap(Type type) override {
        exclusive([&] { b_->unmap(type); });
    }
    bool present() override { return b_->present(); }
    bool refCounted() const override { return b_->refCounted(); }
    Type type() const override { return b_->type(); }
private:
    NativeVideoBufferRef b_;
};

class CudaGpu final : public Gpu
{
public:
    bool create(const std::string& device, std::string* err) {
        if (!api_.load(err))
            return false;
        CUresult r = api_.init(0);
        if (r != 0) {
            *err = "cuInit: " + api_.error(r);
            return false;
        }
        int count = 0;
        api_.deviceGetCount(&count);
        if (count <= 0) {
            *err = "no CUDA device";
            return false;
        }
        ordinal_ = -1;
        for (int i = 0; i < count && ordinal_ < 0; ++i) {
            CUdevice d = 0;
            char name[256] = {};
            if (api_.deviceGet(&d, i) != 0)
                continue;
            api_.deviceGetName(name, sizeof(name), d);
            if (GpuMatches(device, i, name)) {
                ordinal_ = i;
                dev_ = d;
                name_ = name;
            }
        }
        if (ordinal_ < 0) {
            *err = "no CUDA device matches '" + device + "'";
            return false;
        }
        // The primary context: the one the SDK's CUDA runtime uses on this
        // device, and the one MDK shares.
        if ((r = api_.primaryCtxRetain(&ctx_, dev_)) != 0) {
            *err = "cuDevicePrimaryCtxRetain: " + api_.error(r);
            return false;
        }
        exclusive([&] {
            api_.ctxPush(ctx_);
            r = api_.streamCreate(&stream_, 0);
            CUcontext old = nullptr;
            api_.ctxPop(&old);
        });
        if (r != 0) {
            *err = "cuStreamCreate: " + api_.error(r);
            return false;
        }
        return true;
    }

    int lane() const override { return CRMC_LANE_CUDA; }
    std::string deviceName() const override { return name_; }
    crmc_gpu desc() const override {
        crmc_gpu g{};
        g.lane = CRMC_LANE_CUDA;
        g.cuda_device = ordinal_;
        g.cuda_stream = stream_;
        return g;
    }
    void* alloc(uint32_t size) override {
        CUdeviceptr p = 0;
        CUresult r = 0;
        exclusive([&] {
            api_.ctxPush(ctx_);
            r = api_.memAlloc(&p, size);
            CUcontext old = nullptr;
            api_.ctxPop(&old);
        });
        if (r != 0) {
            std::clog << "CRM: cuMemAlloc(" << size << "): " << api_.error(r) << std::endl;
            return nullptr;
        }
        return (void*)(uintptr_t)p;
    }
    void free(void* buf) override {
        if (!buf)
            return;
        exclusive([&] {
            api_.ctxPush(ctx_);
            api_.memFree((CUdeviceptr)(uintptr_t)buf);
            CUcontext old = nullptr;
            api_.ctxPop(&old);
        });
    }
    void sync() override {
        api_.ctxPush(ctx_);
        api_.streamSynchronize(stream_);
        CUcontext old = nullptr;
        api_.ctxPop(&old);
    }
    VideoFrame frame(void* buf, const crmc_output& out, bool copy, NativeVideoBufferPool::Ptr* pool,
                     std::function<void()> recycle) override {
        if (copy || !pool || !*pool) {
            VideoFrame f((int)out.width, (int)out.height, PixelFormat::RGBA64LE);
            int strides[4] = {(int)out.bytes_per_line};
            f.setBuffers(nullptr, strides);
            CUDA_MEMCPY2D c{};
            c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            c.srcDevice = (CUdeviceptr)(uintptr_t)buf;
            c.srcPitch = out.bytes_per_line;
            c.dstMemoryType = CU_MEMORYTYPE_HOST;
            c.dstHost = f.buffer(0)->data();
            c.dstPitch = (size_t)f.buffer(0)->stride();
            c.WidthInBytes = (size_t)out.width * 8;
            c.Height = out.height;
            CUresult r = 0;
            exclusive([&] {
                api_.ctxPush(ctx_);
                r = api_.memcpy2D(&c);
                CUcontext old = nullptr;
                api_.ctxPop(&old);
            });
            recycle();
            if (r != 0) {
                std::clog << "CRM: cuMemcpy2D: " << api_.error(r) << std::endl;
                return VideoFrame();
            }
            return f;
        }
        CUDAResource res{};
        res.ptr[0] = buf;
        res.width = (int)out.width;
        res.height = (int)out.height;
        res.stride[0] = (int)out.bytes_per_line;
        res.format = PixelFormat::RGBA64LE;
        res.context = ctx_;
        res.stream = stream_;
        res.unref = std::move(recycle);
        auto f = VideoFrame::from(pool, res);
        if (auto nb = f.nativeBuffer())
            f.setNativeBuffer(std::make_shared<ExclusiveBuffer>(nb));
        return f;
    }

private:
    CudaApi api_;
    int ordinal_ = -1;
    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    CUstream stream_ = nullptr;
    std::string name_;
};

} // namespace

std::shared_ptr<Gpu> CreateCudaGpu(const std::string& device, std::string* err)
{
    auto g = std::make_shared<CudaGpu>();
    if (!g->create(device, err))
        return nullptr;
    return g;
}

MDK_NS_END
