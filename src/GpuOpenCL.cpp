/*
 * OpenCL, for GPUs without CUDA: the SDK develops on the GPU, and the frame is
 * read back to host memory, since MDK takes no OpenCL memory.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "Gpu.h"
#include "DynLib.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

MDK_NS_BEGIN

namespace {

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_bitfield = cl_ulong;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_event = struct _cl_event*;
using cl_context_properties = intptr_t;

constexpr cl_bitfield CL_DEVICE_TYPE_GPU = 1 << 2;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_bitfield CL_MEM_READ_WRITE = 1 << 0;
constexpr cl_uint CL_TRUE = 1;

#ifdef _WIN32
#define CL_API __stdcall
#else
#define CL_API
#endif

struct ClApi {
    cl_int (CL_API *getPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int (CL_API *getDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_int (CL_API *getDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_context (CL_API *createContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
    cl_command_queue (CL_API *createCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*) = nullptr;
    cl_mem (CL_API *createBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*) = nullptr;
    cl_int (CL_API *releaseMemObject)(cl_mem) = nullptr;
    cl_int (CL_API *finish)(cl_command_queue) = nullptr;
    cl_int (CL_API *enqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;

    bool load(std::string* err) {
        static DynLib lib;
        if (!lib.open({
#ifdef _WIN32
                "OpenCL.dll"
#else
                "libOpenCL.so.1", "libOpenCL.so"
#endif
            })) {
            *err = "no OpenCL runtime";
            return false;
        }
        const bool ok = lib.sym(getPlatformIDs, "clGetPlatformIDs") && lib.sym(getDeviceIDs, "clGetDeviceIDs")
            && lib.sym(getDeviceInfo, "clGetDeviceInfo") && lib.sym(createContext, "clCreateContext")
            && lib.sym(createCommandQueue, "clCreateCommandQueue") && lib.sym(createBuffer, "clCreateBuffer")
            && lib.sym(releaseMemObject, "clReleaseMemObject") && lib.sym(finish, "clFinish")
            && lib.sym(enqueueReadBuffer, "clEnqueueReadBuffer");
        if (!ok)
            *err = "the OpenCL runtime lacks a required function";
        return ok;
    }
};

class OpenCLGpu final : public Gpu
{
public:
    bool create(const std::string& device, std::string* err) {
        if (!api_.load(err))
            return false;
        cl_uint np = 0;
        if (api_.getPlatformIDs(0, nullptr, &np) != 0 || np == 0) {
            *err = "no OpenCL platform";
            return false;
        }
        std::vector<cl_platform_id> platforms(np);
        api_.getPlatformIDs(np, platforms.data(), nullptr);
        int index = 0;
        for (auto p : platforms) {
            cl_uint nd = 0;
            if (api_.getDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != 0 || nd == 0)
                continue;
            std::vector<cl_device_id> devices(nd);
            api_.getDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devices.data(), nullptr);
            for (auto d : devices) {
                char name[256] = {};
                api_.getDeviceInfo(d, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
                if (GpuMatches(device, index, name)) {
                    platform_ = p;
                    device_ = d;
                    name_ = name;
                    break;
                }
                ++index;
            }
            if (device_)
                break;
        }
        if (!device_) {
            *err = device.empty() ? "no OpenCL GPU" : "no OpenCL GPU matches '" + device + "'";
            return false;
        }
        cl_int r = 0;
        const cl_context_properties props[] = {0x1084 /*CL_CONTEXT_PLATFORM*/, (cl_context_properties)platform_, 0};
        context_ = api_.createContext(props, 1, &device_, nullptr, nullptr, &r);
        if (!context_) {
            *err = "clCreateContext: " + std::to_string(r);
            return false;
        }
        queue_ = api_.createCommandQueue(context_, device_, 0, &r);
        if (!queue_) {
            *err = "clCreateCommandQueue: " + std::to_string(r);
            return false;
        }
        return true;
    }

    int lane() const override { return CRMC_LANE_OPENCL; }
    std::string deviceName() const override { return name_; }
    crmc_gpu desc() const override {
        crmc_gpu g{};
        g.lane = CRMC_LANE_OPENCL;
        g.cl_platform = platform_;
        g.cl_device = device_;
        g.cl_context = context_;
        g.cl_queue = queue_;
        return g;
    }
    void* alloc(uint32_t size) override {
        cl_int r = 0;
        auto m = api_.createBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &r);
        if (!m)
            std::clog << "CRM: clCreateBuffer(" << size << "): " << r << std::endl;
        return m;
    }
    void free(void* buf) override {
        if (buf)
            api_.releaseMemObject((cl_mem)buf);
    }
    int sync() override {
        const cl_int r = api_.finish(queue_);
        if (r != 0)
            std::clog << "CRM: clFinish: " << r << std::endl;
        return r;
    }
    VideoFrame frame(void* buf, const crmc_output& out, bool, NativeVideoBufferPool::Ptr*,
                     std::function<void()> recycle, std::string* err) override {
        VideoFrame f((int)out.width, (int)out.height, PixelFormat::RGBA64LE);
        int strides[4] = {(int)out.bytes_per_line};
        f.setBuffers(nullptr, strides);
        const auto dst = f.buffer(0)->data();
        const auto stride = (size_t)f.buffer(0)->stride();
        cl_int r = 0;
        if (stride == out.bytes_per_line) {
            r = api_.enqueueReadBuffer(queue_, (cl_mem)buf, CL_TRUE, 0, out.size, dst, 0, nullptr, nullptr);
        } else {
            std::vector<uint8_t> tmp(out.size);
            r = api_.enqueueReadBuffer(queue_, (cl_mem)buf, CL_TRUE, 0, out.size, tmp.data(), 0, nullptr, nullptr);
            for (uint32_t y = 0; r == 0 && y < out.height; ++y)
                memcpy(dst + y * stride, tmp.data() + (size_t)y * out.bytes_per_line, (size_t)out.width * 8);
        }
        recycle();
        if (r != 0) {
            *err = "reading the frame back from the GPU: clEnqueueReadBuffer " + std::to_string(r);
            return VideoFrame();
        }
        return f;
    }

private:
    ClApi api_;
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::string name_;
};

} // namespace

std::shared_ptr<Gpu> CreateOpenCLGpu(const std::string& device, std::string* err)
{
    auto g = std::make_shared<OpenCLGpu>();
    if (!g->create(device, err))
        return nullptr;
    return g;
}

MDK_NS_END
