/*
 * The GPU the Canon SDK develops on, and the frames handed to MDK from it.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#pragma once
#include "mdk/VideoBuffer.h"
#include "mdk/VideoFrame.h"
#include "crm_core.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

MDK_NS_BEGIN

class Gpu
{
public:
    virtual ~Gpu() = default;
    virtual int lane() const = 0;
    virtual std::string deviceName() const = 0;
    // What crmc_sdk_init binds the SDK to.
    virtual crmc_gpu desc() const = 0;
    // A device buffer of the lane: CUdeviceptr, cl_mem or id<MTLBuffer>. Null on failure.
    virtual void* alloc(uint32_t size) = 0;
    virtual void free(void* buf) = 0;
    // Wait for the work on the stream or queue the SDK develops on. Called by
    // crmc_decode, under the CUDA exclusion on CUDA.
    virtual void sync() = 0;
    /*
      Wrap a developed buffer. `recycle` must run once MDK no longer needs the
      buffer. With `copy` the pixels are copied to host memory and the buffer is
      recycled before returning. `pool` is the reader's MDK buffer pool, for
      lanes that MDK maps itself.
     */
    virtual VideoFrame frame(void* buf, const crmc_output& out, bool copy, NativeVideoBufferPool::Ptr* pool,
                             std::function<void()> recycle) = 0;
};

// Whether device `index` named `name` is the one `device` asks for: all of them
// when empty, the index when a number, else a part of the name.
inline bool GpuMatches(const std::string& device, int index, const char* name)
{
    if (device.empty())
        return true;
    if (device.find_first_not_of("0123456789") == std::string::npos)
        return std::to_string(index) == device;
    return name && std::string(name).find(device) != std::string::npos;
}

// The lanes this platform has, in the order `gpu=auto` tries them.
std::vector<int> GpuLanes();
const char* GpuLaneName(int lane);
// Create the GPU of `lane`. `device` is an index or a part of the device name; empty: the first one.
std::shared_ptr<Gpu> CreateGpu(int lane, const std::string& device, std::string* err);

/*
  Output buffers of one size, recycled between frames. Frames keep the pool
  alive through their recycle callback, so it may outlive the reader.
 */
class GpuBufferPool : public std::enable_shared_from_this<GpuBufferPool>
{
public:
    explicit GpuBufferPool(std::shared_ptr<Gpu> gpu) : gpu_(std::move(gpu)) {}
    ~GpuBufferPool() {
        for (auto b : free_)
            gpu_->free(b);
    }
    // Sizes change with the decode level; buffers of another size are freed.
    void setSize(uint32_t size) {
        std::vector<void*> drop;
        {
            const std::scoped_lock lock(mtx_);
            if (size == size_)
                return;
            size_ = size;
            drop.swap(free_);
        }
        for (auto b : drop)
            gpu_->free(b);
    }
    uint32_t size() const { return size_; }
    void* acquire() {
        {
            const std::scoped_lock lock(mtx_);
            if (!free_.empty()) {
                auto b = free_.back();
                free_.pop_back();
                return b;
            }
        }
        return gpu_->alloc(size_);
    }
    // A recycle callback for `buf`, acquired at the current size.
    std::function<void()> recycler(void* buf) {
        return [pool = shared_from_this(), buf, size = size_] { pool->release(buf, size); };
    }
    void release(void* buf, uint32_t size) {
        {
            const std::scoped_lock lock(mtx_);
            if (size == size_) {
                free_.push_back(buf);
                return;
            }
        }
        gpu_->free(buf);
    }
private:
    std::shared_ptr<Gpu> gpu_;
    std::mutex mtx_;
    std::vector<void*> free_;
    uint32_t size_ = 0;
};

MDK_NS_END
