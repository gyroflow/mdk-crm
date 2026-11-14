/*
 * The GPU the Canon SDK develops on, and the frames handed to MDK from it.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#pragma once
#include "mdk/VideoBuffer.h"
#include "mdk/VideoFrame.h"
#include "crm_core.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
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
    // crmc_decode, under the CUDA exclusion on CUDA. 0 when it finished, else
    // the device's error.
    virtual int sync() = 0;
    /*
      Wrap a developed buffer. `recycle` must run once MDK no longer needs the
      buffer. With `copy` the pixels are copied to host memory and the buffer is
      recycled before returning. `pool` is the reader's MDK buffer pool, for
      lanes that MDK maps itself. An invalid frame on failure (`err` says why).
     */
    virtual VideoFrame frame(void* buf, const crmc_output& out, bool copy, NativeVideoBufferPool::Ptr* pool,
                             std::function<void()> recycle, std::string* err) = 0;
    // How many free buffers to keep back before reusing one: a frame MDK has
    // released may still be read by a command buffer of its renderer that
    // nothing orders against the SDK's next develop into it (Metal samples the
    // buffer itself). CUDA and OpenCL hand MDK copies or map synchronously.
    virtual size_t reuseDelay() const { return 0; }
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
// Run `f` inside an autorelease pool on Apple platforms (objects the SDK
// autoreleases on a plain thread would otherwise live until the thread ends).
void RunInAutoreleasePool(const std::function<void()>& f);

/*
  Output buffers of one size, recycled between frames. Frames keep the pool
  alive through their recycle callback, so it may outlive the reader.

  Buffers are reused in the order they come back. At most `cap` exist at once
  (handed out plus free): `acquire` waits for one to come back rather than
  allocating past it, since each holds a whole frame of GPU memory (283 MB at
  8K). A wait longer than `wait` allocates anyway and raises the cap to what is
  in use, so a consumer that holds more frames than expected stalls once
  instead of locking up or waiting on every frame.
 */
class GpuBufferPool : public std::enable_shared_from_this<GpuBufferPool>
{
public:
    GpuBufferPool(std::shared_ptr<Gpu> gpu, size_t cap) : gpu_(std::move(gpu)), cap_(cap), delay_(gpu_->reuseDelay()) {}
    ~GpuBufferPool() {
        for (auto b : free_)
            gpu_->free(b);
    }
    // Sizes change with the decode level. Each change starts a new epoch: the
    // free buffers of the old one are freed now, and those still handed out
    // when they come back (they are not counted any more), also when the size
    // has meanwhile changed back to theirs.
    void setSize(uint32_t size) {
        std::deque<void*> drop;
        {
            const std::scoped_lock lock(mtx_);
            if (size == size_)
                return;
            size_ = size;
            ++epoch_;
            drop.swap(free_);
            count_ = 0;
        }
        cv_.notify_all();
        for (auto b : drop)
            gpu_->free(b);
    }
    uint32_t size() const {
        const std::scoped_lock lock(mtx_);
        return size_;
    }
    // A buffer of the current size, or null when the device is out of memory.
    void* acquire() {
        uint32_t size = 0;
        {
            std::unique_lock lock(mtx_);
            const auto deadline = std::chrono::steady_clock::now() + wait_;
            while (free_.size() <= delay_ && count_ >= cap_) {
                if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                    std::clog << "CRM: all " << count_ << " output buffers are still held after " << wait_.count()
                              << " ms; allowing " << count_ + 1 << std::endl;
                    cap_ = count_ + 1;
                    break;
                }
            }
            if (free_.size() > delay_) {
                auto b = free_.front();
                free_.pop_front();
                return b;
            }
            ++count_;
            high_ = std::max(high_, count_);
            size = size_;
        }
        auto b = gpu_->alloc(size);
        if (!b) {
            const std::scoped_lock lock(mtx_);
            --count_;
        }
        return b;
    }
    // A recycle callback for `buf`, acquired in the current epoch. Call it once.
    std::function<void()> recycler(void* buf) {
        uint64_t epoch = 0;
        {
            const std::scoped_lock lock(mtx_);
            epoch = epoch_;
        }
        return [pool = shared_from_this(), buf, epoch] { pool->release(buf, epoch); };
    }
    // Free all but `keep` idle buffers, e.g. while paused.
    void trim(size_t keep) {
        std::vector<void*> drop;
        {
            const std::scoped_lock lock(mtx_);
            keep = std::max(keep, delay_);
            while (free_.size() > keep) {
                drop.push_back(free_.front());
                free_.pop_front();
                --count_;
            }
        }
        for (auto b : drop)
            gpu_->free(b);
    }
    // The most buffers that existed at once.
    size_t highWater() const {
        const std::scoped_lock lock(mtx_);
        return high_;
    }
private:
    void release(void* buf, uint64_t epoch) {
        {
            const std::scoped_lock lock(mtx_);
            if (epoch == epoch_) {
                free_.push_back(buf);
                cv_.notify_one();
                return;
            }
        }
        gpu_->free(buf); // of an earlier epoch: counted out when the size changed
    }

    std::shared_ptr<Gpu> gpu_;
    size_t cap_;
    const size_t delay_;
    const std::chrono::milliseconds wait_{1000};
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<void*> free_;
    size_t count_ = 0; // buffers of the current epoch: handed out plus free
    size_t high_ = 0;
    uint32_t size_ = 0;
    uint64_t epoch_ = 0;
};

MDK_NS_END
