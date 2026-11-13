/*
 * Metal: the SDK develops into an MTLBuffer, which MDK samples through a
 * texture made over that same memory (zero-copy) when the device has unified
 * memory and the row pitch meets the linear-texture alignment; otherwise a
 * GPU blit copies it into a texture. copy=1 hands MDK host memory instead.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "Gpu.h"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>
#include <string>

#if !__has_feature(objc_arc)
#error "build with -fobjc-arc"
#endif

MDK_NS_BEGIN

namespace {

class MetalFrameBuffer final : public NativeVideoBuffer
{
public:
    MetalFrameBuffer(id<MTLBuffer> buf, id<MTLTexture> tex, const crmc_output& out, std::function<void()> recycle)
        : buf_(buf), tex_(tex), out_(out), recycle_(std::move(recycle)) {}
    ~MetalFrameBuffer() override {
        tex_ = nil;
        if (recycle_)
            recycle_();
    }
    void* map(Type type, MapParameter* mp) override {
        mp->width[0] = (int)out_.width;
        mp->height[0] = (int)out_.height;
        mp->format = PixelFormat::RGBA64LE;
        if (type == MetalTexture) {
            mt_.tex[0] = (__bridge void*)tex_;
            return &mt_;
        }
        if (type == HostMemory && buf_.storageMode == MTLStorageModeShared) {
            ma_.data[0] = buf_.contents;
            mp->stride[0] = (int)out_.bytes_per_line;
            return &ma_;
        }
        return nullptr;
    }
private:
    id<MTLBuffer> buf_;
    id<MTLTexture> tex_;
    crmc_output out_;
    std::function<void()> recycle_;
    MetalTextures mt_{};
    MemoryArray ma_{};
};

class MetalGpu final : public Gpu
{
public:
    bool create(const std::string& device, std::string* err) {
        if (device.empty()) {
            dev_ = MTLCreateSystemDefaultDevice();
        } else {
            int i = 0;
            for (id<MTLDevice> d in MTLCopyAllDevices()) {
                if (device == std::to_string(i++) || strstr(d.name.UTF8String, device.c_str())) {
                    dev_ = d;
                    break;
                }
            }
        }
        if (!dev_) {
            *err = device.empty() ? "no Metal device" : "no Metal device matches '" + device + "'";
            return false;
        }
        queue_ = [dev_ newCommandQueue];
        blitQueue_ = [dev_ newCommandQueue];
        if (!queue_ || !blitQueue_) {
            *err = "creating a Metal command queue failed";
            return false;
        }
        unified_ = dev_.hasUnifiedMemory;
        align_ = [dev_ minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA16Unorm];
        return true;
    }

    int lane() const override { return CRMC_LANE_METAL; }
    std::string deviceName() const override { return dev_.name.UTF8String; }
    crmc_gpu desc() const override {
        crmc_gpu g{};
        g.lane = CRMC_LANE_METAL;
        g.mtl_device = (__bridge void*)dev_;
        g.mtl_queue = (__bridge void*)queue_;
        return g;
    }
    void* alloc(uint32_t size) override {
        id<MTLBuffer> b = [dev_ newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!b) {
            std::clog << "CRM: allocating a " << size << " byte Metal buffer failed" << std::endl;
            return nullptr;
        }
        return (__bridge_retained void*)b;
    }
    void free(void* buf) override {
        if (buf) {
            id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf;
            b = nil;
        }
    }
    // A command buffer committed behind the SDK's work completes after it.
    void sync() override {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    VideoFrame frame(void* buf, const crmc_output& out, bool copy, NativeVideoBufferPool::Ptr*,
                     std::function<void()> recycle) override {
        @autoreleasepool {
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
            if (copy) {
                VideoFrame f((int)out.width, (int)out.height, PixelFormat::RGBA64LE);
                int strides[4] = {(int)out.bytes_per_line};
                f.setBuffers(nullptr, strides);
                const auto dst = f.buffer(0)->data();
                const auto stride = (size_t)f.buffer(0)->stride();
                const auto src = (const uint8_t*)b.contents;
                for (uint32_t y = 0; y < out.height; ++y)
                    memcpy(dst + y * stride, src + (size_t)y * out.bytes_per_line, (size_t)out.width * 8);
                recycle();
                return f;
            }
            MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Unorm
                                                                                            width:out.width
                                                                                           height:out.height
                                                                                        mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead;
            id<MTLTexture> tex = nil;
            if (unified_ && out.bytes_per_line % align_ == 0) {
                desc.storageMode = MTLStorageModeShared;
                tex = [b newTextureWithDescriptor:desc offset:0 bytesPerRow:out.bytes_per_line];
            }
            if (!tex) {
                desc.storageMode = MTLStorageModePrivate;
                tex = [dev_ newTextureWithDescriptor:desc];
                id<MTLCommandBuffer> cb = [blitQueue_ commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit copyFromBuffer:b
                         sourceOffset:0
                    sourceBytesPerRow:out.bytes_per_line
                  sourceBytesPerImage:out.size
                           sourceSize:MTLSizeMake(out.width, out.height, 1)
                            toTexture:tex
                     destinationSlice:0
                     destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            auto nb = std::make_shared<MetalFrameBuffer>(b, tex, out, std::move(recycle));
            return VideoFrame((int)out.width, (int)out.height, PixelFormat::RGBA64LE, nb);
        }
    }

private:
    id<MTLDevice> dev_ = nil;
    id<MTLCommandQueue> queue_ = nil;     // the SDK develops on it
    id<MTLCommandQueue> blitQueue_ = nil; // ours, for the copy path
    bool unified_ = false;
    NSUInteger align_ = 256;
};

} // namespace

std::shared_ptr<Gpu> CreateMetalGpu(const std::string& device, std::string* err)
{
    auto g = std::make_shared<MetalGpu>();
    if (!g->create(device, err))
        return nullptr;
    return g;
}

MDK_NS_END
