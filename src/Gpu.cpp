/*
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "Gpu.h"

MDK_NS_BEGIN

#if (__APPLE__ + 0)
std::shared_ptr<Gpu> CreateMetalGpu(const std::string& device, std::string* err);
#else
std::shared_ptr<Gpu> CreateCudaGpu(const std::string& device, std::string* err);
std::shared_ptr<Gpu> CreateOpenCLGpu(const std::string& device, std::string* err);
#endif

std::vector<int> GpuLanes()
{
#if (__APPLE__ + 0)
    return {CRMC_LANE_METAL};
#else
    return {CRMC_LANE_CUDA, CRMC_LANE_OPENCL};
#endif
}

const char* GpuLaneName(int lane)
{
    switch (lane) {
    case CRMC_LANE_CUDA: return "CUDA";
    case CRMC_LANE_OPENCL: return "OpenCL";
    case CRMC_LANE_METAL: return "Metal";
    default: return "?";
    }
}

std::shared_ptr<Gpu> CreateGpu(int lane, const std::string& device, std::string* err)
{
    switch (lane) {
#if (__APPLE__ + 0)
    case CRMC_LANE_METAL: return CreateMetalGpu(device, err);
#else
    case CRMC_LANE_CUDA: return CreateCudaGpu(device, err);
    case CRMC_LANE_OPENCL: return CreateOpenCLGpu(device, err);
#endif
    default:
        *err = std::string(GpuLaneName(lane)) + " is not available on this platform";
        return nullptr;
    }
}

MDK_NS_END
