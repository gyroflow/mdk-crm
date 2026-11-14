/*
 * C API of the Rust core (core/): the CRM container, the Canon SDK and the
 * decode pipeline. The plugin (CRMReader.cpp) owns the GPU device, its output
 * buffers and everything MDK-facing.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CRMC_LANE_CUDA = 1,
    CRMC_LANE_OPENCL = 2,
    CRMC_LANE_METAL = 3,
};

/* The GPU the SDK develops on. Only the members of `lane` are read. */
typedef struct crmc_gpu {
    int32_t lane;
    int32_t cuda_device;   /* CUDA ordinal */
    void* cuda_stream;     /* CUstream of the device's primary context; the SDK develops on it */
    void* cl_platform;     /* cl_platform_id */
    void* cl_device;       /* cl_device_id */
    void* cl_context;      /* cl_context */
    void* cl_queue;        /* cl_command_queue; the SDK develops on it */
    void* mtl_device;      /* id<MTLDevice> */
    void* mtl_queue;       /* id<MTLCommandQueue>; the SDK develops on it */
} crmc_gpu;

/*
 * Where the core's diagnostics go: `level` is 1 (error), 2 (warning) or 3
 * (info). Null, the default: standard error. Called from any thread.
 */
typedef void (*crmc_log_fn)(int32_t level, const char* msg);
void crmc_set_log(crmc_log_fn fn);

/*
 * Find and load the Canon SDK library, without binding a GPU. `search` lists
 * directories (or library paths), tried in order: a directory is searched for
 * the library directly and in the SDK's own layout. Loaded once per process;
 * a failure is not remembered, so the SDK may be installed later.
 * Returns 0 on success, otherwise writes a message to `err`.
 */
int crmc_sdk_load(const char* const* search, size_t n_search, char* err, size_t err_len);

/*
 * Load and initialise the Canon SDK. It is a process singleton: the first call
 * loads the library and binds `gpu`, later calls with the same lane and handles
 * succeed at once, and a different one is refused.
 *
 * `search` as for crmc_sdk_load. `luts_dir`, when not null or empty, is the
 * directory that CONTAINS `CanonLUTs`; otherwise the SDK's own places are used.
 * Returns 0 on success. On failure writes a message to `err` and returns -1
 * when the SDK was not handed the GPU, or -2 when it was (and refused it): the
 * SDK may then keep using that GPU's stream or queue, which must stay alive.
 */
int crmc_sdk_init(const char* const* search, size_t n_search, const char* luts_dir, const crmc_gpu* gpu,
                  char* err, size_t err_len);

typedef struct crmc_reader crmc_reader;

typedef struct crmc_open_params {
    int32_t gamma;        /* CrmGamma value; -1: Rec.709 */
    int32_t color_space;  /* CrmColorSpace value; -1: BT.709 */
    int32_t depth;        /* frames decompressed ahead on the CPU (0..3); 0: one synchronous decode per frame */
    int32_t resolution;   /* CrmResolution class of the process refs; 0: the smallest that holds the clip */
} crmc_open_params;

typedef struct crmc_info {
    uint32_t width;        /* output at the full decode level */
    uint32_t height;
    uint64_t frame_count;
    uint32_t fps_num;      /* container frame rate */
    uint32_t fps_den;
    uint32_t bit_depth;    /* coded precision of the clip */
    float pixel_aspect;    /* anamorphic squeeze; 1 when none */
    int32_t gamma;         /* develop recipe in use */
    int32_t color_space;
    int32_t depth;         /* effective read-ahead; 0 when the synchronous path is used */
    int32_t resolution;    /* CrmResolution class of the process refs */
} crmc_info;

/* One output raster: RGBA 16-bit little-endian, `bytes_per_line` apart. */
typedef struct crmc_output {
    uint32_t width;
    uint32_t height;
    uint32_t bytes_per_line;
    uint32_t size;         /* bytes_per_line * height */
    int32_t level;         /* 1: full, 2: half, 4: quarter */
} crmc_output;

/* Open a clip on the initialised SDK. Returns null and writes `err` on failure. */
crmc_reader* crmc_open(const char* path_utf8, const crmc_open_params* params, char* err, size_t err_len);
void crmc_close(crmc_reader* r);
void crmc_get_info(const crmc_reader* r, crmc_info* out);

/* Clip metadata entry `i` (UTF-8, valid until crmc_close). Returns 0 past the end. */
int crmc_metadata(const crmc_reader* r, size_t i, const char** key, const char** value);

/* The output raster at `level` (1, 2 or 4), without applying it. Returns 0 on success. */
int crmc_output_for(const crmc_reader* r, int32_t level, crmc_output* out);
/* The output raster in use. */
void crmc_get_output(const crmc_reader* r, crmc_output* out);
/* Switch to `level`; waits for work in flight. Returns 0 on success. */
int crmc_set_level(crmc_reader* r, int32_t level, crmc_output* out, char* err, size_t err_len);

/* Returns 0 when the work finished, nonzero when the device reported an error. */
typedef int32_t (*crmc_sync_fn)(void* user);

/*
 * Decode `frame` into `dst`, a buffer of the lane's device (CUdeviceptr,
 * cl_mem or id<MTLBuffer>) of `dst_size` bytes, then call `sync(user)`, which
 * must wait until the device work on the stream or queue given to
 * crmc_sdk_init has finished; a nonzero result fails the decode. On CUDA both
 * run under the SDK's CUDA exclusion.
 * With `prefetch` the following frames keep decompressing in the background.
 * Returns 0 on success.
 */
int crmc_decode(crmc_reader* r, uint64_t frame, void* dst, uint32_t dst_size, crmc_sync_fn sync, void* user,
                int prefetch, char* err, size_t err_len);

/* Wait for and drop the read-ahead, e.g. before a long pause. */
void crmc_flush(crmc_reader* r);

/*
 * Run `fn(user)` under the process-wide exclusion the Canon SDK requires
 * between its CUDA calls and the application's own (Interface Spec 9.2.1).
 * Reentrant on the calling thread.
 */
void crmc_cuda_exclusive(void (*fn)(void* user), void* user);

#ifdef __cplusplus
}
#endif
