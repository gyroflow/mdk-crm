// SPDX-License-Identifier: MIT OR Apache-2.0

//! Loading the Canon SDK and binding it to one GPU, once per process.
//!
//! `CrmInitializeSDK*` and `CrmTerminateSDK` are process-global, and a
//! terminate leaves behind the CUDA context its init made, so the SDK is
//! initialised once and kept for the life of the process.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use canon_raw::CrmSdkLibrary;

pub const LANE_CUDA: i32 = 1;
pub const LANE_OPENCL: i32 = 2;
pub const LANE_METAL: i32 = 3;

/// `crmc_gpu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Gpu {
    pub lane: i32,
    pub cuda_device: i32,
    pub cuda_stream: *mut c_void,
    pub cl_platform: *mut c_void,
    pub cl_device: *mut c_void,
    pub cl_context: *mut c_void,
    pub cl_queue: *mut c_void,
    pub mtl_device: *mut c_void,
    pub mtl_queue: *mut c_void,
}

/// What one init bound: the lane plus the handles that identify its device and
/// stream. Two callers share the SDK only when theirs match, because the SDK
/// develops on the stream (or queue) it was initialised with, and that is the
/// one each caller synchronises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuKey {
    pub lane: i32,
    handles: [usize; 2],
}

impl Gpu {
    fn key(&self) -> GpuKey {
        let handles = match self.lane {
            LANE_CUDA => [self.cuda_device as usize, self.cuda_stream as usize],
            LANE_OPENCL => [self.cl_context as usize, self.cl_queue as usize],
            LANE_METAL => [self.mtl_device as usize, self.mtl_queue as usize],
            _ => [0, 0],
        };
        GpuKey { lane: self.lane, handles }
    }
}

pub fn lane_name(lane: i32) -> &'static str {
    match lane {
        LANE_CUDA => "CUDA",
        LANE_OPENCL => "OpenCL",
        LANE_METAL => "Metal",
        _ => "unknown",
    }
}

struct State {
    lib: Option<(Arc<CrmSdkLibrary>, PathBuf)>,
    key: Option<GpuKey>,
}

static STATE: Mutex<State> = Mutex::new(State { lib: None, key: None });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The initialised SDK and the GPU it is bound to.
pub fn current() -> Option<(Arc<CrmSdkLibrary>, GpuKey)> {
    let s = state();
    Some((s.lib.as_ref()?.0.clone(), s.key?))
}

pub fn init(search: &[PathBuf], luts_dir: Option<&Path>, gpu: &Gpu) -> Result<(), String> {
    let key = gpu.key();
    if !matches!(key.lane, LANE_CUDA | LANE_OPENCL | LANE_METAL) {
        return Err(format!("unknown GPU lane {}", gpu.lane));
    }
    let mut s = state();
    if let Some(bound) = s.key {
        if bound == key {
            return Ok(());
        }
        return Err(format!(
            "the Canon SDK is a process singleton and is already initialised for {} on another device or stream",
            lane_name(bound.lane)
        ));
    }
    if s.lib.is_none() {
        let (lib, path) = load(search)?;
        s.lib = Some((Arc::new(lib), path));
    }
    let (lib, path) = s.lib.clone().expect("loaded above");
    // A no-op before the first init, and a clean slate after a failed one.
    // SAFETY: nothing has been created on the SDK: no regime is bound yet.
    let _ = unsafe { lib.terminate_sdk() };
    init_lane(&lib, gpu)?;
    if let Err(e) = check_supported(&lib, gpu.lane) {
        // SAFETY: nothing was created on the regime that was just bound.
        let _ = unsafe { lib.terminate_sdk() };
        return Err(e);
    }
    apply_luts(&lib, &path, luts_dir);
    s.key = Some(key);
    Ok(())
}

fn init_lane(lib: &CrmSdkLibrary, gpu: &Gpu) -> Result<(), String> {
    let lane = lane_name(gpu.lane);
    let r = match gpu.lane {
        #[cfg(not(target_vendor = "apple"))]
        LANE_CUDA => {
            if gpu.cuda_stream.is_null() {
                return Err("CUDA lane without a stream".into());
            }
            let mut init = canon_raw::CrmCudaInit::new(gpu.cuda_device.max(0) as u32, gpu.cuda_stream);
            // SAFETY: the stream belongs to the primary context of `cuda_device`
            // and lives for the rest of the process (the plugin never destroys it).
            unsafe { lib.initialize_with_cuda(init.as_raw_mut()) }
        }
        LANE_OPENCL => {
            if gpu.cl_context.is_null() || gpu.cl_queue.is_null() || gpu.cl_device.is_null() {
                return Err("OpenCL lane without a device, context or queue".into());
            }
            let mut init = canon_raw::CrmOpenClInit::new(gpu.cl_platform, gpu.cl_device, gpu.cl_context, gpu.cl_queue);
            // SAFETY: live OpenCL objects that the plugin keeps for the rest of the process.
            unsafe { lib.initialize_with_opencl(init.as_raw_mut()) }
        }
        #[cfg(target_vendor = "apple")]
        LANE_METAL => {
            if gpu.mtl_device.is_null() || gpu.mtl_queue.is_null() {
                return Err("Metal lane without a device or queue".into());
            }
            // SAFETY: a live id<MTLDevice> and an id<MTLCommandQueue> on it; both are retained.
            let mut init = unsafe { canon_raw::CrmMetalInit::new(gpu.mtl_device, gpu.mtl_queue) };
            unsafe { lib.initialize_with_metal(init.as_raw_mut()) }
        }
        _ => return Err(format!("the {lane} lane is not available on this platform")),
    };
    r.map_err(|e| {
        let detail = lib.get_last_error_string().unwrap_or_default();
        format!("Canon SDK initialisation on {lane} failed: {e} {detail}").trim_end().to_string()
    })
}

/// The SDK binds any device it is handed, then refuses to create a process
/// ref on one it does not support (`DeviceNotSupported`), so every clip would
/// fail. Its own GPU list says which it supports; refusing the lane here lets
/// the caller try another.
fn check_supported(lib: &CrmSdkLibrary, lane: i32) -> Result<(), String> {
    let device = match lane {
        LANE_CUDA => canon_raw::Device::Cuda,
        LANE_OPENCL => canon_raw::Device::OpenCl,
        _ => canon_raw::Device::Metal,
    };
    let Ok(list) = lib.get_gpu_info_list_with_device(device) else { return Ok(()) };
    let names: Vec<String> = list
        .iter()
        .map(|g| format!("{}{}", g.device_name(), if g.is_supported() { "" } else { " (unsupported)" }))
        .collect();
    eprintln!("CRM: the Canon SDK sees on {}: {}", lane_name(lane), names.join(", "));
    if list.is_empty() || list.iter().any(|g| g.is_supported()) {
        return Ok(());
    }
    Err(format!("the Canon SDK does not support {} on {}", names.join(", "), lane_name(lane)))
}

// ─────────────────────────────────────────────────────────────────────
// Locating and loading the library
// ─────────────────────────────────────────────────────────────────────

/// The library's file name. On Windows it has to be exactly this: the SDK
/// finds its resources by stripping `CrmSdk.dll` off its own module path,
/// case-sensitively, and aborts the process when that fails.
#[cfg(target_os = "windows")]
const LIB_NAME: &str = "CrmSdk.dll";
#[cfg(target_vendor = "apple")]
const LIB_NAME: &str = "CrmSdk";
#[cfg(not(any(target_os = "windows", target_vendor = "apple")))]
const LIB_NAME: &str = "libCrmSdk.so";

/// Where the library sits below a directory: beside the application, or in
/// the SDK's own layout (`CRMSDK210/CRMSDK/...`).
fn library_candidates(dir: &Path) -> Vec<PathBuf> {
    #[cfg(target_os = "windows")]
    let rel: &[&str] = &["CrmSdk.dll", "Win_Lib/x64/CrmSdk.dll", "CRMSDK/Win_Lib/x64/CrmSdk.dll"];
    #[cfg(target_vendor = "apple")]
    let rel: &[&str] = &[
        "CrmSdk.framework/CrmSdk",
        "../Frameworks/CrmSdk.framework/CrmSdk",
        "Mac_Lib/x86_64/CrmSdk.framework/CrmSdk",
        "CRMSDK/Mac_Lib/x86_64/CrmSdk.framework/CrmSdk",
    ];
    #[cfg(not(any(target_os = "windows", target_vendor = "apple")))]
    let rel: &[&str] = &["libCrmSdk.so", "Linux_Lib/libCrmSdk.so", "CRMSDK/Linux_Lib/libCrmSdk.so"];
    rel.iter().map(|r| dir.join(r)).collect()
}

fn find_library(search: &[PathBuf]) -> Result<PathBuf, String> {
    for entry in search {
        if entry.as_os_str().is_empty() {
            continue;
        }
        if entry.is_file() {
            return Ok(entry.clone());
        }
        if let Some(p) = library_candidates(entry).into_iter().find(|p| p.is_file()) {
            return Ok(p);
        }
    }
    let tried: Vec<String> = search.iter().map(|p| p.display().to_string()).collect();
    Err(format!("the Canon CRM SDK ({LIB_NAME}) was not found; searched: {}", tried.join(", ")))
}

/// Libraries the SDK links or loads by bare name, loaded first from its own
/// directory so the loader finds them already in the process: the SDK need
/// not sit beside the executable, and the process's search path stays as it
/// is. Order: dependencies before their dependants.
#[cfg(target_os = "windows")]
const SIBLINGS: &[&str] = &["libiomp5md.dll", "cudart64_110.dll", "CESScalingLib.dll", "crxdec.dll"];
#[cfg(not(any(target_os = "windows", target_vendor = "apple")))]
const SIBLINGS: &[&str] = &[
    "libintlc.so.5",
    "libirng.so",
    "libimf.so",
    "libsvml.so",
    "libcudart.so.11.0",
    "libcrxdec.so.2.0",
];
#[cfg(target_vendor = "apple")]
const SIBLINGS: &[&str] = &[];

fn preload_siblings(dir: &Path) {
    for name in SIBLINGS {
        let p = dir.join(name);
        if !p.is_file() {
            continue;
        }
        #[cfg(unix)]
        let r = unsafe {
            libloading::os::unix::Library::open(Some(&p), libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL)
        }
        .map(libloading::Library::from);
        #[cfg(windows)]
        // SAFETY: one of the SDK's own runtime libraries; loading runs its initialisers.
        let r = unsafe { libloading::Library::new(&p) };
        match r {
            // Kept loaded for the life of the process, like the SDK itself.
            Ok(lib) => std::mem::forget(lib),
            Err(e) => eprintln!("CRM: could not preload {}: {e}", p.display()),
        }
    }
}

fn load(search: &[PathBuf]) -> Result<(CrmSdkLibrary, PathBuf), String> {
    let path = find_library(search)?;
    let path = std::fs::canonicalize(&path).unwrap_or(path);
    #[cfg(target_os = "windows")]
    if path.file_name().is_none_or(|n| n != LIB_NAME) {
        return Err(format!(
            "{} cannot be used: the Canon SDK only works when its file is named exactly {LIB_NAME}",
            path.display()
        ));
    }
    // `canonicalize` gives a `\\?\` path on Windows, which the SDK cannot strip
    // its own name from either; use the plain form.
    #[cfg(target_os = "windows")]
    let path = PathBuf::from(path.to_string_lossy().trim_start_matches(r"\\?\"));
    if let Some(dir) = path.parent() {
        preload_siblings(dir);
    }
    // SAFETY: the Canon CRM SDK library, located and name-checked above; loading
    // it runs its initialisers.
    let lib = unsafe { CrmSdkLibrary::load(&path) }.map_err(|e| format!("loading {} failed: {e}", path.display()))?;
    Ok((lib, path))
}

/// Point the SDK at its colour LUTs. It loads `<library dir>/CanonLUTs` by
/// itself at init; this finds them in the SDK's own layout too, where they sit
/// a few levels above the library. `CrmSetLUTsDirectory` takes the directory
/// that CONTAINS `CanonLUTs`, and reports success for one that does not, so
/// the check is ours.
fn apply_luts(lib: &CrmSdkLibrary, lib_path: &Path, explicit: Option<&Path>) {
    const LUTS: &str = "CanonLUTs";
    let dir = match explicit {
        Some(d) => {
            if !d.join(LUTS).is_dir() {
                eprintln!(
                    "CRM: {} has no {LUTS} folder (pass the directory that contains it); HDR and colour-space conversions will fail",
                    d.display()
                );
            }
            Some(d.to_path_buf())
        }
        // Up to the SDK root, from as deep as Mac_Lib/x86_64/CrmSdk.framework/Versions/A/CrmSdk.
        None => lib_path.ancestors().skip(1).take(8).find(|d| d.join(LUTS).is_dir()).map(Path::to_path_buf),
    };
    if let Some(dir) = dir {
        if let Err(e) = lib.set_luts_directory(&dir, true) {
            eprintln!("CRM: setting the LUTs directory {} failed: {e}", dir.display());
        }
    }
}

