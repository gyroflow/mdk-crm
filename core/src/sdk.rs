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

use crate::log;

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

/// Why [`init`] failed: before the SDK was handed the GPU's stream or queue,
/// or after, when the SDK may keep using them and they must outlive it.
pub enum InitError {
    BeforeBind(String),
    AfterBind(String),
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

/// Find and load the library, without binding any GPU. A failure is not
/// remembered: the SDK may be installed while the application runs.
pub fn load_library(search: &[PathBuf]) -> Result<(), String> {
    let mut s = state();
    if s.lib.is_none() {
        let (lib, path) = load(search)?;
        s.lib = Some((Arc::new(lib), path));
    }
    Ok(())
}

pub fn init(search: &[PathBuf], luts_dir: Option<&Path>, gpu: &Gpu) -> Result<(), InitError> {
    let key = gpu.key();
    if !matches!(key.lane, LANE_CUDA | LANE_OPENCL | LANE_METAL) {
        return Err(InitError::BeforeBind(format!("unknown GPU lane {}", gpu.lane)));
    }
    let mut s = state();
    if let Some(bound) = s.key {
        if bound == key {
            return Ok(());
        }
        return Err(InitError::BeforeBind(format!(
            "the Canon SDK is a process singleton and is already initialised for {} on another device or stream",
            lane_name(bound.lane)
        )));
    }
    if s.lib.is_none() {
        let (lib, path) = load(search).map_err(InitError::BeforeBind)?;
        s.lib = Some((Arc::new(lib), path));
    }
    let (lib, path) = s.lib.clone().expect("loaded above");
    // A no-op before the first init, and a clean slate after a failed one.
    // SAFETY: nothing has been created on the SDK: no regime is bound yet.
    let _ = unsafe { lib.terminate_sdk() };
    init_lane(&lib, gpu).map_err(InitError::AfterBind)?;
    if let Err(e) = check_supported(&lib, gpu.lane) {
        // SAFETY: nothing was created on the regime that was just bound.
        let _ = unsafe { lib.terminate_sdk() };
        return Err(InitError::AfterBind(e));
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
    log::info(&format!("the Canon SDK sees on {}: {}", lane_name(lane), names.join(", ")));
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

/// The SDK release whose structures (`*V13`) the bindings pass.
const SDK_MAJOR: u8 = 2;
const SDK_MINOR: u8 = 10;

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
    let tried: Vec<String> = search.iter().filter(|p| !p.as_os_str().is_empty()).map(|p| p.display().to_string()).collect();
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
        // Local scope: the loader matches the SDK's DT_NEEDED entries (and its
        // bare-name dlopen calls) against loaded objects by soname whatever
        // their scope, and these 2017 Intel runtimes and the CUDA runtime must
        // not interpose on copies other libraries of the process bring.
        #[cfg(unix)]
        let r = unsafe {
            libloading::os::unix::Library::open(Some(&p), libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL)
        }
        .map(libloading::Library::from);
        #[cfg(windows)]
        // SAFETY: one of the SDK's own runtime libraries; loading runs its initialisers.
        let r = unsafe { libloading::Library::new(&p) };
        match r {
            // Kept loaded for the life of the process, like the SDK itself.
            Ok(lib) => std::mem::forget(lib),
            Err(e) => log::warn(&format!("could not preload {}: {e}", p.display())),
        }
    }
}

/// `canonicalize` gives verbatim paths on Windows (`\\?\C:\...`,
/// `\\?\UNC\server\share\...`), which the SDK cannot strip its own name from.
/// The plain form of each; anything else as it is.
#[cfg(target_os = "windows")]
fn plain_path(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        let b = rest.as_bytes();
        if b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'\\' {
            return PathBuf::from(rest);
        }
    }
    p
}

/// Whether every character of `p` exists in the system's ANSI code page. The SDK opens its own files (its
/// decompressing library, its resources) by paths in that code page, so from a directory with other characters its
/// initialisation fails ("Failed to initialize the decompressing library").
#[cfg(target_os = "windows")]
fn in_ansi_code_page(p: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;
    unsafe extern "system" {
        fn WideCharToMultiByte(
            code_page: u32,
            flags: u32,
            wide: *const u16,
            wide_len: i32,
            out: *mut u8,
            out_len: i32,
            default_char: *const u8,
            used_default_char: *mut i32,
        ) -> i32;
    }
    const CP_ACP: u32 = 0;
    const WC_NO_BEST_FIT_CHARS: u32 = 0x400;
    let wide: Vec<u16> = p.as_os_str().encode_wide().collect();
    if wide.is_empty() {
        return true;
    }
    let mut used_default = 0i32;
    // SAFETY: `wide` holds `wide.len()` units; no output buffer is given, only the length is asked for.
    let n = unsafe {
        WideCharToMultiByte(
            CP_ACP,
            WC_NO_BEST_FIT_CHARS,
            wide.as_ptr(),
            wide.len() as i32,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            &mut used_default,
        )
    };
    n > 0 && used_default == 0
}

fn load(search: &[PathBuf]) -> Result<(CrmSdkLibrary, PathBuf), String> {
    let found = find_library(search)?;
    let path = std::fs::canonicalize(&found).unwrap_or_else(|_| found.clone());
    #[cfg(target_os = "windows")]
    if path.file_name().is_none_or(|n| n != LIB_NAME) {
        return Err(format!(
            "{} cannot be used: the Canon SDK only works when its file is named exactly {LIB_NAME}",
            path.display()
        ));
    }
    #[cfg(target_os = "windows")]
    let path = plain_path(path);
    #[cfg(target_os = "windows")]
    if path.to_string_lossy().starts_with(r"\\?\") {
        return Err(format!(
            "{} cannot be used: the Canon SDK cannot load from a path this long; install it in a shorter folder",
            path.display()
        ));
    }
    #[cfg(target_os = "windows")]
    if !in_ansi_code_page(&path) {
        return Err(format!(
            "{} cannot be used: the Canon SDK only works from a folder whose name has no characters outside this system's code page; install it in another folder",
            path.display()
        ));
    }
    if let Some(dir) = path.parent() {
        preload_siblings(dir);
    }
    // SAFETY: the Canon CRM SDK library, located and name-checked above; loading
    // it runs its initialisers.
    let lib = unsafe { CrmSdkLibrary::load(&path) }.map_err(|e| format!("loading {} failed: {e}", path.display()))?;
    match lib.get_sdk_info() {
        Ok((v, _, _)) if v.major != SDK_MAJOR || v.minor < SDK_MINOR => {
            return Err(format!(
                "{} is Canon CRM SDK {}.{}; this plugin needs {SDK_MAJOR}.{SDK_MINOR} or a later {SDK_MAJOR}.x",
                path.display(),
                v.major,
                v.minor
            ));
        }
        Ok((v, _, _)) if v.minor != SDK_MINOR => {
            log::warn(&format!(
                "{} is Canon CRM SDK {}.{}; this plugin was made for {SDK_MAJOR}.{SDK_MINOR}",
                path.display(),
                v.major,
                v.minor
            ));
        }
        Ok(_) => {}
        Err(e) => log::warn(&format!("could not read the version of {}: {e}", path.display())),
    }
    Ok((lib, path))
}

/// Where `CanonLUTs` can be for the library at `lib_path`: beside it, beside
/// the framework it is in, and the root of the SDK's own layout
/// (`CRMSDK/CanonLUTs` above `Win_Lib/x64`, `Linux_Lib`, `Mac_Lib/x86_64`).
/// Nowhere else, so that an unrelated folder further up is never taken.
fn luts_candidates(lib_path: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let Some(dir) = lib_path.parent() else { return v };
    v.push(dir.to_path_buf());
    // .../CrmSdk.framework/Versions/A/CrmSdk or .../CrmSdk.framework/CrmSdk
    let framework = lib_path.ancestors().find(|a| a.extension().is_some_and(|e| e == "framework"));
    let lib_home = match framework {
        Some(f) => {
            let parent = f.parent().map(Path::to_path_buf);
            v.extend(parent.clone());
            parent
        }
        None => Some(dir.to_path_buf()),
    };
    if let Some(home) = lib_home {
        let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        let parent = home.parent();
        let grandparent = parent.and_then(Path::parent);
        match (name(&home).as_str(), parent.map(name).as_deref()) {
            ("x64", Some("win_lib")) | ("x86_64", Some("mac_lib")) => v.extend(grandparent.map(Path::to_path_buf)),
            ("linux_lib", _) => v.extend(parent.map(Path::to_path_buf)),
            _ => {}
        }
    }
    v
}

/// Point the SDK at its colour LUTs. `CrmSetLUTsDirectory` takes the directory
/// that CONTAINS `CanonLUTs`, and reports success for one that does not, so the
/// check is ours. Every develop recipe uses them, the default Rec.709 one too,
/// so a missing folder makes every frame fail.
fn apply_luts(lib: &CrmSdkLibrary, lib_path: &Path, explicit: Option<&Path>) {
    const LUTS: &str = "CanonLUTs";
    let dir = match explicit {
        Some(d) => {
            if !d.join(LUTS).is_dir() {
                log::warn(&format!(
                    "{} has no {LUTS} folder (pass the directory that contains it); developing will fail",
                    d.display()
                ));
            }
            Some(d.to_path_buf())
        }
        None => {
            let candidates = luts_candidates(lib_path);
            let found = candidates.iter().find(|d| d.join(LUTS).is_dir()).cloned();
            if found.is_none() {
                let tried: Vec<String> = candidates.iter().map(|d| d.display().to_string()).collect();
                log::warn(&format!(
                    "no {LUTS} folder beside the Canon SDK (looked in {}); developing will fail. Set CRMSDK_LUTS_DIR to the directory that contains it",
                    tried.join(", ")
                ));
            }
            found
        }
    };
    if let Some(dir) = dir
        && let Err(e) = lib.set_luts_directory(&dir, true)
    {
        log::warn(&format!("setting the LUTs directory {} failed: {e}", dir.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "windows")]
    #[test]
    fn plain_paths() {
        assert_eq!(plain_path(PathBuf::from(r"\\?\C:\a\CrmSdk.dll")), PathBuf::from(r"C:\a\CrmSdk.dll"));
        assert_eq!(plain_path(PathBuf::from(r"\\?\UNC\srv\share\b\CrmSdk.dll")), PathBuf::from(r"\\srv\share\b\CrmSdk.dll"));
        assert_eq!(plain_path(PathBuf::from(r"C:\a\CrmSdk.dll")), PathBuf::from(r"C:\a\CrmSdk.dll"));
        assert_eq!(plain_path(PathBuf::from(r"\\?\Volume{x}\a")), PathBuf::from(r"\\?\Volume{x}\a"));
    }

    #[test]
    fn luts_places() {
        let sdk = Path::new("/x/CRMSDK210/CRMSDK");
        let c = luts_candidates(&sdk.join("Win_Lib/x64/CrmSdk.dll"));
        assert_eq!(c, vec![sdk.join("Win_Lib/x64"), sdk.to_path_buf()]);
        let c = luts_candidates(&sdk.join("Linux_Lib/libCrmSdk.so"));
        assert_eq!(c, vec![sdk.join("Linux_Lib"), sdk.to_path_buf()]);
        let c = luts_candidates(&sdk.join("Mac_Lib/x86_64/CrmSdk.framework/Versions/A/CrmSdk"));
        assert_eq!(
            c,
            vec![sdk.join("Mac_Lib/x86_64/CrmSdk.framework/Versions/A"), sdk.join("Mac_Lib/x86_64"), sdk.to_path_buf()]
        );
        let app = Path::new("/Applications/Gyroflow.app/Contents/Frameworks");
        let c = luts_candidates(&app.join("CrmSdk.framework/Versions/A/CrmSdk"));
        assert_eq!(c, vec![app.join("CrmSdk.framework/Versions/A"), app.to_path_buf()]);
        let c = luts_candidates(Path::new("/opt/gyroflow/lib/libCrmSdk.so"));
        assert_eq!(c, vec![PathBuf::from("/opt/gyroflow/lib")]);
    }
}
