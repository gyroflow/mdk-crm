// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode core of the MDK CRM plugin: the C API declared in `src/crm_core.h`,
//! over `crm-container` and `canon-raw`.
//!
//! Every entry point catches panics, so none unwinds into C++.

use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

mod reader;
mod sdk;

use reader::{Info, OpenParams, Output, Reader};

/// Diagnostics, through the plugin's logger when it has set one (`crmc_set_log`).
pub(crate) mod log {
    use std::ffi::{CString, c_char};
    use std::sync::atomic::{AtomicPtr, Ordering};

    pub type LogFn = unsafe extern "C" fn(level: i32, msg: *const c_char);
    static LOGGER: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

    // 1 is an error; the core reports errors through its return values instead.
    pub const WARNING: i32 = 2;
    pub const INFO: i32 = 3;

    pub fn set(f: Option<LogFn>) {
        LOGGER.store(f.map_or(std::ptr::null_mut(), |f| f as *mut ()), Ordering::Release);
    }

    fn write(level: i32, msg: &str) {
        let p = LOGGER.load(Ordering::Acquire);
        if p.is_null() {
            eprintln!("CRM: {msg}");
            return;
        }
        // SAFETY: only `set` stores here, and it stores a `LogFn`.
        let f: LogFn = unsafe { std::mem::transmute::<*mut (), LogFn>(p) };
        let s = CString::new(msg.replace('\0', " ")).unwrap_or_default();
        // SAFETY: the plugin's callback, given a NUL-terminated string valid for the call.
        unsafe { f(level, s.as_ptr()) }
    }

    pub fn warn(msg: &str) {
        write(WARNING, msg)
    }
    pub fn info(msg: &str) {
        write(INFO, msg)
    }
}

/// `crmc_open_params`.
#[repr(C)]
pub struct CrmcOpenParams {
    pub gamma: i32,
    pub color_space: i32,
    pub depth: i32,
    pub resolution: i32,
}

/// `crmc_info`.
#[repr(C)]
pub struct CrmcInfo {
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps_num: u32,
    pub fps_den: u32,
    pub bit_depth: u32,
    pub pixel_aspect: f32,
    pub gamma: i32,
    pub color_space: i32,
    pub depth: i32,
    pub resolution: i32,
}

impl From<Info> for CrmcInfo {
    fn from(i: Info) -> Self {
        Self {
            width: i.width,
            height: i.height,
            frame_count: i.frame_count,
            fps_num: i.fps_num,
            fps_den: i.fps_den,
            bit_depth: i.bit_depth,
            pixel_aspect: i.pixel_aspect,
            gamma: i.gamma,
            color_space: i.color_space,
            depth: i.depth,
            resolution: i.resolution,
        }
    }
}

/// `crmc_output`.
#[repr(C)]
pub struct CrmcOutput {
    pub width: u32,
    pub height: u32,
    pub bytes_per_line: u32,
    pub size: u32,
    pub level: i32,
}

impl From<Output> for CrmcOutput {
    fn from(o: Output) -> Self {
        Self { width: o.width, height: o.height, bytes_per_line: o.bytes_per_line, size: o.size, level: o.level }
    }
}

/// `crmc_reader`. The info and metadata never change after open, so they are
/// read without the lock.
pub struct CrmcReader {
    info: Info,
    metadata: Vec<(std::ffi::CString, std::ffi::CString)>,
    reader: Mutex<Reader>,
}

impl CrmcReader {
    fn lock(&self) -> std::sync::MutexGuard<'_, Reader> {
        self.reader.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn set_err(err: *mut c_char, len: usize, msg: &str) {
    if err.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let mut n = bytes.len().min(len - 1);
    // Do not cut a UTF-8 sequence in half.
    while n > 0 && n < bytes.len() && (bytes[n] & 0xC0) == 0x80 {
        n -= 1;
    }
    // SAFETY: the caller gives `len` writable bytes at `err`.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), err as *mut u8, n);
        *err.add(n) = 0;
    }
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("internal error: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("internal error: {s}")
    } else {
        "internal error".into()
    }
}

/// Run `f`, turning an error or a panic into `fail` plus a message in `err`.
fn guarded<T>(err: *mut c_char, err_len: usize, fail: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_err(err, err_len, &e);
            fail
        }
        Err(p) => {
            set_err(err, err_len, &panic_message(p));
            fail
        }
    }
}

/// # Safety
/// `s` is null or a NUL-terminated string.
unsafe fn path_arg(s: *const c_char) -> Option<PathBuf> {
    if s.is_null() {
        return None;
    }
    // SAFETY: per the contract.
    let s = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
    (!s.is_empty()).then(|| PathBuf::from(s))
}

/// # Safety
/// `search` holds `n_search` entries, each null or a NUL-terminated string.
unsafe fn search_args(search: *const *const c_char, n_search: usize) -> Vec<PathBuf> {
    let mut dirs = Vec::with_capacity(n_search);
    if search.is_null() {
        return dirs;
    }
    for i in 0..n_search {
        // SAFETY: per the contract.
        if let Some(p) = unsafe { path_arg(*search.add(i)) } {
            dirs.push(p);
        }
    }
    dirs
}

/// # Safety
/// `f` is null or a function callable from any thread for the life of the process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_set_log(f: Option<log::LogFn>) {
    log::set(f);
}

/// # Safety
/// See `crm_core.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_sdk_load(
    search: *const *const c_char,
    n_search: usize,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    guarded(err, err_len, -1, || {
        let dirs = unsafe { search_args(search, n_search) };
        sdk::load_library(&dirs)?;
        Ok(0)
    })
}

/// # Safety
/// See `crm_core.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_sdk_init(
    search: *const *const c_char,
    n_search: usize,
    luts_dir: *const c_char,
    gpu: *const sdk::Gpu,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // -1: failed before the SDK was handed the GPU; -2: after (see crm_core.h).
    let mut code = -1;
    let r = guarded(err, err_len, -1, || {
        let gpu = unsafe { gpu.as_ref() }.ok_or("no GPU given")?;
        let dirs = unsafe { search_args(search, n_search) };
        let luts = unsafe { path_arg(luts_dir) };
        match sdk::init(&dirs, luts.as_deref(), gpu) {
            Ok(()) => Ok(0),
            Err(sdk::InitError::BeforeBind(e)) => Err(e),
            Err(sdk::InitError::AfterBind(e)) => {
                code = -2;
                Err(e)
            }
        }
    });
    if r == 0 { 0 } else { code }
}

/// # Safety
/// See `crm_core.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_open(
    path: *const c_char,
    params: *const CrmcOpenParams,
    err: *mut c_char,
    err_len: usize,
) -> *mut CrmcReader {
    guarded(err, err_len, std::ptr::null_mut(), || {
        let path = unsafe { path_arg(path) }.ok_or("no path given")?;
        let p = unsafe { params.as_ref() };
        let gamma = match p.map_or(-1, |p| p.gamma) {
            -1 => None,
            v => Some(reader::gamma_from(v).ok_or_else(|| format!("unknown gamma {v}"))?),
        };
        let color_space = match p.map_or(-1, |p| p.color_space) {
            -1 => None,
            v => Some(reader::color_space_from(v).ok_or_else(|| format!("unknown colour space {v}"))?),
        };
        let depth = p.map_or(reader::MAX_DEPTH as i32, |p| p.depth).max(0) as usize;
        let resolution = match p.map_or(0, |p| p.resolution) {
            0 => None,
            v => Some(reader::resolution_from(v).ok_or_else(|| format!("unknown resolution class {v}"))?),
        };
        let r = Reader::open(Path::new(&path), &OpenParams { gamma, color_space, depth, resolution })?;
        let handle =
            CrmcReader { info: r.info(), metadata: r.metadata().to_vec(), reader: Mutex::new(r) };
        Ok(Box::into_raw(Box::new(handle)))
    })
}

/// # Safety
/// `r` is null or a reader from `crmc_open`, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_close(r: *mut CrmcReader) {
    if r.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: per the contract.
        drop(unsafe { Box::from_raw(r) });
    }));
}

/// # Safety
/// `r` is a live reader, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_get_info(r: *const CrmcReader, out: *mut CrmcInfo) {
    if let (Some(r), false) = (unsafe { r.as_ref() }, out.is_null()) {
        unsafe { out.write(r.info.into()) };
    }
}

/// # Safety
/// `r` is a live reader, `key` and `value` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_metadata(
    r: *const CrmcReader,
    i: usize,
    key: *mut *const c_char,
    value: *mut *const c_char,
) -> i32 {
    let Some(r) = (unsafe { r.as_ref() }) else { return 0 };
    let Some((k, v)) = r.metadata.get(i) else { return 0 };
    unsafe {
        if !key.is_null() {
            *key = k.as_ptr();
        }
        if !value.is_null() {
            *value = v.as_ptr();
        }
    }
    1
}

/// # Safety
/// `r` is a live reader, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_output_for(r: *const CrmcReader, level: i32, out: *mut CrmcOutput) -> i32 {
    guarded(std::ptr::null_mut(), 0, -1, || {
        let r = unsafe { r.as_ref() }.ok_or("no reader")?;
        let level = reader::level_from_factor(level).ok_or("unknown level")?;
        let o = r.lock().output_for(level)?;
        if !out.is_null() {
            unsafe { out.write(o.into()) };
        }
        Ok(0)
    })
}

/// # Safety
/// `r` is a live reader, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_get_output(r: *const CrmcReader, out: *mut CrmcOutput) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let (Some(r), false) = (unsafe { r.as_ref() }, out.is_null()) {
            unsafe { out.write(r.lock().output().into()) };
        }
    }));
}

/// # Safety
/// `r` is a live reader, `out` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_set_level(
    r: *mut CrmcReader,
    level: i32,
    out: *mut CrmcOutput,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    guarded(err, err_len, -1, || {
        let r = unsafe { r.as_ref() }.ok_or("no reader")?;
        let level = reader::level_from_factor(level).ok_or_else(|| format!("unknown level {level}"))?;
        let o = r.lock().set_level(level)?;
        if !out.is_null() {
            unsafe { out.write(o.into()) };
        }
        Ok(0)
    })
}

/// # Safety
/// See `crm_core.h`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn crmc_decode(
    r: *mut CrmcReader,
    frame: u64,
    dst: *mut c_void,
    dst_size: u32,
    sync: Option<unsafe extern "C" fn(*mut c_void) -> i32>,
    user: *mut c_void,
    prefetch: i32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    guarded(err, err_len, -1, || {
        let r = unsafe { r.as_ref() }.ok_or("no reader")?;
        let mut sync = || -> Result<(), String> {
            let Some(f) = sync else { return Ok(()) };
            // SAFETY: the caller's callback and its own argument.
            match unsafe { f(user) } {
                0 => Ok(()),
                e => Err(format!("the GPU did not finish the work (error {e})")),
            }
        };
        r.lock().decode(frame, dst, dst_size, &mut sync, prefetch != 0)?;
        Ok(0)
    })
}

/// # Safety
/// `r` is a live reader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_flush(r: *mut CrmcReader) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(r) = unsafe { r.as_ref() } {
            r.lock().flush();
        }
    }));
}

/// # Safety
/// `f` is a valid callback for `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crmc_cuda_exclusive(f: Option<unsafe extern "C" fn(*mut c_void)>, user: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _lock = canon_raw::cuda_exclusion();
        if let Some(f) = f {
            // SAFETY: per the contract.
            unsafe { f(user) }
        }
    }));
}
