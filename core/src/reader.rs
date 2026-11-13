// SPDX-License-Identifier: MIT OR Apache-2.0

//! One open clip: its container, the develop recipe, and the decode pipeline.
//!
//! # The pipeline
//!
//! The SDK decompresses a frame on the CPU (tens of milliseconds for 6K) and
//! develops it on the GPU (a few). Done in one call per frame, the GPU idles
//! while the CPU decompresses. So by default the two run as separate stages,
//! as Canon's own sample does: a DECOMPRESS-only process ref keeps up to
//! three frames decompressing asynchronously on the CPU (`CrmGetImageAsync`
//! refuses a fourth, and refuses develop work outright), and a DEVELOP-only
//! ref on the GPU develops each one, synchronously, into the caller's buffer.
//! Both stages must use the same decode level.
//!
//! `depth = 0` keeps one synchronous decompress + develop per frame
//! ([`DecodeContext`]).
//!
//! # Scaling
//!
//! Scaling uses `DecodeLevel` (half / quarter), which skips work in the
//! decompress itself. `Downscale` resamples after a full develop and is slower
//! than not scaling at all, and the SDK refuses the two together. A reduced
//! level also requires the HighSpeed debayer, so a level is always applied
//! together with its debayer setting.

use std::collections::VecDeque;
use std::ffi::{CString, c_void};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use canon_raw::{
    AcesIdt, BufferType, ColorSpace, CrmMemory, CrmSdkLibrary, CrmVoid, DataType, DebayerSetting, DecodeContext,
    DecodeLevel, DevParamV13, Device, Downscale, FormatId, FrameRendering, Gamma, Gpu as SdkGpu, Header,
    IsoSetting, LensSqueeze, OutputGeometry, OwnedGetImageFuture, Point, ProcessRefLease, ProcessType, Rect,
    ResParamV13, Resolution, SampleSequence, StructId, WbGain, WbSetting, image_info,
};
use crm_container::CrmContainer;

use crate::sdk::{self, LANE_CUDA, LANE_METAL, LANE_OPENCL};

/// RGBA, 16 bits per sample, little-endian: the only four-channel output the
/// SDK has, and the only one a GPU can sample directly.
const BITS: u32 = 16;
const SAMPLES: u32 = 4;
const SEQUENCE: SampleSequence = SampleSequence::Rgba16BitLe;

/// `CrmGetImageAsync` accepts at most three jobs at once.
pub const MAX_DEPTH: usize = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct Output {
    pub width: u32,
    pub height: u32,
    pub bytes_per_line: u32,
    pub size: u32,
    pub level: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Info {
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
}

pub struct OpenParams {
    pub gamma: Option<Gamma>,
    pub color_space: Option<ColorSpace>,
    pub depth: usize,
}

/// The decode level for a scale factor: 1, 2 or 4.
pub fn level_from_factor(factor: i32) -> Option<DecodeLevel> {
    match factor {
        1 => Some(DecodeLevel::Full),
        2 => Some(DecodeLevel::Half),
        4 => Some(DecodeLevel::Quarter),
        _ => None,
    }
}

fn factor(level: DecodeLevel) -> i32 {
    match level {
        DecodeLevel::Full => 1,
        DecodeLevel::Half => 2,
        DecodeLevel::Quarter => 4,
    }
}

/// The only debayer the SDK accepts at a reduced level is HighSpeed; Full
/// keeps the HighQuality one.
fn debayer_for(level: DecodeLevel) -> DebayerSetting {
    if level == DecodeLevel::Full { DebayerSetting::HighQuality } else { DebayerSetting::HighSpeed }
}

fn lane_device(lane: i32) -> Device {
    match lane {
        LANE_CUDA => Device::Cuda,
        LANE_OPENCL => Device::OpenCl,
        LANE_METAL => Device::Metal,
        _ => Device::Cpu,
    }
}

fn lane_buffer(lane: i32) -> BufferType {
    match lane {
        LANE_CUDA => BufferType::Cuda,
        LANE_OPENCL => BufferType::OpenCl,
        LANE_METAL => BufferType::Metal,
        _ => BufferType::Cpu,
    }
}

/// What does not change for the life of a clip, plus the recipe.
struct Clip {
    sdk: Arc<CrmSdkLibrary>,
    lane: i32,
    format_id: FormatId,
    uuid_box: Arc<[u8]>,
    craw_entry: Arc<[u8]>,
    /// A header pack over frame 0's timed metadata, for raster queries:
    /// `CrmGetOutImageSize` needs a pack, and timed metadata is colour-only.
    probe_hdr: Vec<u8>,
    crop: Rect,
    dev: DevParamV13,
    level: DecodeLevel,
    frame_count: u64,
}

impl Clip {
    #[allow(clippy::too_many_arguments)]
    fn res_param(&self, in_dt: DataType, out_dt: DataType, in_bt: BufferType, out_bt: BufferType) -> ResParamV13 {
        ResParamV13 {
            size: std::mem::size_of::<ResParamV13>() as _,
            struct_id: StructId::ResParamV13,
            format_id: self.format_id,
            debayer_setting: debayer_for(self.level),
            downscale: Downscale::X1,
            crop: self.crop,
            frame_rendering: FrameRendering::Split,
            lens_squeeze: LensSqueeze::Off,
            in_data_type: in_dt,
            in_buffer_type: in_bt,
            out_data_type: out_dt,
            out_buffer_type: out_bt,
            decode_level: self.level,
        }
    }

    fn header_pack(&self, timed_meta: &[u8]) -> Result<Vec<u8>, String> {
        self.sdk
            .create_header_pack(&[
                Header::uuid(&self.uuid_box),
                Header::craw(&self.craw_entry),
                Header::timed_metadata(timed_meta),
            ])
            .map_err(|e| format!("header pack: {e}"))
    }

    fn output_at(&self, level: DecodeLevel) -> Result<Output, String> {
        let mut p = self.res_param(DataType::CrmCompressed, DataType::RgbLutApplied, BufferType::Cpu, BufferType::Cpu);
        p.decode_level = level;
        p.debayer_setting = debayer_for(level);
        let (w, h) = self.sdk.get_out_image_size(&p, &self.probe_hdr).map_err(|e| format!("output size: {e}"))?;
        if w == 0 || h == 0 {
            return Err(format!("the SDK reports an empty output at 1/{}", factor(level)));
        }
        let bpl = OutputGeometry::bytes_per_line(w, BITS, SAMPLES);
        Ok(Output { width: w, height: h, bytes_per_line: bpl, size: bpl * h, level: factor(level) })
    }

    fn develop_lock(&self) -> Option<canon_raw::CudaExclusionGuard> {
        (self.lane == LANE_CUDA).then(canon_raw::cuda_exclusion)
    }
}

/// One frame decompressing on the CPU. Fields drop in order: the future
/// first (dropping it waits for the job), then the buffers the job uses.
struct Job {
    fut: OwnedGetImageFuture,
    frame: u64,
    bayer: CrmMemory,
    hdr: Arc<[u8]>,
}

struct Split {
    /// First: dropping a job waits for it, and that must happen before the
    /// refs it runs on are parked.
    inflight: VecDeque<Job>,
    free: Vec<CrmMemory>,
    develop_ref: ProcessRefLease,
    decompress_ref: ProcessRefLease,
    depth: usize,
    bayer_bpl: u32,
    bayer_size: u32,
}

impl Split {
    fn new(clip: &Clip, gpu: SdkGpu, depth: usize) -> Result<Self, String> {
        let sdk = &clip.sdk;
        // Process refs are leased, never released early: releasing one soon
        // after creating it can crash the Windows SDK.
        let decompress_ref = sdk
            .lease_process_ref(clip.format_id, Device::Cpu, SdkGpu::NotUse, ProcessType::DECOMPRESS, Resolution::K8)
            .map_err(|e| format!("decompress process ref: {e}"))?;
        let develop_ref = sdk
            .lease_process_ref(clip.format_id, lane_device(clip.lane), gpu, ProcessType::DEVELOP, Resolution::K8)
            .map_err(|e| format!("develop process ref: {e}"))?;
        let mut s = Self {
            inflight: VecDeque::with_capacity(depth),
            free: Vec::with_capacity(depth),
            develop_ref,
            decompress_ref,
            depth,
            bayer_bpl: 0,
            bayer_size: 0,
        };
        s.resize(clip)?;
        Ok(s)
    }

    /// Size the decompressed Bayer raster for the clip's level. It is the RAW
    /// frame, larger than the capture area, so it comes from the SDK.
    fn resize(&mut self, clip: &Clip) -> Result<(), String> {
        self.drain();
        let p = clip.res_param(DataType::CrmCompressed, DataType::CrmDecompressed, BufferType::Cpu, BufferType::Cpu);
        let (w, h) = clip.sdk.get_out_image_size(&p, &clip.probe_hdr).map_err(|e| format!("Bayer size: {e}"))?;
        self.bayer_bpl = w * 2;
        self.bayer_size = self.bayer_bpl * h;
        self.free.clear();
        Ok(())
    }

    /// Wait for every job in flight and drop its result. There is no cancel:
    /// resetting the ref is unsafe with jobs outstanding.
    fn drain(&mut self) {
        while let Some(job) = self.inflight.pop_front() {
            let Job { fut, bayer, .. } = job;
            let _ = pollster::block_on(fut);
            self.free.push(bayer);
        }
    }

    fn submit(&mut self, clip: &Clip, movie: &mut CrmContainer<File>, frame: u64) -> Result<(), String> {
        let comp: Arc<[u8]> =
            movie.read_compressed_frame(frame).map_err(|e| format!("reading frame {frame}: {e}"))?.into();
        let meta = movie.read_timed_meta(frame).map_err(|e| format!("reading the metadata of frame {frame}: {e}"))?;
        let hdr: Arc<[u8]> = clip.header_pack(&meta)?.into();
        let bayer = match self.free.pop() {
            Some(b) => b,
            None => CrmMemory::new(clip.sdk.clone(), self.bayer_size, Device::Cpu)
                .map_err(|e| format!("allocating a Bayer buffer: {e}"))?,
        };
        let res = clip.res_param(DataType::CrmCompressed, DataType::CrmDecompressed, BufferType::Cpu, BufferType::Cpu);
        // SAFETY: the decompress ref is live for `self`; `comp` and `hdr` back the
        // source and header pointers and are kept by the future; `bayer` holds
        // `bayer_size` bytes and lives in the same `Job`, dropped after the future.
        let fut = unsafe {
            clip.sdk.get_image_future_owned(
                self.decompress_ref.process_ref(),
                bayer.as_ptr(),
                self.bayer_size,
                16,
                1,
                self.bayer_bpl,
                0, // a 16-bit single-channel Bayer has no sample sequence
                comp.as_ptr(),
                comp.len() as u32,
                Box::new(clip.dev),
                Box::new(res),
                hdr.as_ptr(),
                hdr.len() as u32,
                Box::new(comp.clone()),
                Box::new(hdr.clone()),
            )
        };
        match fut {
            Ok(fut) => {
                self.inflight.push_back(Job { fut, frame, bayer, hdr });
                Ok(())
            }
            Err(e) => {
                self.free.push(bayer);
                Err(format!("decompressing frame {frame}: {e}"))
            }
        }
    }

    /// Keep `from`, `from + 1`, ... decompressing, up to `depth` jobs.
    fn fill(&mut self, clip: &Clip, movie: &mut CrmContainer<File>, from: u64) -> Result<(), String> {
        let mut next = self.inflight.back().map_or(from, |j| j.frame + 1);
        while self.inflight.len() < self.depth && next < clip.frame_count {
            self.submit(clip, movie, next)?;
            next += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        clip: &Clip,
        movie: &mut CrmContainer<File>,
        out: &Output,
        frame: u64,
        dst: *mut c_void,
        sync: &mut dyn FnMut(),
        prefetch: bool,
    ) -> Result<(), String> {
        // Anything but this frame at the head is stale: a seek or a step back.
        if self.inflight.front().map(|j| j.frame) != Some(frame) {
            self.drain();
        }
        if self.inflight.is_empty() {
            self.submit(clip, movie, frame)?;
        }
        if prefetch {
            // A read error ahead surfaces when that frame is decoded.
            let _ = self.fill(clip, movie, frame);
        }
        let Job { fut, bayer, hdr, .. } = self.inflight.pop_front().expect("submitted above");
        let decompressed = match pollster::block_on(fut) {
            Ok(o) => o.dst_data_size,
            Err(e) => {
                self.free.push(bayer);
                return Err(format!("decompressing frame {frame}: {e}"));
            }
        };
        let size = if decompressed > 0 { decompressed.min(self.bayer_size) } else { self.bayer_size };
        // SAFETY: the job has completed, so `bayer` holds `size` decompressed bytes.
        let src = unsafe { std::slice::from_raw_parts(bayer.as_ptr() as *const u8, size as usize) };
        let res = clip.res_param(DataType::CrmDecompressed, DataType::RgbLutApplied, BufferType::Cpu, lane_buffer(clip.lane));
        let r = {
            // The develop and the caller's synchronise are both CUDA work, which
            // spec 9.2.1 forbids from overlapping other CUDA calls of the process.
            let _lock = clip.develop_lock();
            // SAFETY: the develop ref is live for `self`; `dst` is a device buffer
            // of the lane holding at least `out.size` bytes (checked by the caller).
            let r = unsafe {
                clip.sdk.get_image(
                    self.develop_ref.process_ref(),
                    dst as *mut CrmVoid,
                    out.size,
                    BITS,
                    SAMPLES,
                    out.bytes_per_line,
                    SEQUENCE,
                    src,
                    &clip.dev,
                    &res,
                    &hdr,
                )
            };
            if r.is_ok() {
                sync();
            }
            r
        };
        self.free.push(bayer);
        if let Err(e) = r {
            let detail = clip.sdk.get_last_error_string().unwrap_or_default();
            return Err(format!("developing frame {frame}: {e} {detail}").trim_end().to_string());
        }
        if prefetch {
            let _ = self.fill(clip, movie, frame + 1);
        }
        Ok(())
    }
}

enum Pipe {
    Split(Box<Split>),
    Combined(Box<DecodeContext>),
}

pub struct Reader {
    /// First, so that jobs in flight finish and process refs are parked
    /// before anything else goes.
    pipe: Pipe,
    clip: Clip,
    movie: CrmContainer<File>,
    out: Output,
    info: Info,
    metadata: Vec<(CString, CString)>,
}

impl Reader {
    pub fn open(path: &Path, params: &OpenParams) -> Result<Self, String> {
        let (sdk, key) = sdk::current().ok_or("the Canon SDK is not initialised")?;
        let file = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
        let mut movie = CrmContainer::load(file).map_err(|e| format!("not a readable CRM clip: {e}"))?;
        let frame_count = movie.frame_count();
        if frame_count == 0 {
            return Err("the clip has no video frames".into());
        }
        if movie.timed_meta_count() != frame_count {
            return Err("the clip's timed metadata does not have one entry per video frame".into());
        }
        let uuid_box: Arc<[u8]> =
            movie.canon_uuid_box().ok_or("the clip has no Canon metadata box")?.to_vec().into();
        let craw_entry: Arc<[u8]> = movie.craw_entry().to_vec().into();
        let meta0 = movie.read_timed_meta(0).map_err(|e| format!("reading the metadata of frame 0: {e}"))?;
        let probe_hdr = sdk
            .create_header_pack(&[Header::uuid(&uuid_box), Header::craw(&craw_entry), Header::timed_metadata(&meta0)])
            .map_err(|e| format!("header pack: {e}"))?;
        let format_id = sdk.get_format_id(&probe_hdr).map_err(|e| format!("not a clip the SDK supports: {e}"))?;
        let size_err = |e: canon_raw::CrmError| format!("clip size: {e}");
        let cw = u32::from(sdk.get_image_info(format_id, &probe_hdr, image_info::CaptureWidth).map_err(size_err)?);
        let ch = u32::from(sdk.get_image_info(format_id, &probe_hdr, image_info::CaptureHeight).map_err(size_err)?);

        let (fps_num, fps_den) = frame_rate(&movie, &sdk, format_id, &probe_hdr);
        let bit_depth = sdk.get_image_info(format_id, &probe_hdr, image_info::BitDepth).map(u32::from).unwrap_or(0);
        let pixel_aspect = match sdk.get_image_info(format_id, &probe_hdr, image_info::LensSqueezeFactor) {
            Ok(1) => 1.3,
            Ok(2) => 2.0,
            _ => 1.0,
        };

        let gamma = params.gamma.unwrap_or(Gamma::Rec709);
        let color_space = params.color_space.unwrap_or(ColorSpace::Bt709);
        let dev = DevParamV13 {
            size: std::mem::size_of::<DevParamV13>() as _,
            struct_id: StructId::DevParamV13,
            format_id,
            wb_setting: WbSetting::Record,
            click_wb_point: Point { x: 0, y: 0 },
            tint: 0.0,
            exposure: 0.0,
            cct: 0,
            sharpness: 0,
            noise: 0,
            gamma,
            wb_gain: WbGain { r: 0, g1: 0, g2: 0, b: 0 },
            aces_idt: AcesIdt::Auto,
            color_space,
            iso_setting: IsoSetting::Record,
            iso_speed: 0,
        };
        let clip = Clip {
            sdk: sdk.clone(),
            lane: key.lane,
            format_id,
            uuid_box,
            craw_entry,
            probe_hdr,
            crop: Rect { x: 0, y: 0, width: cw, height: ch },
            dev,
            level: DecodeLevel::Full,
            frame_count,
        };
        let full = clip.output_at(DecodeLevel::Full)?;
        let gpu = pick_gpu(&sdk, lane_device(key.lane));

        let depth = params.depth.min(MAX_DEPTH);
        let pipe = match (depth > 0).then(|| Split::new(&clip, gpu, depth)) {
            Some(Ok(s)) => Pipe::Split(Box::new(s)),
            other => {
                if let Some(Err(e)) = other {
                    eprintln!("CRM: the read-ahead pipeline is unavailable ({e}); decoding one frame at a time");
                }
                Pipe::Combined(Box::new(combined(&clip, gpu, &meta0)?))
            }
        };
        let depth = if matches!(pipe, Pipe::Split(_)) { depth } else { 0 };

        let metadata = clip_metadata(&sdk, format_id, &clip.probe_hdr, key.lane, fps_num, fps_den);
        let info = Info {
            width: full.width,
            height: full.height,
            frame_count,
            fps_num,
            fps_den,
            bit_depth,
            pixel_aspect,
            gamma: gamma as i32,
            color_space: color_space as i32,
            depth: depth as i32,
        };
        Ok(Self { pipe, clip, movie, out: full, info, metadata })
    }

    pub fn info(&self) -> Info {
        self.info
    }

    pub fn metadata(&self) -> &[(CString, CString)] {
        &self.metadata
    }

    pub fn output(&self) -> Output {
        self.out
    }

    pub fn output_for(&self, level: DecodeLevel) -> Result<Output, String> {
        self.clip.output_at(level)
    }

    pub fn set_level(&mut self, level: DecodeLevel) -> Result<Output, String> {
        if level == self.clip.level {
            return Ok(self.out);
        }
        let out = self.clip.output_at(level)?;
        let previous = self.clip.level;
        self.clip.level = level;
        let r = match &mut self.pipe {
            Pipe::Split(s) => s.resize(&self.clip),
            Pipe::Combined(ctx) => ctx
                .edit_res_param(|p| {
                    p.decode_level = level;
                    p.debayer_setting = debayer_for(level);
                })
                .map(|_| ())
                .map_err(|e| e.to_string()),
        };
        if let Err(e) = r {
            self.clip.level = previous;
            if let Pipe::Split(s) = &mut self.pipe {
                let _ = s.resize(&self.clip);
            }
            return Err(format!("switching to 1/{}: {e}", factor(level)));
        }
        self.out = out;
        Ok(out)
    }

    pub fn flush(&mut self) {
        if let Pipe::Split(s) = &mut self.pipe {
            s.drain();
        }
    }

    pub fn decode(
        &mut self,
        frame: u64,
        dst: *mut c_void,
        dst_size: u32,
        sync: &mut dyn FnMut(),
        prefetch: bool,
    ) -> Result<(), String> {
        if frame >= self.clip.frame_count {
            return Err(format!("frame {frame} is past the end ({} frames)", self.clip.frame_count));
        }
        if dst.is_null() || dst_size < self.out.size {
            return Err(format!("the output buffer holds {dst_size} bytes, the frame needs {}", self.out.size));
        }
        match &mut self.pipe {
            Pipe::Split(s) => s.decode(&self.clip, &mut self.movie, &self.out, frame, dst, sync, prefetch),
            Pipe::Combined(ctx) => {
                let comp =
                    self.movie.read_compressed_frame(frame).map_err(|e| format!("reading frame {frame}: {e}"))?;
                let meta = self
                    .movie
                    .read_timed_meta(frame)
                    .map_err(|e| format!("reading the metadata of frame {frame}: {e}"))?;
                ctx.set_timed_meta(&meta).map_err(|e| format!("header pack: {e}"))?;
                let _lock = self.clip.develop_lock();
                // SAFETY: `dst` is a device buffer of the lane the context was
                // created for, holding at least the output size (checked above).
                unsafe { ctx.decode_frame_to_gpu(&comp, dst as *mut CrmVoid, dst_size) }.map_err(|e| {
                    let detail = self.clip.sdk.get_last_error_string().unwrap_or_default();
                    format!("decoding frame {frame}: {e} {detail}").trim_end().to_string()
                })?;
                sync();
                Ok(())
            }
        }
    }
}

fn combined(clip: &Clip, gpu: SdkGpu, meta0: &[u8]) -> Result<DecodeContext, String> {
    let res = clip.res_param(DataType::CrmCompressed, DataType::RgbLutApplied, BufferType::Cpu, lane_buffer(clip.lane));
    let geometry = OutputGeometry {
        crop: clip.crop,
        downscale: Downscale::X1,
        decode_level: clip.level,
        bits_per_sample: BITS,
        samples_per_pixel: SAMPLES,
    };
    DecodeContext::new(
        clip.sdk.clone(),
        clip.format_id,
        lane_device(clip.lane),
        gpu,
        ProcessType::DEVELOP | ProcessType::DECOMPRESS,
        Resolution::K8,
        clip.dev,
        res,
        clip.uuid_box.to_vec(),
        clip.craw_entry.to_vec(),
        meta0,
        SEQUENCE,
        geometry,
    )
    .map_err(|e| format!("decode context: {e}"))
}

/// The first GPU the SDK supports on `device`, in the SDK's own numbering.
fn pick_gpu(sdk: &CrmSdkLibrary, device: Device) -> SdkGpu {
    const GPUS: [SdkGpu; 8] = [
        SdkGpu::Gpu1,
        SdkGpu::Gpu2,
        SdkGpu::Gpu3,
        SdkGpu::Gpu4,
        SdkGpu::Gpu5,
        SdkGpu::Gpu6,
        SdkGpu::Gpu7,
        SdkGpu::Gpu8,
    ];
    sdk.get_gpu_info_list_with_device(device)
        .ok()
        .and_then(|list| list.iter().zip(GPUS).find(|(info, _)| info.is_supported()).map(|(_, g)| g))
        .unwrap_or(SdkGpu::Gpu1)
}

/// The container's frame rate, which is what the clip plays at; the SDK's
/// capture rate when the container has none.
fn frame_rate(movie: &CrmContainer<File>, sdk: &CrmSdkLibrary, format_id: FormatId, hdr: &[u8]) -> (u32, u32) {
    let (n, d) = movie.fps();
    if n > 0 && d > 0 {
        let g = gcd(n, d);
        return (n / g, d / g);
    }
    let fps = sdk.get_image_info(format_id, hdr, image_info::CaptureFrameRate).unwrap_or(0);
    let ntsc = sdk.get_image_info(format_id, hdr, image_info::CaptureSystemFlag).unwrap_or(false);
    match (fps, ntsc) {
        (0, _) => (24, 1),
        (f, true) => (u32::from(f) * 1000, 1001),
        (f, false) => (u32::from(f), 1),
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn clip_metadata(
    sdk: &CrmSdkLibrary,
    format_id: FormatId,
    hdr: &[u8],
    lane: i32,
    fps_num: u32,
    fps_den: u32,
) -> Vec<(CString, CString)> {
    let mut v: Vec<(String, String)> = Vec::new();
    let text = |s: String| s.trim_matches(char::from(0)).trim().to_string();
    if let Ok(s) = sdk.get_image_info(format_id, hdr, image_info::DeviceModelName) {
        v.push(("camera".into(), text(s)));
    }
    if let Ok(s) = sdk.get_image_info(format_id, hdr, image_info::DeviceSerialNumber) {
        v.push(("camera_serial".into(), text(s)));
    }
    if let Ok(s) = sdk.get_image_info(format_id, hdr, image_info::DeviceFirmwareVersion) {
        v.push(("camera_firmware".into(), text(s)));
    }
    if let Ok(s) = sdk.get_image_info(format_id, hdr, image_info::LensName) {
        v.push(("lens".into(), text(s)));
    }
    if let Ok(iso) = sdk.get_image_info(format_id, hdr, image_info::IsoSpeed) {
        v.push(("iso".into(), iso.to_string()));
    }
    if let Ok(k) = sdk.get_image_info(format_id, hdr, image_info::ColorTemperature) {
        v.push(("color_temperature".into(), k.to_string()));
    }
    if let Ok(a) = sdk.get_image_info(format_id, hdr, image_info::ShutterAngle) {
        v.push(("shutter_angle".into(), a.to_string()));
    }
    if let Ok(s) = sdk.get_image_info(format_id, hdr, image_info::CreationDateTime) {
        v.push(("creation_time".into(), text(s)));
    }
    v.push(("frame_rate".into(), format!("{fps_num}/{fps_den}")));
    v.push(("crm.gpu".into(), sdk::lane_name(lane).into()));
    v.into_iter()
        .filter(|(_, val)| !val.is_empty())
        .filter_map(|(k, val)| Some((CString::new(k).ok()?, CString::new(val).ok()?)))
        .collect()
}

pub fn gamma_from(v: i32) -> Option<Gamma> {
    use Gamma::*;
    [
        Linear, CanonLog, Rec709, WideDr, Dci, CanonLog2, CanonLog3, Cmt709, Canon709, CmtPq, CmtHlg, CmtDci, WideDr3d,
        Dci3d, Bt2100Pq, Bt2100Hlg,
    ]
    .into_iter()
    .find(|g| *g as i32 == v)
}

pub fn color_space_from(v: i32) -> Option<ColorSpace> {
    use ColorSpace::*;
    [Bt709, DciP3Plus, CinemaGamut, Bt2020, Aces, DciP3, Original].into_iter().find(|c| *c as u32 as i32 == v)
}
