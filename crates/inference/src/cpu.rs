use crate::backend::{AttnShape, Backend, Matrix, Weights};
use crate::kernel::{
    Q8_BLOCK, dot, exp, gelu_slice, layernorm_row, quantize_block, rmsnorm_row, softmax_slice,
    swiglu_slice, tile, tile_q8, weighted_sum_rows,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::kernel::{
    dot_i8, tile_q8_smmla8_8x1, tile_q8_smmla8_8x8_store, use_i8mm, weighted_sum_rows_4_hd64,
};
use crate::pool::Pool;
use std::mem::MaybeUninit;
use std::sync::Mutex;

#[derive(Default)]
struct RopeCache {
    pos0: usize,
    t: usize,
    inv_freq: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[derive(Default)]
struct Q8Scratch {
    x_packed: Vec<i8>,
    s_packed: Vec<f32>,
    x_tail: Vec<i8>,
    x_tail_scales: Vec<f32>,
}

/// Plain Rust on the CPU: NEON kernels, a spinning thread pool. The reference
/// every other backend is compared against.
pub struct CpuBackend {
    pool: Pool,
    rope_cache: Mutex<RopeCache>,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    q8_scratch: Mutex<Q8Scratch>,
}

impl CpuBackend {
    pub fn new(threads: usize) -> Self {
        Self {
            pool: Pool::new(threads),
            rope_cache: Mutex::new(RopeCache::default()),
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            q8_scratch: Mutex::new(Q8Scratch::default()),
        }
    }
}

/// Threads a decoded token uses. Decode is bound by memory, which 8 performance
/// cores already saturate while leaving headroom for OS threads; using all 10
/// or the efficiency cores adds straggler wait at every matmul barrier.
const DECODE_THREADS: usize = 8;

impl CpuBackend {
    /// Threads a pass over `t` tokens runs its regions on.
    fn threads(&self, t: usize) -> usize {
        match t {
            1 => self.pool.threads().min(DECODE_THREADS),
            _ => self.pool.threads(),
        }
    }

    /// Threads for a short elementwise or normalization pass over `bytes` of
    /// input: enough to saturate memory bandwidth (~64 KiB per thread) without
    /// waking all performance cores on tiny buffers where one OS preemption
    /// dominates the barrier.
    fn elem_threads(&self, bytes: usize) -> usize {
        (bytes / (64 << 10)).clamp(1, self.threads(1))
    }

    /// `x` (`[t][k]`) rounded to int8 in a single pass, with one scale per
    /// `Q8_BLOCK` values of each row.
    fn quantize_rows(&self, x: &[f32], t: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
        let blocks = k.div_ceil(Q8_BLOCK);
        let mut xq = vec![0i8; x.len()];
        let mut scales = vec![0.0; t * blocks];
        let quantize_range = |r0: usize, r1: usize, q_out: &mut [i8], s_out: &mut [f32]| {
            for (j, (q_row, s_row)) in q_out
                .chunks_exact_mut(k)
                .zip(s_out.chunks_exact_mut(blocks))
                .enumerate()
            {
                let row = &x[(r0 + j) * k..][..k];
                for ((dst, s), vals) in q_row
                    .chunks_mut(Q8_BLOCK)
                    .zip(s_row.iter_mut())
                    .zip(row.chunks(Q8_BLOCK))
                {
                    *s = quantize_block(vals, dst);
                }
            }
            debug_assert_eq!(r1 - r0, q_out.len() / k.max(1));
        };
        if k == 0 || t == 0 {
            return (xq, scales);
        }
        if x.len() < 8192 || t == 1 {
            quantize_range(0, t, &mut xq, &mut scales);
            return (xq, scales);
        }
        let q_buf = SharedBuf::new(&mut xq);
        let s_buf = SharedBuf::new(&mut scales);
        let threads = self.pool.threads().min(t);
        self.pool.run_threads(threads, &|tid, width| {
            let r0 = tid * t / width;
            let r1 = (tid + 1) * t / width;
            if r0 >= r1 {
                return;
            }
            // SAFETY: `r0..r1` rows are disjoint across threads.
            let q_slice = unsafe { q_buf.slice_mut(r0 * k, (r1 - r0) * k) };
            let s_slice = unsafe { s_buf.slice_mut(r0 * blocks, (r1 - r0) * blocks) };
            quantize_range(r0, r1, q_slice, s_slice);
        });
        (xq, scales)
    }
}

impl CpuBackend {
    /// `f(row, out)` for every `width`-wide row of `x`, rows in parallel, each
    /// result written directly into uninitialized output memory.
    fn map_rows(
        &self,
        x: &[f32],
        width: usize,
        f: impl Fn(&[f32], &mut [MaybeUninit<f32>]) + Sync,
    ) -> Vec<f32> {
        assert_eq!(x.len() % width, 0);
        let total = x.len();
        let rows = total / width;
        let mut out: Vec<f32> = Vec::with_capacity(total);
        if total == 0 {
            return out;
        }
        let spare = &mut out.spare_capacity_mut()[..total];
        let threads = self.elem_threads(total * 4).min(rows);
        if threads <= 1 {
            for (row, o) in x.chunks_exact(width).zip(spare.chunks_exact_mut(width)) {
                f(row, o);
            }
        } else {
            let out_buf = SharedBuf::new(spare);
            self.pool.run_threads(threads, &|tid, nw| {
                let r0 = tid * rows / nw;
                let r1 = (tid + 1) * rows / nw;
                if r0 >= r1 {
                    return;
                }
                // SAFETY: `r0..r1` rows are disjoint across threads.
                let dst = unsafe { out_buf.slice_mut(r0 * width, (r1 - r0) * width) };
                for (row, o) in x[r0 * width..r1 * width]
                    .chunks_exact(width)
                    .zip(dst.chunks_exact_mut(width))
                {
                    f(row, o);
                }
            });
        }
        // SAFETY: every row `0..rows` was initialized above.
        unsafe { out.set_len(total) };
        out
    }
}

impl Default for CpuBackend {
    /// `INFERENCE_THREADS` if set, else one thread per performance core.
    fn default() -> Self {
        let threads = std::env::var("INFERENCE_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(performance_cores)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        Self::new(threads)
    }
}

/// Performance cores, if the OS says. Efficiency cores run at half the speed,
/// and a region ends when its slowest thread does, so using them costs more
/// than they add: all 14 cores of an M3 Max decode 17% slower than 8.
// shortcut: macOS only; other systems get every core, which is wrong on
// big.LITTLE Linux, read the cpu capacity files there when that machine matters.
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            old_len: *mut usize,
            new: *mut std::ffi::c_void,
            new_len: usize,
        ) -> std::ffi::c_int;
    }
    let (mut n, mut len) = (0i32, size_of::<i32>());
    // SAFETY: the name is NUL-terminated and `old` has room for `len` bytes.
    let status = unsafe {
        sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            (&raw mut n).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && n > 0).then_some(n as usize)
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}

pub(crate) fn softmax_in_place(x: &mut [f32]) {
    softmax_slice(x);
}

/// The two sides of one matrix product, so `matmul_block` runs the same loops
/// over f32 and over int8.
trait Operands: Sync {
    /// Weight rows `r..r + R` against tokens `i..i + T`: `out[a][b]` is the dot
    /// product of weight row `r + a` with token `i + b`.
    fn tile<const R: usize, const T: usize, const U: usize>(
        &self,
        r: usize,
        i: usize,
    ) -> [[f32; T]; R];
}

struct F32Operands<'a> {
    w: &'a [f32],
    x: &'a [f32],
    k: usize,
}

impl Operands for F32Operands<'_> {
    fn tile<const R: usize, const T: usize, const U: usize>(
        &self,
        r: usize,
        i: usize,
    ) -> [[f32; T]; R] {
        let k = self.k;
        tile::<R, T, U>(&self.w[r * k..], &self.x[i * k..], k)
    }
}

/// Int8 weights with a scale per row against int8 activations with a scale per
/// `Q8_BLOCK` values: the integer dot product, scaled per block by `tile_q8`
/// and once per weight row here, is the f32 one. `w` is stored as
/// `Matrix::q8_index` says: groups of four rows, so a tile of four rows starts
/// at a multiple of four, and a single row is one of the leftovers at the end.
struct Q8Operands<'a> {
    w: &'a [i8],
    w_scales: &'a [f32],
    x: &'a [i8],
    x_scales: &'a [f32],
    k: usize,
}

impl Operands for Q8Operands<'_> {
    // `U` is f32's accumulator count per output; `tile_q8` has one per output.
    fn tile<const R: usize, const T: usize, const _U: usize>(
        &self,
        r: usize,
        i: usize,
    ) -> [[f32; T]; R] {
        let (k, blocks) = (self.k, self.k.div_ceil(Q8_BLOCK));
        debug_assert!(match R {
            4 => r.is_multiple_of(4),
            _ => r >= self.w_scales.len() / 4 * 4,
        });
        let mut out = tile_q8::<R, T>(
            &self.w[r * k..],
            &self.x[i * k..],
            &self.x_scales[i * blocks..],
            k,
        );
        for (a, row) in out.iter_mut().enumerate() {
            row.iter_mut().for_each(|v| *v *= self.w_scales[r + a]);
        }
        out
    }
}

impl CpuBackend {
    /// `ops` is a `[t][n_in]` activation matrix against `n_out` weight rows;
    /// returns `[t][n_out]` with `bias` added.
    fn product(
        &self,
        ops: &impl Operands,
        t: usize,
        n_out: usize,
        row_bytes: usize,
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        if let Some(b) = bias {
            assert_eq!(b.len(), n_out);
        }
        if n_out == 0 || t == 0 {
            return Vec::new();
        }
        // One token: `[n_out][1]` is already `[1][n_out]`, so tasks write the
        // result and bias directly into uninitialized output memory. Small
        // matrices (<= 2 MB) split statically; larger ones divide into an exact
        // multiple of `threads` (`3` tasks per thread) so no remainder task
        // stalls the barrier.
        if t == 1 {
            let mut out: Vec<f32> = Vec::with_capacity(n_out);
            let total_bytes = row_bytes * n_out;
            let groups = n_out.div_ceil(4);
            let max_by_bytes = (total_bytes / (64 << 10)).max(1);
            let threads = self.threads(t).min(groups.max(1)).min(max_by_bytes);
            let out_buf = SharedBuf::new(&mut out.spare_capacity_mut()[..n_out]);
            let run_range = |g0: usize, g1: usize| {
                let r0 = g0 * 4;
                let r1 = (g1 * 4).min(n_out);
                if r0 >= r1 {
                    return;
                }
                let mut r = r0;
                while r + 4 <= r1 {
                    let tl = ops.tile::<4, 1, 4>(r, 0);
                    let mut vals = [tl[0][0], tl[1][0], tl[2][0], tl[3][0]];
                    if let Some(b) = bias {
                        for a in 0..4 {
                            vals[a] += b[r + a];
                        }
                    }
                    // SAFETY: `r0..r1` is disjoint across tasks and inside `out`.
                    unsafe { out_buf.write_init(r, &vals) };
                    r += 4;
                }
                for r in r..r1 {
                    let v = ops.tile::<1, 1, 4>(r, 0)[0][0] + bias.map_or(0.0, |b| b[r]);
                    // SAFETY: `r0..r1` is disjoint across tasks and inside `out`.
                    unsafe { out_buf.write_init(r, &[v]) };
                }
            };
            if total_bytes <= 2 << 20 {
                self.pool.run_threads(threads, &|tid, width| {
                    run_range(tid * groups / width, (tid + 1) * groups / width);
                });
            } else {
                let n_tasks = (threads * 3).min(groups);
                self.pool.run_width(threads, n_tasks, &|task| {
                    run_range(task * groups / n_tasks, (task + 1) * groups / n_tasks);
                });
            }
            // SAFETY: every element in `0..n_out` was initialized above.
            unsafe { out.set_len(n_out) };
            return out;
        }

        // Several tokens: each task owns a slab of whole 4-row tiles (~3 tasks
        // per thread) and writes each tile straight into uninitialized output
        // memory so zero-fill and page faults happen in parallel.
        let total = t * n_out;
        let mut out: Vec<f32> = Vec::with_capacity(total);
        let out_buf = SharedBuf::new(&mut out.spare_capacity_mut()[..total]);
        let threads = self.pool.threads();
        let rows_per_task = (n_out.div_ceil(threads * 3).max(16)).div_ceil(4) * 4;
        self.pool.run(n_out.div_ceil(rows_per_task), &|c| {
            let first = c * rows_per_task;
            let end = (first + rows_per_task).min(n_out);
            let mut r = first;
            while r + 4 <= end {
                let b0 = bias.map_or(0.0, |b| b[r]);
                let b1 = bias.map_or(0.0, |b| b[r + 1]);
                let b2 = bias.map_or(0.0, |b| b[r + 2]);
                let b3 = bias.map_or(0.0, |b| b[r + 3]);
                let mut i = 0;
                while i + 4 <= t {
                    let tl = ops.tile::<4, 4, 1>(r, i);
                    for j in 0..4 {
                        let vals = [tl[0][j] + b0, tl[1][j] + b1, tl[2][j] + b2, tl[3][j] + b3];
                        // SAFETY: this task owns columns `first..end` of every row.
                        unsafe { out_buf.write_init((i + j) * n_out + r, &vals) };
                    }
                    i += 4;
                }
                for i in i..t {
                    let tl = ops.tile::<4, 1, 4>(r, i);
                    let vals = [tl[0][0] + b0, tl[1][0] + b1, tl[2][0] + b2, tl[3][0] + b3];
                    // SAFETY: this task owns columns `first..end` of every row.
                    unsafe { out_buf.write_init(i * n_out + r, &vals) };
                }
                r += 4;
            }
            for r in r..end {
                let b = bias.map_or(0.0, |b| b[r]);
                for i in 0..t {
                    let v = ops.tile::<1, 1, 4>(r, i)[0][0] + b;
                    // SAFETY: this task owns columns `first..end` of every row.
                    unsafe { out_buf.write_init(i * n_out + r, &[v]) };
                }
            }
        });
        // SAFETY: every `(i, r)` in `0..t × 0..n_out` was initialized above.
        unsafe { out.set_len(total) };
        out
    }

    /// Float32 prefill (`t > 1`) using 6×4 NEON tiles (`24` accumulators = `2.4`
    /// FMAs/load vs `2.0` for 4×4, zero spills and zero weight packing).
    #[allow(clippy::needless_range_loop)]
    fn prefill_f32(
        &self,
        w: &[f32],
        x: &[f32],
        t: usize,
        n_out: usize,
        k: usize,
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        if let Some(b) = bias {
            assert_eq!(b.len(), n_out);
        }
        if n_out == 0 || t == 0 {
            return Vec::new();
        }
        let total = t * n_out;
        let mut out: Vec<f32> = Vec::with_capacity(total);
        let out_buf = SharedBuf::new(&mut out.spare_capacity_mut()[..total]);
        let threads = self.pool.threads();
        let groups = n_out.div_ceil(6);
        let n_tasks = if groups <= threads * 20 {
            groups
        } else {
            threads * 8
        };
        self.pool.run(n_tasks, &|task| {
            let first = (task * groups / n_tasks) * 6;
            let end = ((task + 1) * groups / n_tasks * 6).min(n_out);
            let mut r = first;
            while r + 6 <= end {
                let bv: [f32; 6] = std::array::from_fn(|a| bias.map_or(0.0, |b| b[r + a]));
                let mut i = 0;
                while i + 4 <= t {
                    let tl = tile::<6, 4, 1>(&w[r * k..], &x[i * k..], k);
                    for j in 0..4 {
                        let vals: [f32; 6] = std::array::from_fn(|a| tl[a][j] + bv[a]);
                        // SAFETY: this task owns columns `first..end` of every row.
                        unsafe { out_buf.write_init((i + j) * n_out + r, &vals) };
                    }
                    i += 4;
                }
                for i in i..t {
                    let tl = tile::<6, 1, 2>(&w[r * k..], &x[i * k..], k);
                    let vals: [f32; 6] = std::array::from_fn(|a| tl[a][0] + bv[a]);
                    // SAFETY: this task owns columns `first..end` of every row.
                    unsafe { out_buf.write_init(i * n_out + r, &vals) };
                }
                r += 6;
            }
            while r + 2 <= end {
                let b0 = bias.map_or(0.0, |b| b[r]);
                let b1 = bias.map_or(0.0, |b| b[r + 1]);
                let mut i = 0;
                while i + 4 <= t {
                    let tl = tile::<2, 4, 2>(&w[r * k..], &x[i * k..], k);
                    for j in 0..4 {
                        let vals = [tl[0][j] + b0, tl[1][j] + b1];
                        unsafe { out_buf.write_init((i + j) * n_out + r, &vals) };
                    }
                    i += 4;
                }
                for i in i..t {
                    let tl = tile::<2, 1, 4>(&w[r * k..], &x[i * k..], k);
                    let vals = [tl[0][0] + b0, tl[1][0] + b1];
                    unsafe { out_buf.write_init(i * n_out + r, &vals) };
                }
                r += 2;
            }
            for r in r..end {
                let b = bias.map_or(0.0, |b| b[r]);
                let mut i = 0;
                while i + 4 <= t {
                    let tl = tile::<1, 4, 4>(&w[r * k..], &x[i * k..], k);
                    for j in 0..4 {
                        unsafe { out_buf.write_init((i + j) * n_out + r, &[tl[0][j] + b]) };
                    }
                    i += 4;
                }
                for i in i..t {
                    let v = tile::<1, 1, 4>(&w[r * k..], &x[i * k..], k)[0][0] + b;
                    unsafe { out_buf.write_init(i * n_out + r, &[v]) };
                }
            }
        });
        // SAFETY: every `(i, r)` in `0..t × 0..n_out` was initialized above.
        unsafe { out.set_len(total) };
        out
    }

    /// Int8 matmul using ARMv8.6 `smmla` 8-row weight layout (`packed_index_smmla8`):
    /// 8×1 streaming decode and 8×8 prefill with fused quantize-and-pack.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn matmul_q8_smmla8(
        &self,
        x: &[f32],
        t: usize,
        mat: &Matrix,
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let (n_out, k) = (mat.rows, mat.cols);
        let Weights::Q8 {
            values: w,
            scales: w_scales,
        } = &mat.weights
        else {
            unreachable!()
        };
        let row_bytes = mat.bytes() / n_out.max(1);
        if let Some(b) = bias {
            assert_eq!(b.len(), n_out);
        }
        if n_out == 0 || t == 0 {
            return Vec::new();
        }
        let whole8 = n_out / 8 * 8;
        let full = k / Q8_BLOCK;
        let rem = k - full * Q8_BLOCK;

        if t == 1 {
            // Quantize single token into stack buffers for normal model widths
            // (eliminating ~200 heap allocations per decoded token).
            let blocks = k.div_ceil(Q8_BLOCK);
            let mut stack_q = [0i8; 8192];
            let mut stack_s = [0.0f32; 64];
            let (heap_q, heap_s);
            let (xq, x_scales): (&[i8], &[f32]) = if k <= stack_q.len() && blocks <= stack_s.len() {
                for ((dst, s), vals) in stack_q[..k]
                    .chunks_mut(Q8_BLOCK)
                    .zip(stack_s[..blocks].iter_mut())
                    .zip(x.chunks(Q8_BLOCK))
                {
                    *s = quantize_block(vals, dst);
                }
                (&stack_q[..k], &stack_s[..blocks])
            } else {
                (heap_q, heap_s) = self.quantize_rows(x, 1, k);
                (&heap_q, &heap_s)
            };

            let mut out: Vec<f32> = Vec::with_capacity(n_out);
            let total_bytes = row_bytes * n_out;
            let groups = n_out.div_ceil(8);
            let max_by_bytes = (total_bytes / (64 << 10)).max(1);
            let threads = self.threads(1).min(groups.max(1)).min(max_by_bytes);
            let out_buf = SharedBuf::new(&mut out.spare_capacity_mut()[..n_out]);
            let run_range = |g0: usize, g1: usize| {
                let r0 = g0 * 8;
                let r1 = (g1 * 8).min(n_out);
                if r0 >= r1 {
                    return;
                }
                let mut r = r0;
                while r + 8 <= r1 {
                    // SAFETY: caller checked `use_i8mm()`, and `r` is a multiple of 8 < `whole8`.
                    let mut vals =
                        unsafe { tile_q8_smmla8_8x1(&w[r * k..], xq, x_scales, k) };
                    for a in 0..8 {
                        vals[a] *= w_scales[r + a];
                    }
                    if let Some(b) = bias {
                        for a in 0..8 {
                            vals[a] += b[r + a];
                        }
                    }
                    // SAFETY: `r0..r1` is disjoint across tasks and inside `out`.
                    unsafe { out_buf.write_init(r, &vals) };
                    r += 8;
                }
                for r in r..r1 {
                    let v = tile_q8::<1, 1>(&w[r * k..], xq, x_scales, k)[0][0] * w_scales[r]
                        + bias.map_or(0.0, |b| b[r]);
                    // SAFETY: `r0..r1` is disjoint across tasks and inside `out`.
                    unsafe { out_buf.write_init(r, &[v]) };
                }
            };
            if total_bytes <= 2 << 20 {
                self.pool.run_threads(threads, &|tid, width| {
                    run_range(tid * groups / width, (tid + 1) * groups / width);
                });
            } else {
                let n_tasks = (threads * 3).min(groups);
                self.pool.run_width(threads, n_tasks, &|task| {
                    run_range(task * groups / n_tasks, (task + 1) * groups / n_tasks);
                });
            }
            // SAFETY: every element in `0..n_out` was initialized above.
            unsafe { out.set_len(n_out) };
            return out;
        }

        // Prefill (`t > 1`): quantize and pack activations into 8-token `smmla`
        // groups in one parallel pass, reusing `q8_scratch` across matmuls.
        let t_groups = t.div_ceil(8);
        let x_need = t_groups * full * 8 * Q8_BLOCK;
        let s_need = t_groups * full * 16;
        let xt_need = if rem > 0 { t * rem } else { 0 };
        let xts_need = if rem > 0 { t } else { 0 };
        let mut scratch = self.q8_scratch.lock().unwrap();
        if scratch.x_packed.len() < x_need {
            scratch.x_packed.resize(x_need, 0);
        }
        if scratch.s_packed.len() < s_need {
            scratch.s_packed.resize(s_need, 0.0);
        }
        if scratch.x_tail.len() < xt_need {
            scratch.x_tail.resize(xt_need, 0);
        }
        if scratch.x_tail_scales.len() < xts_need {
            scratch.x_tail_scales.resize(xts_need, 0.0);
        }
        let scratch_ref = &mut *scratch;
        let x_packed = &mut scratch_ref.x_packed[..x_need];
        let s_packed = &mut scratch_ref.s_packed[..s_need];
        let x_tail = &mut scratch_ref.x_tail[..xt_need];
        let x_tail_scales = &mut scratch_ref.x_tail_scales[..xts_need];
        {
            let xp_buf = SharedBuf::new(x_packed);
            let sp_buf = SharedBuf::new(s_packed);
            let xt_buf = SharedBuf::new(x_tail);
            let xts_buf = SharedBuf::new(x_tail_scales);
            let threads = self.elem_threads(x.len() * 4).min(t_groups);
            self.pool.run_threads(threads, &|tid, width| {
                let g0 = tid * t_groups / width;
                let g1 = (tid + 1) * t_groups / width;
                let mut qbuf = [[0i8; Q8_BLOCK]; 8];
                for tg in g0..g1 {
                    let t0 = tg * 8;
                    let tn = 8.min(t - t0);
                    for b in 0..full {
                        let mut sc = [0.0f32; 8];
                        for j in 0..tn {
                            sc[j] = quantize_block(
                                &x[(t0 + j) * k + b * Q8_BLOCK..][..Q8_BLOCK],
                                &mut qbuf[j],
                            );
                        }
                        for q in &mut qbuf[tn..8] {
                            q.fill(0);
                        }
                        // SAFETY: each `tg` writes disjoint 1024-byte / 16-float slots.
                        let x_dst = unsafe {
                            xp_buf.slice_mut((tg * full + b) * 8 * Q8_BLOCK, 8 * Q8_BLOCK)
                        };
                        let s_dst = unsafe { sp_buf.slice_mut((tg * full + b) * 16, 16) };
                        for step in 0..(Q8_BLOCK / 8) {
                            let off = step * 8;
                            for p in 0..4 {
                                let base = (step * 4 + p) * 16;
                                x_dst[base..base + 8].copy_from_slice(&qbuf[p * 2][off..off + 8]);
                                x_dst[base + 8..base + 16]
                                    .copy_from_slice(&qbuf[p * 2 + 1][off..off + 8]);
                            }
                        }
                        for p in 0..4 {
                            let (s0, s1) = (sc[p * 2], sc[p * 2 + 1]);
                            s_dst[p * 4..p * 4 + 4].copy_from_slice(&[s0, s1, s0, s1]);
                        }
                    }
                    if rem > 0 {
                        let at = full * Q8_BLOCK;
                        for j in 0..tn {
                            let ti = t0 + j;
                            // SAFETY: each `ti` belongs to a single `tg`.
                            let dst = unsafe { xt_buf.slice_mut(ti * rem, rem) };
                            let s = quantize_block(&x[ti * k + at..][..rem], dst);
                            unsafe { xts_buf.write(ti, &[s]) };
                        }
                    }
                }
            });
        }

        let x_packed = &*x_packed;
        let s_packed = &*s_packed;
        let x_tail = &*x_tail;
        let x_tail_scales = &*x_tail_scales;
        let total = t * n_out;
        let mut out: Vec<f32> = Vec::with_capacity(total);
        let out_buf = SharedBuf::new(&mut out.spare_capacity_mut()[..total]);
        let threads = self.pool.threads();
        let row_groups = whole8 / 8;
        if row_groups > 0 {
            let n_tasks = if row_groups <= threads * 16 {
                row_groups
            } else {
                threads * 4
            };
            self.pool.run(n_tasks, &|task| {
                let g0 = task * row_groups / n_tasks;
                let g1 = (task + 1) * row_groups / n_tasks;
                for rg in g0..g1 {
                    let r_base = rg * 8;
                    let w_blk = &w[r_base * k..][..8 * k];
                    let ws: [f32; 8] = std::array::from_fn(|a| w_scales[r_base + a]);
                    let bv: [f32; 8] =
                        std::array::from_fn(|a| bias.map_or(0.0, |b| b[r_base + a]));
                    for tg in 0..t_groups {
                        let t0 = tg * 8;
                        let tn = 8.min(t - t0);
                        let tail = if rem > 0 {
                            Some((&x_tail[t0 * rem..], &x_tail_scales[t0..], rem))
                        } else {
                            None
                        };
                        // SAFETY: caller checked `use_i8mm()`, and this task owns
                        // disjoint columns `r_base..r_base + 8` of every token row.
                        unsafe {
                            tile_q8_smmla8_8x8_store(
                                w_blk,
                                &x_packed[tg * full * 8 * Q8_BLOCK..],
                                &s_packed[tg * full * 16..],
                                full,
                                &ws,
                                &bv,
                                tail,
                                out_buf.ptr_at_init(t0 * n_out + r_base),
                                n_out,
                                tn,
                            );
                        }
                    }
                }
            });
        }
        // Leftover `< 8` plain weight rows at the end (when `n_out % 8 != 0`).
        for r in whole8..n_out {
            let ws = w_scales[r];
            let bv = bias.map_or(0.0, |b| b[r]);
            let w_row = &w[r * k..][..k];
            for i in 0..t {
                let (tg, j) = (i / 8, i % 8);
                let (p, in_p) = (j / 2, j % 2);
                let mut acc = 0.0f32;
                for b in 0..full {
                    let mut sum = 0i32;
                    let xb = (tg * full + b) * 8 * Q8_BLOCK;
                    for step in 0..(Q8_BLOCK / 8) {
                        let x_off = xb + (step * 4 + p) * 16 + in_p * 8;
                        let w_off = b * Q8_BLOCK + step * 8;
                        sum += dot_i8(&w_row[w_off..w_off + 8], &x_packed[x_off..x_off + 8]);
                    }
                    acc += sum as f32 * s_packed[(tg * full + b) * 16 + p * 4 + in_p];
                }
                if rem > 0 {
                    let sum = dot_i8(&w_row[full * Q8_BLOCK..], &x_tail[i * rem..][..rem]);
                    acc += sum as f32 * x_tail_scales[i];
                }
                // SAFETY: columns `whole8..n_out` are written exclusively here.
                unsafe { out_buf.write_init(i * n_out + r, &[acc * ws + bv]) };
            }
        }
        // SAFETY: every `(i, r)` in `0..t × 0..n_out` was initialized above.
        unsafe { out.set_len(total) };
        out
    }
}

/// A buffer that tasks write disjoint pieces of at once, where the pieces may
/// not be contiguous (column strips of each row of a matrix) or need multiple
/// output buffers side by side in one `run_threads` region.
struct SharedBuf<'a, T> {
    ptr: *mut T,
    len: usize,
    buffer: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: the pointer is only accessed via `write`, `write_init`, and
// `slice_mut`, whose callers promise no two threads access overlapping elements.
unsafe impl<T: Send> Sync for SharedBuf<'_, T> {}

impl<'a, T: Copy> SharedBuf<'a, T> {
    fn new(buffer: &'a mut [T]) -> Self {
        Self {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len(),
            buffer: std::marker::PhantomData,
        }
    }

    /// Copies `values` to the buffer starting at `at`.
    ///
    /// # Safety
    /// No other thread may write or read any of those elements while this runs.
    unsafe fn write(&self, at: usize, values: &[T]) {
        assert!(at + values.len() <= self.len);
        // SAFETY: in bounds by the assert, and `values` cannot overlap the
        // buffer, which `SharedBuf` borrows exclusively; exclusivity of the
        // destination is the caller's promise.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), self.ptr.add(at), values.len()) };
    }

    /// Exclusive slice over `at..at + len`.
    ///
    /// # Safety
    /// No other thread may access `at..at + len` while the returned slice lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut(&self, at: usize, len: usize) -> &mut [T] {
        assert!(at + len <= self.len);
        // SAFETY: in bounds by the assert, exclusively borrowed via `SharedBuf`,
        // and disjoint across threads by the caller's promise.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(at), len) }
    }

    /// Raw pointer to element `at` for strided SIMD writes across rows.
    ///
    /// # Safety
    /// Caller must stay in bounds and ensure disjointness across threads.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe fn ptr_at(&self, at: usize) -> *mut T {
        debug_assert!(at < self.len);
        // SAFETY: in bounds by debug_assert, provenance comes from the whole
        // underlying buffer rather than a sub-slice.
        unsafe { self.ptr.add(at) }
    }
}

impl<'a, T: Copy> SharedBuf<'a, MaybeUninit<T>> {
    /// Initializes `at..at + values.len()` in an uninitialized buffer.
    ///
    /// # Safety
    /// No other thread may access `at..at + values.len()` while this runs.
    unsafe fn write_init(&self, at: usize, values: &[T]) {
        assert!(at + values.len() <= self.len);
        // SAFETY: in bounds, non-overlapping, and `MaybeUninit<T>` has identical
        // layout to `T`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.ptr.add(at).cast::<T>(),
                values.len(),
            )
        };
    }

    /// Raw pointer to element `at` as `*mut T` for direct SIMD stores.
    ///
    /// # Safety
    /// Caller must stay in bounds and ensure disjointness across threads.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe fn ptr_at_init(&self, at: usize) -> *mut T {
        debug_assert!(at < self.len);
        unsafe { self.ptr.add(at).cast::<T>() }
    }
}

impl Backend for CpuBackend {
    fn matmul(&self, x: &[f32], t: usize, w: &Matrix, bias: Option<&[f32]>) -> Vec<f32> {
        let (n_out, k) = (w.rows, w.cols);
        assert_eq!(x.len(), t * k);
        match &w.weights {
            Weights::F32(data) => {
                if t > 1 {
                    return self.prefill_f32(data, x, t, n_out, k, bias);
                }
                self.product(
                    &F32Operands { w: data, x, k },
                    t,
                    n_out,
                    w.bytes() / n_out.max(1),
                    bias,
                )
            }
            Weights::Q8 { values, scales } => {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                if use_i8mm() {
                    return self.matmul_q8_smmla8(x, t, w, bias);
                }
                // Quantize each token's activations once for the whole product.
                let (xq, x_scales) = self.quantize_rows(x, t, k);
                let ops = Q8Operands {
                    w: values,
                    w_scales: scales,
                    x: &xq,
                    x_scales: &x_scales,
                    k,
                };
                self.product(&ops, t, n_out, w.bytes() / n_out.max(1), bias)
            }
        }
    }

    fn layernorm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let width = gamma.len();
        assert_eq!(beta.len(), width);
        self.map_rows(x, width, |row, out| {
            layernorm_row(row, gamma, beta, eps, out);
        })
    }

    fn rmsnorm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let width = gamma.len();
        self.map_rows(x, width, |row, out| {
            rmsnorm_row(row, gamma, eps, out);
        })
    }

    fn swiglu(&self, gate_up: &[f32], width: usize) -> Vec<f32> {
        assert_eq!(gate_up.len() % (2 * width), 0);
        let out_len = gate_up.len() / 2;
        let rows = out_len / width;
        let mut out: Vec<f32> = Vec::with_capacity(out_len);
        if out_len == 0 {
            return out;
        }
        let spare = &mut out.spare_capacity_mut()[..out_len];
        // SwiGLU is exp-heavy (~2.5 µs for 4864 elements), so decode benefits
        // from 4 threads and prefill scales across all performance cores.
        let threads = (gate_up.len() * 4 / (8 << 10)).clamp(1, self.threads(rows));
        if threads <= 1 {
            for r in 0..rows {
                let first = r * 2 * width;
                swiglu_slice(
                    &gate_up[first..first + width],
                    &gate_up[first + width..first + 2 * width],
                    &mut spare[r * width..(r + 1) * width],
                );
            }
        } else if rows >= threads {
            let out_buf = SharedBuf::new(spare);
            self.pool.run_threads(threads, &|tid, nw| {
                let r0 = tid * rows / nw;
                let r1 = (tid + 1) * rows / nw;
                for r in r0..r1 {
                    let first = r * 2 * width;
                    // SAFETY: `r0..r1` rows are disjoint across threads.
                    let dst = unsafe { out_buf.slice_mut(r * width, width) };
                    swiglu_slice(
                        &gate_up[first..first + width],
                        &gate_up[first + width..first + 2 * width],
                        dst,
                    );
                }
            });
        } else {
            // Few rows (e.g. decode if `width >= 8192`): split flat output across threads.
            let out_buf = SharedBuf::new(spare);
            self.pool.run_threads(threads, &|tid, nw| {
                let start = (tid * out_len / nw) & !15;
                let end = if tid + 1 == nw {
                    out_len
                } else {
                    ((tid + 1) * out_len / nw) & !15
                };
                if start >= end {
                    return;
                }
                let mut at = start;
                // SAFETY: `start..end` is disjoint across threads.
                let mut rest = unsafe { out_buf.slice_mut(start, end - start) };
                while !rest.is_empty() {
                    let (row, j) = (at / width, at % width);
                    let run_len = rest.len().min(width - j);
                    let (run, tail) = rest.split_at_mut(run_len);
                    let first = row * 2 * width + j;
                    swiglu_slice(
                        &gate_up[first..][..run_len],
                        &gate_up[first + width..][..run_len],
                        run,
                    );
                    at += run_len;
                    rest = tail;
                }
            });
        }
        // SAFETY: every element in `0..out_len` was written above.
        unsafe { out.set_len(out_len) };
        out
    }

    fn rope(&self, x: &mut [f32], pos0: usize, n_heads: usize, head_dim: usize, inv_freq: &[f32]) {
        let half = head_dim / 2;
        assert_eq!(inv_freq.len(), half);
        let width = n_heads * head_dim;
        assert_eq!(x.len() % width, 0);
        let t = x.len() / width;
        if t == 0 {
            return;
        }
        let mut cache = self.rope_cache.lock().unwrap();
        let hit = cache.inv_freq == inv_freq
            && pos0 >= cache.pos0
            && pos0 + t <= cache.pos0 + cache.t;
        if !hit {
            cache.pos0 = pos0;
            cache.t = t;
            cache.inv_freq.clear();
            cache.inv_freq.extend_from_slice(inv_freq);
            cache.cos.resize(t * half, 0.0);
            cache.sin.resize(t * half, 0.0);
            for i in 0..t {
                let pos = (pos0 + i) as f32;
                let base = i * half;
                for (j, &f) in inv_freq.iter().enumerate() {
                    let (s, c) = (pos * f).sin_cos();
                    cache.sin[base + j] = s;
                    cache.cos[base + j] = c;
                }
            }
        }
        let off = (pos0 - cache.pos0) * half;
        let cos_tbl = &cache.cos[off..off + t * half];
        let sin_tbl = &cache.sin[off..off + t * half];

        let apply_range = |r0: usize, rows_slice: &mut [f32]| {
            for (i, row) in rows_slice.chunks_exact_mut(width).enumerate() {
                let base = (r0 + i) * half;
                let cs = &cos_tbl[base..base + half];
                let sn = &sin_tbl[base..base + half];
                for head in row.chunks_exact_mut(head_dim) {
                    let (lo, hi) = head.split_at_mut(half);
                    for j in 0..half {
                        let (a, b, c, s) = (lo[j], hi[j], cs[j], sn[j]);
                        lo[j] = a * c - b * s;
                        hi[j] = b * c + a * s;
                    }
                }
            }
        };

        let threads = self.elem_threads(x.len() * 4).min(t);
        if threads <= 1 {
            apply_range(0, x);
        } else {
            let x_buf = SharedBuf::new(x);
            self.pool.run_threads(threads, &|tid, nw| {
                let r0 = tid * t / nw;
                let r1 = (tid + 1) * t / nw;
                if r0 >= r1 {
                    return;
                }
                // SAFETY: `r0..r1` rows are disjoint across threads.
                let slice = unsafe { x_buf.slice_mut(r0 * width, (r1 - r0) * width) };
                apply_range(r0, slice);
            });
        }
    }

    fn gelu(&self, x: &mut [f32]) {
        let bytes = x.len() * 4;
        let max_t = if bytes >= (512 << 10) {
            self.pool.threads()
        } else {
            self.threads(1)
        };
        let threads = (bytes / (64 << 10)).clamp(1, max_t);
        if threads <= 1 {
            gelu_slice(x);
            return;
        }
        let n = x.len();
        let x_buf = SharedBuf::new(x);
        self.pool.run_threads(threads, &|tid, nw| {
            let start = (tid * n / nw) & !15;
            let end = if tid + 1 == nw {
                n
            } else {
                ((tid + 1) * n / nw) & !15
            };
            if start >= end {
                return;
            }
            // SAFETY: `start..end` is disjoint across threads.
            gelu_slice(unsafe { x_buf.slice_mut(start, end - start) });
        });
    }

    fn softmax(&self, x: &mut [f32]) {
        softmax_in_place(x);
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], shape: AttnShape) -> Vec<f32> {
        let AttnShape {
            t,
            pos0,
            n_heads,
            n_kv_heads,
            head_dim,
        } = shape;
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
        let n_keys_total = pos0 + t;
        assert_eq!(q.len(), t * q_dim);
        assert_eq!(k.len(), n_keys_total * kv_dim);
        assert_eq!(v.len(), k.len());
        assert_eq!(n_heads % n_kv_heads, 0);
        let group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0; t * q_dim];

        // Decode (`t == 1`): group all `group` query heads sharing one KV head so
        // each cached key and value vector is read once instead of `group` times.
        // At long contexts (`n_keys >= 512`) split keys across threads and merge.
        if t == 1 {
            let n_keys = n_keys_total;
            let out_buf = SharedBuf::new(&mut out);
            let max_threads = self.threads(1);
            let splits = if n_keys >= 512 && n_kv_heads < max_threads {
                (max_threads / n_kv_heads).max(1)
            } else {
                1
            };
            if splits == 1 {
                // Short context (`n_keys < 512`): KV fits in L1/L2 cache, so
                // parallelize across all `n_heads` query heads (zero heap
                // allocation via stack `scores` and register-resident `P @ V`).
                let threads = max_threads.min(n_heads);
                self.pool.run_threads(threads, &|tid, nw| {
                    let h0 = tid * n_heads / nw;
                    let h1 = (tid + 1) * n_heads / nw;
                    let mut heap_scores = Vec::new();
                    let mut stack_scores = [0.0f32; 512];
                    for h in h0..h1 {
                        let kv_off = (h / group) * head_dim;
                        let qv = &q[h * head_dim..][..head_dim];
                        let scores = if n_keys <= stack_scores.len() {
                            &mut stack_scores[..n_keys]
                        } else {
                            heap_scores.resize(n_keys, 0.0);
                            &mut heap_scores[..n_keys]
                        };
                        for (j, s) in scores.iter_mut().enumerate() {
                            *s = dot(qv, &k[j * kv_dim + kv_off..][..head_dim]) * scale;
                        }
                        softmax_slice(scores);
                        // SAFETY: each head `h` owns a disjoint slice of `out`.
                        let dst = unsafe { out_buf.slice_mut(h * head_dim, head_dim) };
                        weighted_sum_rows(scores, &v[kv_off..], kv_dim, dst);
                    }
                });
            } else {
                // FlashDecoding split-K merge for long contexts (`n_keys >= 512`).
                let width = n_kv_heads * splits;
                let mut partial_o = vec![0.0f32; width * group * head_dim];
                let mut partial_m = vec![f32::NEG_INFINITY; width * group];
                let mut partial_l = vec![0.0f32; width * group];
                let po_buf = SharedBuf::new(&mut partial_o);
                let pm_buf = SharedBuf::new(&mut partial_m);
                let pl_buf = SharedBuf::new(&mut partial_l);
                self.pool.run_threads(width, &|tid, _| {
                    let (kvh, sp) = (tid / splits, tid % splits);
                    let j0 = sp * n_keys / splits;
                    let j1 = (sp + 1) * n_keys / splits;
                    let len = j1 - j0;
                    let kv_off = kvh * head_dim;
                    let q_base = kvh * group;
                    let q_grp = &q[q_base * head_dim..][..group * head_dim];
                    let mut scores = vec![0.0f32; group * len];
                    for j in 0..len {
                        let kj = &k[(j0 + j) * kv_dim + kv_off..][..head_dim];
                        let mut g = 0;
                        while g + 4 <= group {
                            let tl = tile::<4, 1, 4>(&q_grp[g * head_dim..], kj, head_dim);
                            for a in 0..4 {
                                scores[(g + a) * len + j] = tl[a][0] * scale;
                            }
                            g += 4;
                        }
                        for g in g..group {
                            scores[g * len + j] =
                                dot(&q_grp[g * head_dim..][..head_dim], kj) * scale;
                        }
                    }
                    for g in 0..group {
                        let sc = &mut scores[g * len..][..len];
                        let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut l = 0.0f32;
                        for val in sc.iter_mut() {
                            let e = exp(*val - m);
                            *val = e;
                            l += e;
                        }
                        // SAFETY: `tid * group + g` is unique per thread.
                        unsafe {
                            pm_buf.write(tid * group + g, &[m]);
                            pl_buf.write(tid * group + g, &[l]);
                        }
                    }
                    // SAFETY: `tid` owns its slice of `partial_o`.
                    let po = unsafe { po_buf.slice_mut(tid * group * head_dim, group * head_dim) };
                    for j in 0..len {
                        let vj = &v[(j0 + j) * kv_dim + kv_off..][..head_dim];
                        for g in 0..group {
                            let p = scores[g * len + j];
                            let o = &mut po[g * head_dim..][..head_dim];
                            for (acc, &val) in o.iter_mut().zip(vj) {
                                *acc = p.mul_add(val, *acc);
                            }
                        }
                    }
                });
                for kvh in 0..n_kv_heads {
                    for g in 0..group {
                        let h = kvh * group + g;
                        let mut m_all = f32::NEG_INFINITY;
                        for sp in 0..splits {
                            m_all = m_all.max(partial_m[(kvh * splits + sp) * group + g]);
                        }
                        let mut l_all = 0.0f32;
                        let dst = &mut out[h * head_dim..][..head_dim];
                        for sp in 0..splits {
                            let tid = kvh * splits + sp;
                            let w = exp(partial_m[tid * group + g] - m_all);
                            l_all += w * partial_l[tid * group + g];
                            let src = &partial_o[(tid * group + g) * head_dim..][..head_dim];
                            for (d, &s) in dst.iter_mut().zip(src) {
                                *d = w.mul_add(s, *d);
                            }
                        }
                        let inv = 1.0 / l_all;
                        for d in dst.iter_mut() {
                            *d *= inv;
                        }
                    }
                }
            }
            return out;
        }

        // Prefill (`t > 1`): repack K and V once into contiguous head-major
        // `[n_kv_heads][n_keys_total][head_dim]` with sequential writes, and
        // tile 4 queries × 4 keys.
        let kv_stride = n_keys_total * head_dim;
        let mut k_hm = Vec::with_capacity(k.len());
        let mut v_hm = Vec::with_capacity(v.len());
        for kvh in 0..n_kv_heads {
            let off = kvh * head_dim;
            for pos in 0..n_keys_total {
                let src = pos * kv_dim + off;
                k_hm.extend_from_slice(&k[src..src + head_dim]);
                v_hm.extend_from_slice(&v[src..src + head_dim]);
            }
        }

        // Causal triangle load balancing: split each head into two query ranges
        // with roughly equal key counts (0..~0.7*t and ~0.7*t..t) when `t >= 16`.
        let halves = if t >= 16 { 2 } else { 1 };
        let split = (t * 7 / 10) / 4 * 4;
        let n_tasks = n_heads * halves;
        let out_buf = SharedBuf::new(&mut out);
        // Cap prefill attention at `DECODE_THREADS` (8) with dynamic claiming to
        // avoid 10-core OS stragglers on short ~50 µs attention regions.
        let threads = self.threads(1).min(n_tasks);
        self.pool.run_width(threads, n_tasks, &|task| {
            let h = task / halves;
            let half = task % halves;
            let (i_start, i_end) = if halves == 1 {
                (0, t)
            } else if half == 0 {
                (0, split)
            } else {
                (split, t)
            };
            let len_q = i_end - i_start;
            if len_q == 0 {
                return;
            }
            let mut q_contig = vec![0.0f32; len_q * head_dim];
            let mut scores = vec![0.0f32; 4 * n_keys_total];
            for j in 0..len_q {
                q_contig[j * head_dim..][..head_dim]
                    .copy_from_slice(&q[(i_start + j) * q_dim + h * head_dim..][..head_dim]);
            }
            let kvh = h / group;
            let ks = &k_hm[kvh * kv_stride..][..kv_stride];
            let vs = &v_hm[kvh * kv_stride..][..kv_stride];

            let mut qi = 0;
            while qi + 4 <= len_q {
                let pos_max = pos0 + i_start + qi + 4;
                let mut kj = 0;
                while kj + 4 <= pos_max {
                    let tl = tile::<4, 4, 1>(
                        &q_contig[qi * head_dim..],
                        &ks[kj * head_dim..],
                        head_dim,
                    );
                    for a in 0..4 {
                        for b in 0..4 {
                            scores[a * n_keys_total + kj + b] = tl[a][b] * scale;
                        }
                    }
                    kj += 4;
                }
                for b in kj..pos_max {
                    let tl = tile::<4, 1, 4>(
                        &q_contig[qi * head_dim..],
                        &ks[b * head_dim..],
                        head_dim,
                    );
                    for a in 0..4 {
                        scores[a * n_keys_total + b] = tl[a][0] * scale;
                    }
                }
                for a in 0..4 {
                    let n_keys = pos0 + i_start + qi + a + 1;
                    let sc = &mut scores[a * n_keys_total..][..n_keys];
                    softmax_slice(sc);
                    scores[a * n_keys_total + n_keys..a * n_keys_total + pos_max].fill(0.0);
                }
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                if head_dim == 64 {
                    // SAFETY: queries `i_start + qi .. + 4` of head `h` are unique to this task.
                    let base = unsafe { out_buf.ptr_at((i_start + qi) * q_dim + h * head_dim) };
                    unsafe {
                        weighted_sum_rows_4_hd64(&scores, n_keys_total, pos_max, vs, base, q_dim);
                    }
                    qi += 4;
                    continue;
                }
                for a in 0..4 {
                    let n_keys = pos0 + i_start + qi + a + 1;
                    let sc = &scores[a * n_keys_total..][..n_keys];
                    // SAFETY: `(i_start + qi + a, h)` is unique to this task.
                    let dst = unsafe {
                        out_buf.slice_mut(
                            (i_start + qi + a) * q_dim + h * head_dim,
                            head_dim,
                        )
                    };
                    weighted_sum_rows(sc, vs, head_dim, dst);
                }
                qi += 4;
            }
            for a in qi..len_q {
                let n_keys = pos0 + i_start + a + 1;
                let qv = &q_contig[a * head_dim..][..head_dim];
                let mut kj = 0;
                while kj + 4 <= n_keys {
                    let tl = tile::<4, 1, 4>(&ks[kj * head_dim..], qv, head_dim);
                    for b in 0..4 {
                        scores[kj + b] = tl[b][0] * scale;
                    }
                    kj += 4;
                }
                for b in kj..n_keys {
                    scores[b] = dot(qv, &ks[b * head_dim..][..head_dim]) * scale;
                }
                let sc = &mut scores[..n_keys];
                softmax_slice(sc);
                // SAFETY: `(i_start + a, h)` is unique to this task.
                let dst = unsafe {
                    out_buf.slice_mut((i_start + a) * q_dim + h * head_dim, head_dim)
                };
                weighted_sum_rows(sc, vs, head_dim, dst);
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;
    use crate::kernel::{q8_scale, quantize_into};
    use crate::test_data::data;

    fn cpu() -> CpuBackend {
        CpuBackend::new(3)
    }

    #[test]
    fn matmul_toy() {
        // x [2,3] @ W^T, W [2,3] -> [2,2]
        let x = [1., 2., 3., 4., 5., 6.];
        let w = Matrix::new(2, 3, vec![1., 0., 1., 0., 1., 0.]);
        let got = cpu().matmul(&x, 2, &w, Some(&[10., 20.]));
        assert_eq!(got, [14., 22., 20., 25.]);
    }

    #[test]
    fn matmul_matches_naive_loops() {
        // sizes hit: a k tail, a partial chunk of rows, rows % 4 != 0, t % 4 != 0, t >= 4
        for (t, n_in, n_out) in [
            (1, 19, 37),
            (5, 19, 37),
            (3, 64, 16),
            (1, 5, 1),
            (9, 40, 18),
            (8, 768, 33),
        ] {
            let x = data(t * n_in, 1);
            let w = Matrix::new(n_out, n_in, data(n_out * n_in, 2));
            let bias = data(n_out, 3);
            let got = cpu().matmul(&x, t, &w, Some(&bias));
            let mut want = vec![0.0f32; t * n_out];
            for tt in 0..t {
                for o in 0..n_out {
                    let mut s = bias[o] as f64;
                    for i in 0..n_in {
                        s += x[tt * n_in + i] as f64 * w.f32()[o * n_in + i] as f64;
                    }
                    want[tt * n_out + o] = s as f32;
                }
            }
            assert_close(&got, &want, 1e-5, 1e-5);
        }
    }

    #[test]
    fn layernorm_toy() {
        // mean 2.5, variance 1.25, so (x - 2.5) / sqrt(1.25)
        let got = cpu().layernorm(&[1., 2., 3., 4.], &[1.; 4], &[0.; 4], 0.0);
        assert_close(
            &got,
            &[-1.3416408, -0.4472136, 0.4472136, 1.3416408],
            1e-6,
            0.0,
        );
        // gamma scales and beta shifts; each row is normalized on its own, so
        // [1, 3] and [10, 14] (different mean and spread) land on the same output
        let got = cpu().layernorm(&[1., 3., 10., 14.], &[2., 2.], &[1., 1.], 0.0);
        assert_close(&got, &[-1., 3., -1., 3.], 1e-6, 0.0);
    }

    // The int8 product is exact integer arithmetic around the rounding, so it
    // must equal the f32 product of the same rounded values: a wrong block
    // boundary, scale index or tail shows up here, not as a vague accuracy loss.
    #[test]
    fn int8_matmul_equals_f32_matmul_on_the_dequantized_operands() {
        // two whole blocks and a partial one
        let (n_out, n_in) = (37, 2 * Q8_BLOCK + 44);
        let mut q = Matrix::new(n_out, n_in, data(n_out * n_in, 4));
        q.quantize();
        let Weights::Q8 { values, scales } = &q.weights else {
            unreachable!()
        };
        let dequantized_w: Vec<f32> = (0..n_out * n_in)
            .map(|i| values[q.q8_index(i / n_in, i % n_in)] as f32 * scales[i / n_in])
            .collect();
        let dequantized_w = Matrix::new(n_out, n_in, dequantized_w);
        let bias = data(n_out, 6);
        // 1 token takes the decode path; 2 and 9 the prefill one, with a token tail;
        // 40 is enough values to split the quantization into several tasks
        for t in [1, 2, 9, 40] {
            let x = data(t * n_in, 8);
            let mut dequantized_x = Vec::new();
            // blocks never straddle two tokens: each token ends in a partial one
            for block in x.chunks_exact(n_in).flat_map(|row| row.chunks(Q8_BLOCK)) {
                let scale = q8_scale(block);
                let mut q = vec![0i8; block.len()];
                quantize_into(block, scale, &mut q);
                dequantized_x.extend(q.iter().map(|&v| v as f32 * scale));
            }
            let got = cpu().matmul(&x, t, &q, Some(&bias));
            let want = cpu().matmul(&dequantized_x, t, &dequantized_w, Some(&bias));
            assert_close(&got, &want, 1e-5, 1e-4);
        }
    }

    #[test]
    fn rmsnorm_toy() {
        // mean square 12.5, so every value is divided by sqrt(12.5) = 3.5355339
        let got = cpu().rmsnorm(&[3., 4.], &[1., 2.], 0.0);
        assert_close(&got, &[0.8485281, 2.2627417], 1e-6, 0.0);
    }

    #[test]
    fn swiglu_toy() {
        // one row, width 2: silu(1) * 2 and silu(-1) * 3
        let got = cpu().swiglu(&[1., -1., 2., 3.], 2);
        assert_close(&got, &[1.4621172, -0.8068243], 1e-6, 0.0);
        // more values than one task takes (8192), so tasks start mid-row and cross row ends
        let (rows, width) = (30, 700);
        let x = data(rows * 2 * width, 5);
        let got = cpu().swiglu(&x, width);
        for r in 0..rows {
            for j in 0..width {
                let (g, u) = (x[r * 2 * width + j], x[r * 2 * width + width + j]);
                assert!((got[r * width + j] - g / (1.0 + (-g).exp()) * u).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rope_rotates_half_pairs() {
        // head_dim 4: elements 0 and 2 pair, 1 and 3 pair. At position 1 a
        // quarter turn (pi / 2) sends (1, 0) to (0, 1) and leaves the other
        // pair, whose frequency is 0, alone.
        let mut x = [1., 5., 0., 7.];
        cpu().rope(&mut x, 1, 1, 4, &[std::f32::consts::FRAC_PI_2, 0.0]);
        assert_close(&x, &[0., 5., 1., 7.], 1e-6, 0.0);
    }

    // Enough rows to be several tasks: each must start at its own position.
    #[test]
    fn rope_and_norms_over_many_rows_equal_row_by_row() {
        let (n_heads, head_dim, rows) = (2, 4, 3000);
        let width = n_heads * head_dim;
        let inv_freq = [1.0, 0.01];
        let x = data(rows * width, 11);
        let mut all = x.clone();
        cpu().rope(&mut all, 7, n_heads, head_dim, &inv_freq);
        let gamma = data(width, 12);
        let beta = data(width, 13);
        let (ln, rms) = (
            cpu().layernorm(&x, &gamma, &beta, 1e-5),
            cpu().rmsnorm(&x, &gamma, 1e-5),
        );
        for i in 0..rows {
            let row = &x[i * width..(i + 1) * width];
            let mut one = row.to_vec();
            cpu().rope(&mut one, 7 + i, n_heads, head_dim, &inv_freq);
            assert_close(&all[i * width..(i + 1) * width], &one, 1e-6, 1e-6);
            let at = i * width..(i + 1) * width;
            assert_close(
                &ln[at.clone()],
                &cpu().layernorm(row, &gamma, &beta, 1e-5),
                1e-6,
                1e-6,
            );
            assert_close(&rms[at], &cpu().rmsnorm(row, &gamma, 1e-5), 1e-6, 1e-6);
        }
    }

    #[test]
    fn rope_position_zero_is_identity_and_offset_matches() {
        let (n_heads, head_dim) = (3, 8);
        let inv_freq: Vec<f32> = (0..4).map(|j| 10000f32.powf(-(j as f32) / 4.0)).collect();
        let x = data(4 * n_heads * head_dim, 9);
        let mut all = x.clone();
        cpu().rope(&mut all, 0, n_heads, head_dim, &inv_freq);
        assert_close(
            &all[..n_heads * head_dim],
            &x[..n_heads * head_dim],
            1e-7,
            0.0,
        );
        // The property the KV cache relies on: row 3 rotated alone at pos0 = 3
        // equals row 3 of the full pass.
        let row = n_heads * head_dim;
        let mut last = x[3 * row..].to_vec();
        cpu().rope(&mut last, 3, n_heads, head_dim, &inv_freq);
        assert_close(&last, &all[3 * row..], 1e-7, 0.0);
        // A rotation does not change length.
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((norm(&all) - norm(&x)).abs() < 1e-4);
    }

    #[test]
    fn gelu_known_values() {
        let mut x = [0.0, 1.0, -1.0, 10.0, -10.0];
        cpu().gelu(&mut x);
        assert_close(&x, &[0.0, 0.841192, -0.158808, 10.0, 0.0], 1e-5, 0.0);
    }

    #[test]
    fn softmax_is_stable_and_sums_to_one() {
        let mut x = [1000.0, 1000.0, 0.0];
        cpu().softmax(&mut x);
        assert_close(&x, &[0.5, 0.5, 0.0], 1e-6, 0.0);
        let mut y = data(10, 7);
        cpu().softmax(&mut y);
        assert!((y.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    fn shape(t: usize, pos0: usize, n_heads: usize, n_kv_heads: usize) -> AttnShape {
        AttnShape {
            t,
            pos0,
            n_heads,
            n_kv_heads,
            head_dim: 4,
        }
    }

    #[test]
    fn attention_first_token_returns_its_own_value() {
        // Query 0 sees only key 0, so softmax is [1] and the output is v[0].
        let s = shape(3, 0, 2, 2);
        let (q, k, v) = (data(3 * 8, 1), data(3 * 8, 2), data(3 * 8, 3));
        let got = cpu().attention(&q, &k, &v, s);
        assert_close(&got[..8], &v[..8], 1e-6, 0.0);
    }

    #[test]
    fn attention_decode_equals_last_row_of_prefill() {
        // The property the KV cache relies on: one query at pos0 = 4 against
        // cached keys gives the same row as the full 5-token pass.
        for (n_heads, n_kv_heads) in [(2, 2), (4, 2), (4, 1)] {
            let (q_dim, kv_dim) = (n_heads * 4, n_kv_heads * 4);
            let (q, k, v) = (data(5 * q_dim, 1), data(5 * kv_dim, 2), data(5 * kv_dim, 3));
            let full = cpu().attention(&q, &k, &v, shape(5, 0, n_heads, n_kv_heads));
            let last = cpu().attention(&q[4 * q_dim..], &k, &v, shape(1, 4, n_heads, n_kv_heads));
            assert_close(&last, &full[4 * q_dim..], 1e-6, 0.0);
        }
    }

    #[test]
    fn attention_grouped_heads_share_one_kv_head() {
        // Two query heads with identical q and one kv head must give identical outputs.
        let q1 = data(4, 1);
        let q: Vec<f32> = q1.iter().chain(&q1).copied().collect();
        let (k, v) = (data(3 * 4, 2), data(3 * 4, 3));
        let got = cpu().attention(&q[..8], &k, &v, shape(1, 2, 2, 1));
        assert_close(&got[..4], &got[4..], 1e-7, 0.0);
    }

    #[test]
    fn attention_tiled_prefill_with_offset_and_split_k_decode_match() {
        // Covers: head_dim = 64 NEON path, pos0 > 0 tiled prefill with causal
        // split (t >= 16), and long-context FlashDecoding split-K (n_keys >= 512).
        let backend = CpuBackend::new(8);
        let (n_heads, n_kv_heads, head_dim) = (14, 2, 64);
        let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
        let (pos0, t) = (530, 21);
        let n_keys = pos0 + t;
        let q = data(t * q_dim, 21);
        let k = data(n_keys * kv_dim, 22);
        let v = data(n_keys * kv_dim, 23);
        let prefill = backend.attention(
            &q,
            &k,
            &v,
            AttnShape {
                t,
                pos0,
                n_heads,
                n_kv_heads,
                head_dim,
            },
        );
        let split_k_last = backend.attention(
            &q[(t - 1) * q_dim..],
            &k,
            &v,
            AttnShape {
                t: 1,
                pos0: pos0 + t - 1,
                n_heads,
                n_kv_heads,
                head_dim,
            },
        );
        // Single-thread decode takes the non-split-K path; compare all three.
        let single_last = CpuBackend::new(1).attention(
            &q[(t - 1) * q_dim..],
            &k,
            &v,
            AttnShape {
                t: 1,
                pos0: pos0 + t - 1,
                n_heads,
                n_kv_heads,
                head_dim,
            },
        );
        assert_close(&split_k_last, &single_last, 1e-5, 1e-5);
        assert_close(&prefill[(t - 1) * q_dim..], &single_last, 1e-5, 1e-5);
    }

    #[test]
    fn gelu_and_layernorm_neon_paths_match_scalar() {
        let mut x = data(4113, 17);
        let orig = x.clone();
        cpu().gelu(&mut x);
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        let want_gelu: Vec<f32> = orig
            .iter()
            .map(|&v| {
                let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + inner.tanh())
            })
            .collect();
        assert_close(&x, &want_gelu, 1e-5, 1e-5);

        let w = 773; // hits 16-wide NEON loop + scalar tail
        let rows = 15;
        let input = data(rows * w, 18);
        let gamma = data(w, 19);
        let beta = data(w, 20);
        let got = cpu().layernorm(&input, &gamma, &beta, 1e-5);
        for r in 0..rows {
            let row = &input[r * w..(r + 1) * w];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / w as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / w as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            let want: Vec<f32> = row
                .iter()
                .enumerate()
                .map(|(j, &v)| ((v as f64 - mean) * inv * gamma[j] as f64 + beta[j] as f64) as f32)
                .collect();
            assert_close(&got[r * w..(r + 1) * w], &want, 1e-5, 1e-5);
        }
    }
}
