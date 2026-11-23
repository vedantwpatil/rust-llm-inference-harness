use crate::backend::{AttnShape, Backend, Matrix, Weights};
use crate::kernel::{Q8_BLOCK, dot, exp, q8_scale, quantize_into, tile, tile_q8};
use crate::pool::Pool;

/// Plain Rust on the CPU: NEON kernels, a spinning thread pool. The reference
/// every other backend is compared against.
pub struct CpuBackend {
    pool: Pool,
}

impl CpuBackend {
    pub fn new(threads: usize) -> Self {
        Self {
            pool: Pool::new(threads),
        }
    }
}

impl CpuBackend {
    /// Elements per task when `len` of them are shared out, about four tasks per
    /// thread, but never fewer than `min`. A task has to outweigh the ~120 ns it
    /// costs to claim it (every thread's claim queues on one counter), so a task
    /// of a few hundred elements makes a region slower than doing it inline;
    /// `len <= min` is one task, which `Pool` runs on the calling thread.
    fn chunk(&self, len: usize, min: usize) -> usize {
        len.div_ceil(self.pool.threads() * 4).max(min)
    }

    /// `x` (`[t][k]`) rounded to int8, with one scale per `Q8_BLOCK` values of
    /// each row.
    fn quantize_rows(&self, x: &[f32], t: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
        let blocks = k.div_ceil(Q8_BLOCK);
        // Whole rows per task.
        let rows = self.chunk(x.len(), 8192).div_ceil(k);
        let mut scales = vec![0.0; t * blocks];
        self.pool
            .for_each_chunk_mut(&mut scales, rows * blocks, |c, out| {
                let pairs = x[c * rows * k..]
                    .chunks_exact(k)
                    .zip(out.chunks_mut(blocks));
                for (row, out) in pairs {
                    for (s, vals) in out.iter_mut().zip(row.chunks(Q8_BLOCK)) {
                        *s = q8_scale(vals);
                    }
                }
            });
        let mut xq = vec![0i8; x.len()];
        self.pool.for_each_chunk_mut(&mut xq, rows * k, |c, out| {
            for (j, out) in out.chunks_exact_mut(k).enumerate() {
                let i = c * rows + j;
                let pairs = out
                    .chunks_mut(Q8_BLOCK)
                    .zip(x[i * k..][..k].chunks(Q8_BLOCK));
                for (b, (out, vals)) in pairs.enumerate() {
                    quantize_into(vals, scales[i * blocks + b], out);
                }
            }
        });
        (xq, scales)
    }
}

impl CpuBackend {
    /// `f(row, out)` for every `width`-wide row of `x`, rows in parallel, each
    /// result written to the matching row of the returned `Vec`.
    fn map_rows(&self, x: &[f32], width: usize, f: impl Fn(&[f32], &mut [f32]) + Sync) -> Vec<f32> {
        assert_eq!(x.len() % width, 0);
        let rows = self.chunk(x.len(), 8192).div_ceil(width);
        let mut out = vec![0.0; x.len()];
        self.pool
            .for_each_chunk_mut(&mut out, rows * width, |c, out| {
                let pairs = x[c * rows * width..]
                    .chunks_exact(width)
                    .zip(out.chunks_exact_mut(width));
                pairs.for_each(|(row, out)| f(row, out));
            });
        out
    }
}

impl Default for CpuBackend {
    /// `INFERENCE_THREADS` if set, else one thread per core up to 8.
    // shortcut: the cap of 8 is measured on an M3 Max (10 performance + 4
    // efficiency cores), where all 14 threads decode slower than 8 because the
    // efficiency cores straggle; read the performance-core count when another
    // machine matters.
    fn default() -> Self {
        let threads = std::env::var("INFERENCE_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get().min(8)));
        Self::new(threads)
    }
}

pub(crate) fn softmax_in_place(x: &mut [f32]) {
    // Subtracting the max keeps exp() from overflowing; the result is the same.
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() {
        *v = exp(*v - max);
    }
    let sum: f32 = x.iter().sum();
    for v in x.iter_mut() {
        *v /= sum;
    }
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

/// Weight rows `first..first + rows` against every token: `out[r * t + i]` is
/// row `first + r` dotted with token `i`. Rows go four at a time, tokens four
/// at a time when there are that many, so a loaded weight is reused across
/// tokens.
fn matmul_block(ops: &impl Operands, first: usize, t: usize, out: &mut [f32]) {
    let rows = out.len() / t;
    let mut r = 0;
    while r + 4 <= rows {
        let mut i = 0;
        while i + 4 <= t {
            let tl = ops.tile::<4, 4, 1>(first + r, i);
            for (a, row) in tl.iter().enumerate() {
                out[(r + a) * t + i..][..4].copy_from_slice(row);
            }
            i += 4;
        }
        for i in i..t {
            let tl = ops.tile::<4, 1, 4>(first + r, i);
            for (a, row) in tl.iter().enumerate() {
                out[(r + a) * t + i] = row[0];
            }
        }
        r += 4;
    }
    for r in r..rows {
        for i in 0..t {
            out[r * t + i] = ops.tile::<1, 1, 4>(first + r, i)[0][0];
        }
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
        // Each task owns whole weight rows and every token's dot product with
        // them, so a weight row is read from memory once however large t is.
        const PREFILL_ROWS: usize = 16;

        // One token: `[n_out][1]` is already `[1][n_out]`, so the tasks write
        // the result and the bias directly. A second parallel region would do
        // nothing but cost a fork-join, and a decoded token has ~100 matmuls.
        if t == 1 {
            // Decode is bound by memory, and a task that streams too few bytes
            // never gets the memory system up to speed: 16 int8 rows (14 KB)
            // reached 80 of the 126 GB/s the chip reads, 256 KB tasks reach
            // about 105. So tasks are sized in bytes, but never so large that a
            // thread gets fewer than a few of them to balance the load.
            const TASK_BYTES: usize = 256 << 10;
            const TASKS_PER_THREAD: usize = 4;
            let target = (row_bytes * n_out / (self.pool.threads() * TASKS_PER_THREAD))
                .clamp(16 << 10, TASK_BYTES);
            // Whole tiles of four rows, so `matmul_block` has no row tail.
            let rows = (target / row_bytes).max(4) / 4 * 4;
            let mut out = vec![0.0; n_out];
            self.pool.for_each_chunk_mut(&mut out, rows, |c, slab| {
                matmul_block(ops, c * rows, 1, slab);
                if let Some(b) = bias {
                    for (v, b) in slab.iter_mut().zip(&b[c * rows..]) {
                        *v += b;
                    }
                }
            });
            return out;
        }

        // Several tokens. A task's rows give a 16-wide strip of every token's
        // output row, so it computes them as `[rows][t]` in a small block and
        // copies each token's strip to its place in the result.
        let mut out = vec![0.0; t * n_out];
        let strips = Strips::new(&mut out);
        self.pool.run(n_out.div_ceil(PREFILL_ROWS), &|c| {
            let first = c * PREFILL_ROWS;
            let rows = PREFILL_ROWS.min(n_out - first);
            let mut block = vec![0.0; rows * t];
            matmul_block(ops, first, t, &mut block);
            let mut strip = [0.0; PREFILL_ROWS];
            for i in 0..t {
                for (r, v) in strip[..rows].iter_mut().enumerate() {
                    *v = block[r * t + i] + bias.map_or(0.0, |b| b[first + r]);
                }
                // SAFETY: this task writes only columns `first..first + rows` of
                // each row of `out`, and other tasks have other columns.
                unsafe { strips.write(i * n_out + first, &strip[..rows]) };
            }
        });
        out
    }
}

/// A buffer that tasks write disjoint pieces of at once, where the pieces are
/// not contiguous (a column strip of each row of a matrix), so `&mut` slices
/// cannot be handed out the way `Pool::for_each_chunk_mut` does.
struct Strips<'a> {
    ptr: *mut f32,
    len: usize,
    buffer: std::marker::PhantomData<&'a mut [f32]>,
}

// SAFETY: the pointer is only used by `write`, whose callers promise that no two
// threads write the same element.
unsafe impl Sync for Strips<'_> {}

impl<'a> Strips<'a> {
    fn new(buffer: &'a mut [f32]) -> Self {
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
    unsafe fn write(&self, at: usize, values: &[f32]) {
        assert!(at + values.len() <= self.len);
        // SAFETY: in bounds by the assert, and `values` cannot overlap the
        // buffer, which `Strips` borrows exclusively; exclusivity of the
        // destination is the caller's promise.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), self.ptr.add(at), values.len()) };
    }
}

impl Backend for CpuBackend {
    fn matmul(&self, x: &[f32], t: usize, w: &Matrix, bias: Option<&[f32]>) -> Vec<f32> {
        let (n_out, k) = (w.rows, w.cols);
        assert_eq!(x.len(), t * k);
        match &w.weights {
            Weights::F32(data) => self.product(
                &F32Operands { w: data, x, k },
                t,
                n_out,
                w.bytes() / n_out,
                bias,
            ),
            Weights::Q8 { values, scales } => {
                // Quantize each token's activations once for the whole product.
                let (xq, x_scales) = self.quantize_rows(x, t, k);
                let ops = Q8Operands {
                    w: values,
                    w_scales: scales,
                    x: &xq,
                    x_scales: &x_scales,
                    k,
                };
                self.product(&ops, t, n_out, w.bytes() / n_out, bias)
            }
        }
    }

    fn layernorm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let width = gamma.len();
        assert_eq!(beta.len(), width);
        self.map_rows(x, width, |row, out| {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let params = gamma.iter().zip(beta);
            for ((o, v), (g, b)) in out.iter_mut().zip(row).zip(params) {
                *o = (v - mean) * inv_std * g + b;
            }
        })
    }

    fn rmsnorm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let width = gamma.len();
        self.map_rows(x, width, |row, out| {
            let scale = 1.0 / (dot(row, row) / width as f32 + eps).sqrt();
            for ((o, v), g) in out.iter_mut().zip(row).zip(gamma) {
                *o = v * scale * g;
            }
        })
    }

    fn swiglu(&self, gate_up: &[f32], width: usize) -> Vec<f32> {
        assert_eq!(gate_up.len() % (2 * width), 0);
        let mut out = vec![0.0; gate_up.len() / 2];
        // Chunks over the flat output, not one per row: a decoded token is a
        // single row and would otherwise run on one thread.
        let chunk_len = self.chunk(out.len(), 8192);
        self.pool
            .for_each_chunk_mut(&mut out, chunk_len, |c, chunk| {
                // Cut the chunk where a row ends: each piece is a plain zip of
                // slices, which the compiler vectorizes.
                let (mut at, mut rest) = (c * chunk_len, chunk);
                while !rest.is_empty() {
                    let (row, j) = (at / width, at % width);
                    let (run, tail) = rest.split_at_mut(rest.len().min(width - j));
                    let first = row * 2 * width + j;
                    let gate = &gate_up[first..][..run.len()];
                    let up = &gate_up[first + width..][..run.len()];
                    for ((v, &g), &u) in run.iter_mut().zip(gate).zip(up) {
                        *v = g / (1.0 + exp(-g)) * u;
                    }
                    at += run.len();
                    rest = tail;
                }
            });
        out
    }

    fn rope(&self, x: &mut [f32], pos0: usize, n_heads: usize, head_dim: usize, inv_freq: &[f32]) {
        let half = head_dim / 2;
        assert_eq!(inv_freq.len(), half);
        let width = n_heads * head_dim;
        assert_eq!(x.len() % width, 0);
        let rows = self.chunk(x.len(), 8192).div_ceil(width);
        self.pool.for_each_chunk_mut(x, rows * width, |c, chunk| {
            for (i, row) in chunk.chunks_exact_mut(width).enumerate() {
                let pos = (pos0 + c * rows + i) as f32;
                // The angles depend on the position and pair index only, so every head shares them.
                let angles: Vec<(f32, f32)> =
                    inv_freq.iter().map(|f| (pos * f).sin_cos()).collect();
                for head in row.chunks_exact_mut(head_dim) {
                    let (lo, hi) = head.split_at_mut(half);
                    for (j, &(sin, cos)) in angles.iter().enumerate() {
                        let (a, b) = (lo[j], hi[j]);
                        lo[j] = a * cos - b * sin;
                        hi[j] = b * cos + a * sin;
                    }
                }
            }
        });
    }

    fn gelu(&self, x: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        // Serial, this was a third of prefill time. Decode's 3072 values are one
        // inline task.
        self.pool
            .for_each_chunk_mut(x, self.chunk(x.len(), 4096), |_, chunk| {
                for v in chunk {
                    let inner = SQRT_2_OVER_PI * (*v + 0.044715 * *v * *v * *v);
                    // tanh(a) = 1 - 2 / (e^(2a) + 1)
                    *v = 0.5 * *v * (2.0 - 2.0 / (exp(2.0 * inner) + 1.0));
                }
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
        assert_eq!(q.len(), t * q_dim);
        assert_eq!(k.len(), (pos0 + t) * kv_dim);
        assert_eq!(v.len(), k.len());
        assert_eq!(n_heads % n_kv_heads, 0);
        let group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0; t * q_dim];
        // Work items are (query, head) pairs, item `idx` owning chunk `idx` of
        // `out`. A task takes several when there are many (a whole query's
        // heads in prefill, where one item is ~2 us), one in decode.
        let per_task = (t * n_heads)
            .div_ceil(self.pool.threads() * 4)
            .clamp(1, n_heads);
        self.pool
            .for_each_chunk_mut(&mut out, per_task * head_dim, |c, chunk| {
                let mut scores = Vec::new();
                for (j, o) in chunk.chunks_exact_mut(head_dim).enumerate() {
                    let idx = c * per_task + j;
                    let (i, h) = (idx / n_heads, idx % n_heads);
                    let kv_off = (h / group) * head_dim;
                    let qv = &q[i * q_dim + h * head_dim..][..head_dim];
                    // The causal mask: key j > pos0 + i is never scored, which is the
                    // same as a -inf score before softmax.
                    let n_keys = pos0 + i + 1;
                    scores.clear();
                    scores.extend(
                        (0..n_keys).map(|j| dot(qv, &k[j * kv_dim + kv_off..][..head_dim]) * scale),
                    );
                    softmax_in_place(&mut scores);
                    for (j, p) in scores.iter().enumerate() {
                        let vj = &v[j * kv_dim + kv_off..][..head_dim];
                        for (acc, x) in o.iter_mut().zip(vj) {
                            *acc = p.mul_add(*x, *acc);
                        }
                    }
                }
            });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;
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
}
