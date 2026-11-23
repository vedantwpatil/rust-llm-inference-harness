//! What this chip can do at most, with no memory and no model: register-only
//! loops of the instructions the kernels are built from, on 1 to 14 threads.
//! These are the ceilings the model benchmarks are compared against.
//!
//!     cargo run --release -p inference --example ceilings
//!
//! shortcut: aarch64 only, it times NEON instructions.
//!
//! It also measures the other ceiling: how fast the cores can stream memory.

use std::arch::aarch64::*;
use std::hint::black_box;
use std::sync::Barrier;
use std::time::Instant;

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
}
const QOS_USER_INTERACTIVE: u32 = 0x21;

const ITERS: u64 = 200_000_000;

type Kernel = fn(u64) -> f32;

/// 16 independent accumulators, so the multiply-add units never wait on a result.
fn fma_f32(iters: u64) -> f32 {
    unsafe {
        let (a, b) = (vdupq_n_f32(1.000_000_1), vdupq_n_f32(0.999_999_9));
        let mut acc = [vdupq_n_f32(0.5); 16];
        for _ in 0..iters {
            for r in acc.iter_mut() {
                *r = vfmaq_f32(*r, a, b);
            }
        }
        acc.iter().map(|&v| vaddvq_f32(v)).sum()
    }
}

fn sdot_i8(iters: u64) -> f32 {
    unsafe {
        let (a, b) = (vdupq_n_s8(3), vdupq_n_s8(5));
        let mut acc = [vdupq_n_s32(1); 16];
        for _ in 0..iters {
            for r in acc.iter_mut() {
                *r = vdotq_s32(*r, a, b);
            }
        }
        acc.iter().map(|&v| vaddvq_s32(v) as f32).sum()
    }
}

/// `smmla` (int8 matrix multiply) is an unstable intrinsic, so this one is
/// written in assembly: 16 independent accumulators, as above.
fn smmla_i8(iters: u64) -> f32 {
    let mut out = [0i32; 4];
    unsafe {
        std::arch::asm!(
            ".arch_extension i8mm",
            "movi v16.16b, #3",
            "movi v17.16b, #5",
            "eor v0.16b, v0.16b, v0.16b",
            "eor v1.16b, v1.16b, v1.16b",
            "eor v2.16b, v2.16b, v2.16b",
            "eor v3.16b, v3.16b, v3.16b",
            "eor v4.16b, v4.16b, v4.16b",
            "eor v5.16b, v5.16b, v5.16b",
            "eor v6.16b, v6.16b, v6.16b",
            "eor v7.16b, v7.16b, v7.16b",
            "eor v8.16b, v8.16b, v8.16b",
            "eor v9.16b, v9.16b, v9.16b",
            "eor v10.16b, v10.16b, v10.16b",
            "eor v11.16b, v11.16b, v11.16b",
            "eor v12.16b, v12.16b, v12.16b",
            "eor v13.16b, v13.16b, v13.16b",
            "eor v14.16b, v14.16b, v14.16b",
            "eor v15.16b, v15.16b, v15.16b",
            "2:",
            "smmla v0.4s, v16.16b, v17.16b",
            "smmla v1.4s, v16.16b, v17.16b",
            "smmla v2.4s, v16.16b, v17.16b",
            "smmla v3.4s, v16.16b, v17.16b",
            "smmla v4.4s, v16.16b, v17.16b",
            "smmla v5.4s, v16.16b, v17.16b",
            "smmla v6.4s, v16.16b, v17.16b",
            "smmla v7.4s, v16.16b, v17.16b",
            "smmla v8.4s, v16.16b, v17.16b",
            "smmla v9.4s, v16.16b, v17.16b",
            "smmla v10.4s, v16.16b, v17.16b",
            "smmla v11.4s, v16.16b, v17.16b",
            "smmla v12.4s, v16.16b, v17.16b",
            "smmla v13.4s, v16.16b, v17.16b",
            "smmla v14.4s, v16.16b, v17.16b",
            "smmla v15.4s, v16.16b, v17.16b",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            "str q0, [{out}]",
            n = inout(reg) iters => _,
            out = in(reg) out.as_mut_ptr(),
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            options(nostack),
        );
    }
    out[0] as f32
}

/// Seconds from the first worker's start to the last worker's end. Each worker
/// takes its own timestamps after the barrier: a main thread that is descheduled
/// right after releasing it would otherwise shorten the interval.
fn timed(threads: usize, qos: bool, work: impl Fn(usize) + Sync) -> f64 {
    let barrier = Barrier::new(threads);
    let spans: Vec<(Instant, Instant)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (barrier, work) = (&barrier, &work);
                s.spawn(move || {
                    if qos {
                        unsafe { pthread_set_qos_class_self_np(QOS_USER_INTERACTIVE, 0) };
                    }
                    barrier.wait();
                    let start = Instant::now();
                    work(t);
                    (start, Instant::now())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let first = spans.iter().map(|s| s.0).min().unwrap();
    let last = spans.iter().map(|s| s.1).max().unwrap();
    (last - first).as_secs_f64()
}

/// Giga-operations per second over `threads` threads started together.
fn run(threads: usize, qos: bool, ops_per_iter: u64, kernel: Kernel) -> f64 {
    let secs = timed(threads, qos, |_| {
        black_box(kernel(black_box(ITERS)));
    });
    (threads as u64 * ITERS * ops_per_iter) as f64 / secs / 1e9
}

/// Gigabytes per second reading one buffer, each thread a contiguous slice of it.
fn read_bandwidth(data: &[f32], threads: usize, qos: bool) -> f64 {
    let chunk = data.len() / threads;
    let secs = timed(threads, qos, |t| {
        black_box(sum(&data[t * chunk..(t + 1) * chunk]));
    });
    (threads * chunk * 4) as f64 / secs / 1e9
}

/// 8 vector loads per step into 8 accumulators: loads are the only real work.
fn sum(x: &[f32]) -> f32 {
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 8];
        for block in x.as_chunks::<32>().0 {
            for (i, a) in acc.iter_mut().enumerate() {
                *a = vaddq_f32(*a, vld1q_f32(block.as_ptr().add(4 * i)));
            }
        }
        acc.iter().map(|&v| vaddvq_f32(v)).sum()
    }
}

fn main() {
    let data = vec![1.0f32; 1 << 28]; // 1 GiB
    println!("memory read: GB/s by thread count, median / best of 16 passes over 1 GiB");
    for qos in [false, true] {
        let row: Vec<String> = [1, 2, 4, 6, 8, 10, 12, 14]
            .iter()
            .map(|&t| {
                let mut passes: Vec<f64> = (0..16).map(|_| read_bandwidth(&data, t, qos)).collect();
                passes.sort_by(f64::total_cmp);
                format!("{t}t {:5.1}/{:5.1}", passes[8], passes[15])
            })
            .collect();
        println!("  qos {:5}  {}", qos, row.join("  "));
    }
    // Operations counted per loop iteration: 16 accumulators times the work of
    // one instruction (FMA: 4 lanes x 2; sdot: 16 multiply-adds x 2; smmla: 32 x 2).
    let kernels: [(&str, u64, Kernel); 3] = [
        ("f32 fma", 16 * 8, fma_f32),
        ("int8 sdot", 16 * 32, sdot_i8),
        ("int8 smmla", 16 * 64, smmla_i8),
    ];
    for (name, ops, kernel) in kernels {
        println!("{name}: Gop/s by thread count (best of 2)");
        for qos in [false, true] {
            let row: Vec<String> = [1, 2, 4, 8, 10, 12, 14]
                .iter()
                .map(|&t| {
                    let best = (0..2).map(|_| run(t, qos, ops, kernel)).fold(0.0, f64::max);
                    format!("{t}t {best:7.0}")
                })
                .collect();
            println!("  qos {:5}  {}", qos, row.join("  "));
        }
    }
}
