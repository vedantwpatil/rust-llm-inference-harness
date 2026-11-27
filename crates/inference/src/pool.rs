//! A fork-join thread pool for the short parallel regions of a forward pass.
//!
//! Decode runs about fifty dependent regions per token (every matmul is one),
//! each a few tens of microseconds of work. rayon's wake-up cost per region is
//! the same order as that work, so workers here are persistent and spin on an
//! atomic for a while before they sleep: starting a region is a handful of
//! atomic operations when the pool is busy.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How long an idle worker spins before it sleeps on the condition variable.
/// Longer than the serial gaps between two regions of one token, shorter than
/// anything a caller does between tokens.
const SPIN: Duration = Duration::from_micros(200);

/// Pads a field to Apple Silicon's 128-byte cache line so hot atomics written
/// by different threads do not share a line.
#[repr(align(128))]
struct CacheAligned<T>(T);

type RegionFn = *const (dyn Fn(usize, usize) + Sync);

fn noop_region(_: usize, _: usize) {}

/// Read by every worker when a region starts; written only by the caller
/// between regions while holding `Pool::run_lock`.
struct Dispatch {
    /// Upper 56 bits: monotonically increasing region epoch.
    /// Lower 8 bits: participating thread count (`width <= 64`).
    state: AtomicU64,
    /// Closure for the current region with its lifetime erased; see safety
    /// comments in `Pool::run_threads` and `worker`.
    job: UnsafeCell<RegionFn>,
    shutdown: AtomicBool,
}

/// Cold sleep/wake state and rare panic flag, kept off `Dispatch`'s cache line.
struct Sleep {
    /// Bit `id` is set while worker `id` sleeps (or is entering sleep) on `wake`.
    sleeping: AtomicU64,
    lock: Mutex<()>,
    wake: Condvar,
    panicked: AtomicBool,
}

struct Shared {
    dispatch: CacheAligned<Dispatch>,
    sleep: CacheAligned<Sleep>,
    /// Worker `id` stores the epoch it finished here; one cache line per worker
    /// so barrier completion never contends on a single counter.
    worker_done: Box<[CacheAligned<AtomicU64>]>,
}

// SAFETY: `dispatch.job` points to a `Sync` closure and is written only while
// holding `Pool::run_lock` after every participating worker of the prior region
// has completed; workers read it only after observing the matching epoch with
// `Acquire`/`SeqCst` and verifying `id < width`. Every other field is atomic or
// synchronized.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// One region at a time; holds the current epoch so two callers take turns.
    run_lock: Mutex<u64>,
}

impl Pool {
    /// A pool of `threads` threads in total: the caller of `run` is one of them.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        assert!(threads <= 64, "`Sleep::sleeping` has one bit per thread");
        let worker_done = (0..threads)
            .map(|_| CacheAligned(AtomicU64::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let shared = Arc::new(Shared {
            dispatch: CacheAligned(Dispatch {
                state: AtomicU64::new(0),
                job: UnsafeCell::new(&noop_region as RegionFn),
                shutdown: AtomicBool::new(false),
            }),
            sleep: CacheAligned(Sleep {
                sleeping: AtomicU64::new(0),
                lock: Mutex::new(()),
                wake: Condvar::new(),
                panicked: AtomicBool::new(false),
            }),
            worker_done,
        });
        let workers = (1..threads)
            .map(|id| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || worker(&shared, id))
            })
            .collect();
        Self {
            shared,
            workers,
            run_lock: Mutex::new(0),
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `f(thread_id, actual_width)` once on each of `actual_width`
    /// threads (`width.min(self.threads())`), returning when all finish.
    /// Zero heap allocation and zero mutex locks while workers are spinning.
    pub fn run_threads(&self, width: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let actual_width = width.min(self.threads());
        if actual_width <= 1 {
            f(0, 1);
            return;
        }
        let mut epoch_guard = self.run_lock.lock().unwrap();
        *epoch_guard += 1;
        let epoch = *epoch_guard;

        // SAFETY: only the lifetime is erased. `run_lock` guarantees no other
        // caller writes `job`, the prior region waited for all of its active
        // workers to finish reading `job`, and this call will not return until
        // every worker in `1..actual_width` stores `epoch` to `worker_done`.
        let erased: RegionFn = unsafe { std::mem::transmute(f) };
        unsafe { *self.shared.dispatch.0.job.get() = erased };

        let packed = (epoch << 8) | actual_width as u64;
        self.shared.dispatch.0.state.store(packed, SeqCst);

        // Only a sleeper this region needs is worth a lock + system call:
        // workers outside `actual_width` stay asleep through narrow regions.
        let needed = (u64::MAX >> (64 - actual_width)) & !1;
        if self.shared.sleep.0.sleeping.load(SeqCst) & needed != 0 {
            let _guard = self.shared.sleep.0.lock.lock().unwrap();
            self.shared.sleep.0.wake.notify_all();
        }

        // Catch panics on the caller too so we still wait for workers before
        // unwinding and freeing the closure.
        let caller_panicked = catch_unwind(AssertUnwindSafe(|| f(0, actual_width))).is_err();

        for slot in &self.shared.worker_done[1..actual_width] {
            while slot.0.load(Acquire) != epoch {
                std::hint::spin_loop();
            }
        }

        let worker_panicked = self.shared.sleep.0.panicked.load(Relaxed);
        if worker_panicked {
            self.shared.sleep.0.panicked.store(false, Relaxed);
        }
        assert!(
            !caller_panicked && !worker_panicked,
            "a task in a parallel region panicked"
        );
    }

    /// Calls `f(i)` for every `i` in `0..n_tasks`, spread over the pool, and
    /// returns when all calls are done. `f` must not call `run` on this pool.
    pub fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        self.run_width(usize::MAX, n_tasks, f);
    }

    /// `run` on at most `width` threads, the caller included. A region that
    /// streams memory gets no faster past the width that saturates it, and every
    /// extra thread is one more that can be late to the barrier.
    pub fn run_width(&self, width: usize, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_tasks <= 1 || width <= 1 || self.workers.is_empty() {
            (0..n_tasks).for_each(f);
            return;
        }
        let next = AtomicUsize::new(0);
        self.run_threads(width.min(n_tasks), &|_, _| loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= n_tasks {
                break;
            }
            f(i);
        });
    }

    /// Splits `data` into chunks of `chunk` elements (the last may be shorter)
    /// and calls `f(chunk_index, chunk)` for each, in parallel.
    pub fn for_each_chunk_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        f: impl Fn(usize, &mut [T]) + Sync,
    ) {
        self.for_each_chunk_mut_width(usize::MAX, data, chunk, f);
    }

    /// `for_each_chunk_mut` on at most `width` threads, as `run_width`.
    pub fn for_each_chunk_mut_width<T: Send>(
        &self,
        width: usize,
        data: &mut [T],
        chunk: usize,
        f: impl Fn(usize, &mut [T]) + Sync,
    ) {
        assert!(chunk > 0);
        struct Base<T>(*mut T);
        // SAFETY: each task gets a different, non-overlapping range of `data`.
        unsafe impl<T: Send> Sync for Base<T> {}
        impl<T> Base<T> {
            // A method, not a field access: closures capture disjoint fields,
            // which would capture the bare (non-Sync) pointer.
            fn get(&self) -> *mut T {
                self.0
            }
        }

        let len = data.len();
        let base = Base(data.as_mut_ptr());
        self.run_width(width, len.div_ceil(chunk), &|i| {
            let start = i * chunk;
            let n = chunk.min(len - start);
            // SAFETY: `start..start + n` lies inside `data`, no other task gets
            // any of it, and `data` stays mutably borrowed until `run` returns.
            f(i, unsafe {
                std::slice::from_raw_parts_mut(base.get().add(start), n)
            });
        });
    }
}

/// Pins a worker thread to macOS's interactive QoS class so XNU keeps spinning
/// workers on performance cores rather than migrating them to efficiency cores.
#[cfg(target_os = "macos")]
fn prefer_performance_core() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
    }
    const QOS_USER_INTERACTIVE: u32 = 0x21;
    // SAFETY: valid QoS class constant, applies only to the calling thread.
    unsafe { pthread_set_qos_class_self_np(QOS_USER_INTERACTIVE, 0) };
}

#[cfg(not(target_os = "macos"))]
fn prefer_performance_core() {}

fn worker(shared: &Shared, id: usize) {
    prefer_performance_core();
    let bit = 1u64 << id;
    let mut seen = 0u64;
    let mut polls = 0u32;
    let mut spin_until: Option<Instant> = None;
    loop {
        let s = loop {
            let s = shared.dispatch.0.state.load(Acquire);
            if (s >> 8) != seen {
                break s;
            }
            if shared.dispatch.0.shutdown.load(Relaxed) {
                return;
            }
            polls = polls.wrapping_add(1);
            // Reading the clock costs more than a poll: check it rarely, and
            // start the spin timer lazily so busy back-to-back regions never
            // touch the clock.
            if polls.is_multiple_of(256) {
                let now = Instant::now();
                let deadline = *spin_until.get_or_insert(now + SPIN);
                if now > deadline {
                    let mut guard = shared.sleep.0.lock.lock().unwrap();
                    shared.sleep.0.sleeping.fetch_or(bit, SeqCst);
                    let s = loop {
                        let s = shared.dispatch.0.state.load(SeqCst);
                        if shared.dispatch.0.shutdown.load(SeqCst) {
                            shared.sleep.0.sleeping.fetch_and(!bit, Release);
                            return;
                        }
                        let epoch = s >> 8;
                        if epoch != seen {
                            if id < (s & 0xff) as usize {
                                break s;
                            }
                            // Narrow region excluding this worker: stay asleep.
                            seen = epoch;
                        }
                        guard = shared.sleep.0.wake.wait(guard).unwrap();
                    };
                    shared.sleep.0.sleeping.fetch_and(!bit, Release);
                    drop(guard);
                    break s;
                }
            }
            std::hint::spin_loop();
        };
        let epoch = s >> 8;
        let width = (s & 0xff) as usize;
        seen = epoch;
        if id >= width {
            // Left out of this region: do not reset `spin_until`, so a worker
            // skipped by region after region goes to sleep.
            continue;
        }
        // SAFETY: `epoch` was observed with `Acquire`/`SeqCst`, ordering the
        // caller's write to `job` before this read, and `id < width` means the
        // caller will not return or overwrite `job` before `worker_done[id] == epoch`.
        let f = unsafe { &**shared.dispatch.0.job.get() };
        if catch_unwind(AssertUnwindSafe(|| f(id, width))).is_err() {
            shared.sleep.0.panicked.store(true, Relaxed);
        }
        shared.worker_done[id].0.store(epoch, Release);
        spin_until = None;
        polls = 0;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.dispatch.0.shutdown.store(true, SeqCst);
        // Taking the lock orders this after any worker that is about to sleep.
        let _guard = self.shared.sleep.0.lock.lock().unwrap();
        self.shared.sleep.0.wake.notify_all();
        drop(_guard);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_exactly_once() {
        let pool = Pool::new(4);
        for n in [0, 1, 2, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "n = {n}");
        }
    }

    #[test]
    fn a_narrow_region_uses_at_most_its_width() {
        let pool = Pool::new(6);
        // Rising, falling and back to back, with tasks long enough that every
        // thread allowed in gets a share; the quiet gap sends workers to sleep.
        for round in 0..60 {
            if round == 30 {
                thread::sleep(Duration::from_millis(20));
            }
            let width = [100, 1, 6, 2, 3, 6, 2][round % 7];
            let threads = Mutex::new(std::collections::HashSet::new());
            pool.run_width(width, 24, &|_| {
                threads.lock().unwrap().insert(thread::current().id());
                thread::sleep(Duration::from_micros(100));
            });
            let used = threads.lock().unwrap().len();
            assert!(
                used <= width.min(6),
                "round {round}: width {width} used {used}"
            );
        }
    }

    #[test]
    fn many_back_to_back_regions_and_sleeping_workers() {
        let pool = Pool::new(4);
        let total = AtomicUsize::new(0);
        for round in 0..3000 {
            pool.run(16, &|_| {
                total.fetch_add(1, Relaxed);
            });
            if round == 1500 {
                // long enough for the workers to give up spinning and sleep
                thread::sleep(Duration::from_millis(20));
            }
        }
        assert_eq!(total.load(Relaxed), 3000 * 16);
    }

    #[test]
    fn chunks_cover_the_slice_once() {
        let pool = Pool::new(3);
        let mut data = vec![0usize; 103];
        pool.for_each_chunk_mut(&mut data, 10, |c, chunk| {
            for v in chunk.iter_mut() {
                *v += c + 1;
            }
        });
        let want: Vec<usize> = (0..103).map(|i| i / 10 + 1).collect();
        assert_eq!(data, want);
    }

    #[test]
    #[should_panic(expected = "panicked")]
    fn a_panicking_task_is_reported_after_the_region_ends() {
        let pool = Pool::new(3);
        pool.run(32, &|i| assert_ne!(i, 17));
    }

    #[test]
    fn single_thread_pool_runs_inline() {
        let pool = Pool::new(1);
        let mut data = vec![1u8; 5];
        pool.for_each_chunk_mut(&mut data, 2, |_, c| c.fill(9));
        assert_eq!(data, [9; 5]);
    }

    #[test]
    fn run_threads_passes_distinct_ids_and_catches_caller_panic() {
        let pool = Pool::new(4);
        for req in [0, 1, 2, 4, 8] {
            let want = req.clamp(1, 4);
            let seen: Vec<AtomicUsize> = (0..want).map(|_| AtomicUsize::new(0)).collect();
            pool.run_threads(req, &|tid, width| {
                assert_eq!(width, want);
                seen[tid].fetch_add(1, Relaxed);
            });
            assert!(seen.iter().all(|s| s.load(Relaxed) == 1));
        }
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            pool.run_threads(4, &|tid, _| assert_ne!(tid, 0));
        }));
        assert!(panicked.is_err());
    }
}
