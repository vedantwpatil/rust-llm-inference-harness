//! A fork-join thread pool for the short parallel regions of a forward pass.
//!
//! Decode runs about fifty dependent regions per token (every matmul is one),
//! each a few tens of microseconds of work. rayon's wake-up cost per region is
//! the same order as that work, so workers here are persistent and spin on an
//! atomic for a while before they sleep: starting a region is a handful of
//! atomic operations when the pool is busy.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How long an idle worker spins before it sleeps on the condition variable.
/// Longer than the serial gaps between two regions of one token, shorter than
/// anything a caller does between tokens.
const SPIN: Duration = Duration::from_micros(200);

/// One parallel region: calls `f(i)` once for every `i` in `0..n_tasks`.
struct Job {
    /// The caller's closure with its lifetime erased; see the safety comment in
    /// `Job::work`.
    f: *const (dyn Fn(usize) + Sync),
    n_tasks: usize,
    /// Threads that may work on it, the caller included.
    width: usize,
    /// Next task index nobody has claimed.
    next: AtomicUsize,
    /// Tasks that have finished running.
    done: AtomicUsize,
    panicked: AtomicBool,
}

// SAFETY: `f` points at a `Sync` closure and is only dereferenced under the
// conditions in `Job::work`; every other field is atomic.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// Claims and runs tasks until none are left.
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, Relaxed);
            if i >= self.n_tasks {
                return;
            }
            // SAFETY: `i < n_tasks` means this task is not done yet, so
            // `done < n_tasks`, and `Pool::run` (which owns the closure) does
            // not return before `done == n_tasks`. A thread holding a stale job
            // always fails the check above, so it never reaches this line.
            let f = unsafe { &*self.f };
            // A panic must not unwind out of here: `run` would return early
            // and free the closure while other threads still run it.
            if catch_unwind(AssertUnwindSafe(|| f(i))).is_err() {
                self.panicked.store(true, Relaxed);
            }
            self.done.fetch_add(1, Release);
        }
    }
}

struct State {
    generation: usize,
    job: Option<Arc<Job>>,
}

struct Shared {
    state: Mutex<State>,
    /// Mirror of `State::generation`, so a spinning worker can poll without
    /// taking the lock.
    generation: AtomicUsize,
    /// Threads taking part in the current region, the caller included: workers
    /// `1..width` join, the rest skip it without touching the lock.
    width: AtomicUsize,
    /// Bit `i` is set while worker `i` sleeps on `wake`.
    sleeping: AtomicU64,
    wake: Condvar,
    shutdown: AtomicBool,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// One region at a time: two threads sharing a pool take turns.
    run_lock: Mutex<()>,
}

impl Pool {
    /// A pool of `threads` threads in total: the caller of `run` is one of them.
    pub fn new(threads: usize) -> Self {
        assert!(threads <= 64, "`Shared::sleeping` has one bit per thread");
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                generation: 0,
                job: None,
            }),
            generation: AtomicUsize::new(0),
            width: AtomicUsize::new(0),
            sleeping: AtomicU64::new(0),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let workers = (1..threads.max(1))
            .map(|id| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || worker(&shared, id))
            })
            .collect();
        Self {
            shared,
            workers,
            run_lock: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
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
        let _one_region = self.run_lock.lock().unwrap();
        // SAFETY: only the lifetime is erased. This function does not return
        // until `done == n_tasks`, and `Job::work` dereferences the pointer only
        // for tasks that are not done yet.
        let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Arc::new(Job {
            f,
            n_tasks,
            width,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });

        {
            let mut state = self.shared.state.lock().unwrap();
            state.generation += 1;
            state.job = Some(Arc::clone(&job));
            self.shared.width.store(width, Relaxed);
            self.shared.generation.store(state.generation, Release);
        }
        // Only a sleeper this region needs is worth a system call: workers
        // outside its width sleep through a run of narrow regions.
        let needed = u64::MAX >> (64 - width.min(64)) & !1;
        if self.shared.sleeping.load(Acquire) & needed != 0 {
            self.shared.wake.notify_all();
        }

        job.work();
        while job.done.load(Acquire) < n_tasks {
            std::hint::spin_loop();
        }
        assert!(
            !job.panicked.load(Relaxed),
            "a task in a parallel region panicked"
        );
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

fn worker(shared: &Shared, id: usize) {
    let bit = 1 << id;
    let mut seen = 0;
    let mut spin_until = Instant::now() + SPIN;
    loop {
        let mut polls = 0u32;
        while shared.generation.load(Acquire) == seen {
            if shared.shutdown.load(Relaxed) {
                return;
            }
            polls += 1;
            // Reading the clock costs more than a poll: check it rarely.
            if polls.is_multiple_of(256) && Instant::now() > spin_until {
                let mut state = shared.state.lock().unwrap();
                // Re-checked under the lock: `run` publishes under it too, so
                // either it already did (no sleep) or it will see `sleeping`.
                while state.generation == seen && !shared.shutdown.load(Relaxed) {
                    shared.sleeping.fetch_or(bit, Release);
                    state = shared.wake.wait(state).unwrap();
                    shared.sleeping.fetch_and(!bit, Release);
                }
                break;
            }
            std::hint::spin_loop();
        }
        // The width was stored before the generation this worker just saw, so a
        // region that leaves it out costs it one load and no lock. The timer is
        // not restarted: a worker left out of region after region goes to sleep.
        // (If a newer region has already replaced the width, this worker sits
        // out that one too: one region's parallelism is lost, never a task.)
        if id >= shared.width.load(Relaxed) {
            seen = shared.generation.load(Acquire);
            continue;
        }
        // A worker that missed a whole region just takes the newest job, and
        // checks the width of that one: it may be narrower than the last.
        let job = {
            let state = shared.state.lock().unwrap();
            seen = state.generation;
            state.job.clone()
        };
        if let Some(job) = job.filter(|job| id < job.width) {
            job.work();
        }
        spin_until = Instant::now() + SPIN;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Relaxed);
        // Taking the lock orders this after any worker that is about to sleep.
        drop(self.shared.state.lock().unwrap());
        self.shared.wake.notify_all();
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
}
