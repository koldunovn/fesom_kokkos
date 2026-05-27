#ifndef FESOM_PROFILE_HPP
#define FESOM_PROFILE_HPP
//
// fesom_profile — lightweight per-phase wall-time profiler (M5.6).
//
// Gated by env FESOM_STEP_PROFILE (zero cost when off). FENCE-BOUNDED so it
// captures HOST *and* DEVICE time in each phase — the M5.5b/blmc lesson: a host
// kernel's compute (e.g. a single-threaded host smoother) is INVISIBLE to a
// GPU-kernel profiler like nsys, but shows up here. Begin/accumulate into named
// buckets; fesom_main resets at the loop-timer warmup boundary and reports a
// sorted %-of-loop breakdown. Use it to find where the step spends time
// unnecessarily (hidden host compute, host-staged round-trips, …).
//
#ifdef __cplusplus
namespace fesom_prof {

// Is profiling on? (FESOM_STEP_PROFILE set; cached.)
bool enabled();

// Kokkos::fence() + MPI_Wtime() — the start of a timed region (0.0 if disabled,
// so the fence is skipped entirely when off).
double tic();

// Fence + accumulate (MPI_Wtime() - t0) into the named bucket (no-op if disabled).
void toc(const char *name, double t0);

void reset();                                       // zero all buckets
void report(double loop_s, long timed_steps, int mype);  // rank-0 sorted print

} // namespace fesom_prof

// Convenience: time the statement/block between BEG and END into `name`.
#define FPROF_BEG(t)        double t = fesom_prof::tic()
#define FPROF_END(t, name)  fesom_prof::toc((name), (t))
#endif // __cplusplus

#endif // FESOM_PROFILE_HPP
