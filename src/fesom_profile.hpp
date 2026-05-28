#ifndef FESOM_PROFILE_HPP
#define FESOM_PROFILE_HPP
//
// fesom_profile — lightweight per-phase + per-kernel wall-time profiler.
//
// M5.6: fence-bounded host+device PHASE timer (via tic/toc + the PMARK chain
//   pattern in fesom_step.cpp / fesom_ice.cpp). Captures HOST *and* DEVICE time
//   in each phase — the M5.5b/blmc lesson: a host kernel's compute (e.g. a
//   single-threaded host smoother) is INVISIBLE to a GPU-kernel profiler like
//   nsys, but shows up here.
//
// M5.11: KERNEL-level auto-instrumentation via Kokkos profiling-tools callbacks
//   (register parallel_{for,reduce,scan} begin/end + deep_copy begin in
//   install_callbacks). Every parallel_for with a string label automatically
//   accumulates into its own bucket — NO source-level wrapping needed. The
//   per-kernel buckets coexist with the existing phase buckets in the same
//   report (sorted by % of loop).
//
// All gated by env FESOM_STEP_PROFILE — zero cost when off (callbacks return
// immediately, tic/toc skip the fence, push/pop_region skip the NVTX bridge).
//
#ifdef __cplusplus
namespace fesom_prof {

// Is profiling on? (FESOM_STEP_PROFILE set; cached.)
bool enabled();

// PHASE timing primitives (M5.6). tic = fence + Wtime; toc = fence +
// accumulate into named bucket. Use directly (e.g. PMARK chains in
// fesom_step.cpp) or via the FPROF_BEG/END macros below.
double tic();
void   toc(const char *name, double t0);

// NVTX bridge (M5.11). Standalone push/pop region — visible in nsys when
// `-t nvtx` is on. Safe to call whether profiling is enabled or not; no-op
// when off.
void push_region(const char *name);
void pop_region();

// M5.11: install Kokkos profiling-tools callbacks (per-kernel timing +
// deep_copy counter). Call once after Kokkos::initialize(). No-op when
// FESOM_STEP_PROFILE is unset. The callbacks hold no Kokkos state, so no
// cleanup-before-Kokkos::finalize() is needed (unlike the M5.8 EVP static
// Kokkos::View trap).
void install_callbacks();

void reset();                                            // zero all buckets
void report(double loop_s, long timed_steps, int mype);  // rank-0 sorted print

} // namespace fesom_prof

// Phase-scope convenience macros (existing, M5.6). BEG takes no name, END
// takes the name. NOT used by the existing PMARK chain pattern (which calls
// tic/toc directly) — provided for code that prefers an explicit BEG/END pair.
#define FPROF_BEG(t)        double t = fesom_prof::tic()
#define FPROF_END(t, name)  fesom_prof::toc((name), (t))
#endif // __cplusplus

#endif // FESOM_PROFILE_HPP
