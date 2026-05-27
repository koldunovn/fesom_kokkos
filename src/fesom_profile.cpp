/* fesom_profile.cpp — see fesom_profile.hpp. Fence-bounded host+device phase timer. */
#include "fesom_profile.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace fesom_prof {

namespace {
std::map<std::string, double> g_secs;   // accumulated wall per phase
std::map<std::string, long>   g_calls;  // call count per phase
}

bool enabled()
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("FESOM_STEP_PROFILE");
        cached = (e && e[0]) ? 1 : 0;
    }
    return cached != 0;
}

double tic()
{
    if (!enabled()) return 0.0;
    Kokkos::fence();
    return MPI_Wtime();
}

void toc(const char *name, double t0)
{
    if (!enabled()) return;
    Kokkos::fence();
    g_secs [name] += MPI_Wtime() - t0;
    g_calls[name] += 1;
}

void reset()
{
    g_secs.clear();
    g_calls.clear();
}

void report(double loop_s, long timed_steps, int mype)
{
    if (!enabled() || mype != 0 || loop_s <= 0.0 || timed_steps <= 0) return;

    std::vector<std::pair<std::string, double>> v(g_secs.begin(), g_secs.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string,double>&a, const std::pair<std::string,double>&b) {
                  return a.second > b.second;
              });

    std::printf("[fesom_prof] phase breakdown (rank0, sorted by %% of loop):\n");
    std::printf("[fesom_prof]   %-18s  %7s  %12s  %9s\n", "phase", "%loop", "s/step", "calls/step");
    double sum = 0.0;
    for (const auto &p : v) {
        sum += p.second;
        std::printf("[fesom_prof]   %-18s  %6.2f%%  %10.4f s  %9.1f\n",
                    p.first.c_str(), 100.0 * p.second / loop_s,
                    p.second / (double)timed_steps,
                    (double)g_calls[p.first] / (double)timed_steps);
    }
    std::printf("[fesom_prof]   %-18s  %6.2f%%   (rest = gaps/host/MPI not in a phase)\n",
                "[sum]", 100.0 * sum / loop_s);
    std::fflush(stdout);
}

} // namespace fesom_prof
