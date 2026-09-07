/*
 * test_field.cpp — unit test for fesom::Field (Milestone M1.1).
 *
 * Exercises the DualView round trip on whatever backend the binary was built for
 * (Serial / OpenMP / CUDA): host write -> sync_device -> mutate on device in a parallel_for ->
 * sync_host -> assert identity; plus a deep_copy path. All arithmetic is exact-integer-valued so
 * the `==` comparisons are deterministic on every backend.
 */
#include "fesom_field.hpp"
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstddef>

static int g_failures = 0;
#define CHECK(cond, msg) \
    do { if (!(cond)) { std::printf("  FAIL: %s\n", (msg)); ++g_failures; } } while (0)

template <class FieldType, class T>
static void roundtrip(const char* name) {
    const std::size_t N = 1000;
    FieldType f;
    f.alloc(name, N);
    CHECK(f.allocated(),   "allocated");
    CHECK(f.size() == N,   "size");

    /* host write: exact-integer values so device arithmetic stays bit-deterministic */
    T* h = f.h();
    for (std::size_t i = 0; i < N; ++i) h[i] = (T)(2 * i + 1);
    f.modify_host();
    CHECK(f.h_checked() == f.h(), "h_checked while host-authoritative");

    /* push to device, multiply by 3 on device, pull back */
    f.sync_device();
    auto dv = f.d();                       /* capture the View by value (device-safe) */
    Kokkos::parallel_for(name, N, KOKKOS_LAMBDA(const std::size_t i) { dv(i) = dv(i) * (T)3; });
    Kokkos::fence();
    f.modify_device();
    f.sync_host();

    int bad = 0;
    const T* h2 = f.h();
    for (std::size_t i = 0; i < N; ++i) {
        const T want = (T)((2 * i + 1) * 3);
        if (h2[i] != want) { if (bad < 3) std::printf("  [%zu] got %g want %g\n",
                                                       i, (double)h2[i], (double)want); ++bad; }
    }
    CHECK(bad == 0, "device round-trip identity");

    /* deep_copy path: copy the device view into a fresh View, mirror to host, compare */
    typename FieldType::dev_view_t dcopy(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "dcopy"), N);
    Kokkos::deep_copy(dcopy, f.d());
    auto hcopy = Kokkos::create_mirror_view(dcopy);
    Kokkos::deep_copy(hcopy, dcopy);
    int badc = 0;
    for (std::size_t i = 0; i < N; ++i) if (hcopy(i) != h2[i]) ++badc;
    CHECK(badc == 0, "deep_copy round-trip");

    f.free();
    CHECK(!f.allocated(), "freed");
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::printf("[test_field] backend: %s\n", Kokkos::DefaultExecutionSpace::name());
        roundtrip<fesom::Field,    real_t>("Field_real_t");   /* M16: Field is FieldT<real_t> */
        roundtrip<fesom::IntField, int   >("IntField_int");
    }
    Kokkos::finalize();
    if (g_failures) { std::printf("test_field: %d FAILURE(S)\n", g_failures); return 1; }
    std::printf("test_field: ALL PASS\n");
    return 0;
}
