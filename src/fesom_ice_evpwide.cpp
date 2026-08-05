/* E.EVP1 — FESOM_SPEED_EVPWIDE=K wide-halo EVP: discovery, build, exchange.
 * Design + proofs: docs/plans/20260720-m7-evpwide-design.md. Ghost kernel BODIES live in
 * fesom_ice_evp.cpp (same TU as the owned kernels — FMA-contraction byte parity, the cgpipe
 * argument). This file owns everything else and follows cgpipe_build/cgpipe_exchange_rr
 * (fesom_ssh.cpp) structurally: want-lists per OWNER, Alltoall count handshake, flat
 * per-partner device lists, pack -> fence -> Irecv/Isend -> Waitall -> unpack, prof hooks.
 *
 * Index spaces:
 *   node slots: [0, myDim) owned | [myDim, N) the dist-file halo (ring 1 + any extras)
 *               | [N, N+next) NEW extended slots (BFS rings 2..R, (dist, gid)-ascending).
 *   elements:   [0, E=myDim_elem2D) owned/shared (the kernels' loop today)
 *               | ghost list [0, Eg) with unified index E+eg (includes re-derived copies of
 *               any eDim/eXDim dist-file halo elements — their mesh storage has -1 vertex
 *               refs and is NOT used).
 *
 * gid -> owner: NEVER partit->part ranges (gids are not contiguous per rank — the session-11
 * cgpipe bug, fesom_ssh.cpp:703). The hook builds the TRUE owner vector once:
 * own[gid]=mype for owned, MPI_Allreduce(MAX). Ring-1 owners are cross-checked against the
 * com graph at build.
 */

#include "fesom_ice_evpwide.h"
#include "fesom_constants.h"
#include "fesom_halo.h"          /* prof hooks */
#include "fesom_ice_types.h"
#include "fesom_mesh.h"
#include "fesom_partit.h"
#include "fesom_speed.hpp"

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

static_assert(sizeof(real_t) == sizeof(double), "evpwide: real_t must be double (MPI_DOUBLE)");

namespace {

using DevV = fesom::Field::dev_view_t;

/* ---- module state ------------------------------------------------------- */
struct EvpwState {
    /* knob */
    int  env_cache = -2;          /* fesom_speed_int cache */
    bool announced = false;
    int  K = 0, R = 0;

    /* stage A: scatter-time stash (host) */
    bool hook_done = false;
    int  myDim = 0, eDim = 0, N = 0, E = 0;
    int  next = 0;                              /* extended slots */
    std::vector<int>         ext_gid1;          /* [next] 1-based gids */
    std::vector<int>         ext_owner;         /* [next] */
    std::vector<real_t>      ext_coord;         /* [2*next] rotated lon/lat (radians) */
    std::vector<real_t>      ext_geolat;        /* [next] geographic lat */
    std::vector<signed char> dist;              /* [N+next] BFS distance (127 = far) */
    std::vector<char>        gcoast;            /* [N+next] GLOBAL-edge coastal mask */
    std::vector<int>         halo_owner;        /* [eDim] owner of slot myDim+i (owner vector) */
    int  Eg = 0;
    std::vector<int> eg_gid1;                   /* [Eg] element gids, 1-based */
    std::vector<int> eg_vert;                   /* [3Eg] unified vertex slots */
    std::vector<int> eg_owner;                  /* [Eg] sigma owner (first-vertex owner) */
    long ring1_bfs = 0, ring1_extra = 0;        /* diagnostics: BFS ring-1 vs eDim */

    /* stage B: lazy build */
    bool built = false;
    FesomEvpwideDev dev;
    std::vector<int> partner;                   /* ascending ranks */
    std::vector<int> soff, roff;                /* [P+1] node-slot offsets */
    std::vector<int> esoff, eroff;              /* [P+1] elem-segment offsets (sigma refresh) */
    int nsend = 0, nrecv = 0, nesend = 0, nerecv = 0;
    Kokkos::View<int*>    sidx_d, ridx_d;       /* owned idx / slot idx */
    Kokkos::View<int*>    esidx_d, eridx_d;     /* owner-local elem idx / my ghost elem idx */
    Kokkos::View<char*>   selff_d;              /* [nrecv] selfcheck flag: dist(slot) <= K-1 */
    Kokkos::View<real_t*> sbuf_d, rbuf_d;       /* [nsend*11] / [nrecv*11] */
    Kokkos::View<real_t*> esbuf_d, erbuf_d;     /* [nesend*3] / [nerecv*3] */
    /* ── D1 / P0b: the FUSED form of the two-segment ship (cell ② and std EVP) ────────────
     * Same bytes, same pack work, ONE message per partner instead of two: the node segment and
     * the element-sigma segment are laid out back-to-back per partner in a single buffer.
     * Off by default; FESOM_SPEED_EVPWIDE_FUSE=1 selects it. Buffers and the per-entry partner
     * maps are only allocated when it is on. */
    int  fuse_cache = -2;
    bool fuse = false;
    Kokkos::View<real_t*> fsbuf_d, frbuf_d;     /* [nsend*11+nesend*3] / [nrecv*11+nerecv*3] */
    Kokkos::View<int*> spart_d, rpart_d;        /* [nsend]/[nrecv]  partner INDEX of each entry */
    Kokkos::View<int*> espart_d, erpart_d;      /* [nesend]/[nerecv] ditto, element segment */
    Kokkos::View<int*> soff_d, roff_d;          /* [P+1] device copies of the offsets */
    Kokkos::View<int*> esoff_d, eroff_d;
    std::vector<MPI_Request> reqs;
    int selfcheck = -1;                         /* env FESOM_EVPWIDE_SELFCHECK */
    long exch_count = 0;
    long msg_count = 0;                         /* MPI_Isend+Irecv POSTED (fused halves it) */
    double dbl_count = 0.0;                     /* doubles received (fusing must not change it) */
    /* Split by WHICH ship, because the two are criticised separately. S. Danilov's objection to
     * this design is that the per-step 11-field ship is far more than the 2 or 4 fields the
     * sub-cycle actually needs, and that it exists only to buy bit-identity. The counter-argument
     * is that it fires ONCE per step against 120/K window ships -- so the objection is about
     * field count and the answer is about frequency, and only the measured split settles it. */
    long msg_pre = 0, msg_win = 0;
    double dbl_pre = 0.0, dbl_win = 0.0;
    const real_t *gs_h = nullptr, *ea_h = nullptr, *mf_h = nullptr;   /* dump aids (host mesh) */
};
EvpwState g_w;

/* Geometry (gradient_sca/elem_area/metric_factor) and coriolis_node for the ghost zone are
 * SHIPPED from the owners at build — never recomputed locally. First NG5 run falsified the
 * "pure formula => byte-equal" assumption for anything transcendental: gcc -O3 vectorizes the
 * mesh's geometry loops (libmvec SIMD cos/tan), a scalar recompute in a different loop shape
 * differs in the last bit on unlucky inputs (CORE2's checked elements were lucky, NG5's
 * elem 48 was not — caught by the build cross-check). The cgpipe rule stands: ship owner
 * bytes for EVERYTHING the replay consumes. */

template <typename T>
Kokkos::View<T*> push_dev(const char *lbl, const std::vector<T> &v)
{
    Kokkos::View<T*> d(std::string(lbl), v.size());   /* std::string: the grow() idiom */
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
}

} /* anonymous namespace */

/* ---- knob resolve --------------------------------------------------------- */
int fesom_evpwide_env_K(void)
{
    return fesom_speed_int("EVPWIDE", 0, &g_w.env_cache);
}

/* D1 / P0b — FESOM_SPEED_EVPWIDE_FUSE=1: ship the node and element-sigma segments as ONE
 * message per partner. A pure TRANSPORT variant of cell ② (and of std EVP's window ship):
 * the same values are packed and unpacked in the same order, so it MUST be byte-identical to
 * the unfused form on every backend — that identity is the gate, and the message counter is
 * the measurement. It exists to separate the two terms H3 conflates: ②'s deficit against
 * cell ④ is consistent both with ②'s 2x bytes and with its 2x messages, and only removing
 * the message term by construction can say which. */
int fesom_evpwide_env_fuse(void)
{
    return fesom_speed_int("EVPWIDE_FUSE", 0, &g_w.fuse_cache);
}

int fesom_evpwide_next(void) { return g_w.hook_done ? g_w.next : 0; }

/* ---- stage A: scatter-time discovery -------------------------------------- */
void fesom_evpwide_mesh_hook(struct fesom_mesh *m, struct fesom_partit *p,
                             const int *node_g2l, const int *elem_g2l)
{
    const int K = fesom_evpwide_env_K();
    if (K <= 0 || p->npes <= 1) return;
    FESOM_CHECK(K <= 60, "evpwide: K=%d unreasonable", K);
    /* R = K with PER-WINDOW SIGMA REFRESH (design §2 as corrected by the first CORE2 gate):
     * velocity-only refresh can NOT be exact for ANY finite R — carried ghost sigma is never
     * re-baselined and the cleanliness recursion has no fixed point (outer-element staleness
     * reaches ring-1 from window 2 on; observed as the linear ~1e-5/window u_rhs drift).
     * Refreshing sigma on the ghost elements every exchange closes the induction at R = K. */
    int R = K;
    /* debug override: R > K widens the end-clean set to rings <= R-K, turning the drift
     * diagnostic into an EXACT-ZERO check there (e.g. K=2 R=3: ring-1 drift MUST be 0). */
    if (const char *rr = getenv("FESOM_EVPWIDE_RINGS")) { const int v = atoi(rr); if (v > R) R = v; }

    const int myDim = p->myDim_nod2D, eDim = p->eDim_nod2D;
    const int N = myDim + eDim;
    const int gN = m->nod2D, gE = m->elem2D;

    /* owner vector: own[gid0] = owning rank, via Allreduce(MAX) over -1 init. */
    std::vector<int> own((size_t)gN, -1);
    for (int i = 0; i < myDim; ++i) own[(size_t)(p->myList_nod2D[i] - 1)] = p->mype;
    MPI_Allreduce(MPI_IN_PLACE, own.data(), gN, MPI_INT, MPI_MAX, p->MPI_COMM_FESOM);
    long unowned = 0;
    for (int i = 0; i < gN; ++i) if (own[(size_t)i] < 0) ++unowned;
    FESOM_CHECK(unowned == 0, "evpwide: %ld global nodes have no owner", unowned);

    /* BFS rings by rounds over the GLOBAL element list (mark-sweep; no adjacency alloc).
     * dist 0 = owned; ring r = element-adjacent to ring r-1. */
    std::vector<signed char> gdist((size_t)gN, 127);
    for (int i = 0; i < myDim; ++i) gdist[(size_t)(p->myList_nod2D[i] - 1)] = 0;
    for (int r = 1; r <= R; ++r) {
        for (int e = 0; e < gE; ++e) {
            const int v0 = m->elem_nodes[3*e+0], v1 = m->elem_nodes[3*e+1], v2 = m->elem_nodes[3*e+2];
            const signed char d0 = gdist[(size_t)v0], d1 = gdist[(size_t)v1], d2 = gdist[(size_t)v2];
            signed char dmin = d0 < d1 ? d0 : d1; if (d2 < dmin) dmin = d2;
            if (dmin > (signed char)(r - 1)) continue;
            if (d0 == 127) gdist[(size_t)v0] = (signed char)r;
            if (d1 == 127) gdist[(size_t)v1] = (signed char)r;
            if (d2 == 127) gdist[(size_t)v2] = (signed char)r;
        }
    }

    /* GLOBAL-edge coastal mask (owner parity; asserted vs the local mask at build).
     * Boundary edges = 1-based gid > edge2D_in, i.e. 0-based idx >= edge2D_in. */
    std::vector<char> gmask((size_t)gN, 0);
    for (int e = m->edge2D_in; e < m->edge2D; ++e) {
        gmask[(size_t)m->edges[2*e + 0]] = 1;
        gmask[(size_t)m->edges[2*e + 1]] = 1;
    }

    /* extended node slots: dist in [2..R] and not already local; (dist, gid) ascending. */
    std::vector<int> extg;
    for (int g = 0; g < gN; ++g)
        if (gdist[(size_t)g] >= 2 && gdist[(size_t)g] <= R && node_g2l[g] < 0)
            extg.push_back(g);
    std::sort(extg.begin(), extg.end(), [&](int a, int b) {
        if (gdist[(size_t)a] != gdist[(size_t)b]) return gdist[(size_t)a] < gdist[(size_t)b];
        return a < b;
    });
    const int next = (int)extg.size();
    std::unordered_map<int, int> g2x;           /* gid0 -> ext ordinal */
    g2x.reserve((size_t)next * 2);
    for (int t = 0; t < next; ++t) g2x.emplace(extg[(size_t)t], t);

    g_w.myDim = myDim; g_w.eDim = eDim; g_w.N = N; g_w.E = p->myDim_elem2D;
    g_w.K = K; g_w.R = R; g_w.next = next;
    g_w.ext_gid1.resize((size_t)next);
    g_w.ext_owner.resize((size_t)next);
    g_w.ext_coord.resize((size_t)next * 2);
    g_w.ext_geolat.resize((size_t)next);
    for (int t = 0; t < next; ++t) {
        const int g = extg[(size_t)t];
        g_w.ext_gid1  [(size_t)t]     = g + 1;
        g_w.ext_owner [(size_t)t]     = own[(size_t)g];
        g_w.ext_coord [(size_t)2*t+0] = m->coord_nod2D[2*g + 0];
        g_w.ext_coord [(size_t)2*t+1] = m->coord_nod2D[2*g + 1];
        g_w.ext_geolat[(size_t)t]     = m->geo_coord_nod2D[2*g + 1];
    }

    /* dist + coastal for local slots [0,N) and ext slots; ring-1 owner stash. */
    g_w.dist.assign((size_t)(N + next), 127);
    g_w.gcoast.assign((size_t)(N + next), 0);
    for (int i = 0; i < N; ++i) {
        const int g = p->myList_nod2D[i] - 1;
        g_w.dist  [(size_t)i] = gdist[(size_t)g];
        g_w.gcoast[(size_t)i] = gmask[(size_t)g];
    }
    for (int t = 0; t < next; ++t) {
        g_w.dist  [(size_t)(N + t)] = gdist[(size_t)extg[(size_t)t]];
        g_w.gcoast[(size_t)(N + t)] = gmask[(size_t)extg[(size_t)t]];
    }
    g_w.halo_owner.resize((size_t)eDim);
    for (int i = 0; i < eDim; ++i)
        g_w.halo_owner[(size_t)i] = own[(size_t)(p->myList_nod2D[myDim + i] - 1)];

    /* BFS ring-1 vs the dist-file eDim halo (diagnostic; extras stay refreshed-but-frozen). */
    long bfs1 = 0, extra = 0;
    for (int g = 0; g < gN; ++g) if (gdist[(size_t)g] == 1) ++bfs1;
    for (int i = 0; i < eDim; ++i) if (g_w.dist[(size_t)(myDim + i)] != 1) ++extra;
    g_w.ring1_bfs = bfs1; g_w.ring1_extra = extra;

    /* ghost elements: any vertex with dist <= R-1, not in the owned/shared block [0, E).
     * (dist-file eDim/eXDim halo elements re-derived here on purpose — their mesh vertex
     * refs contain -1.) Vertex slots resolved to the unified space now. Each ghost element
     * gets a deterministic SIGMA OWNER = owner of its FIRST vertex (global elem_nodes order
     * is global) — that rank holds it as a maxring<=1 element with owner-clean sigma. */
    std::vector<int> egid1, evert, eown;
    for (int e = 0; e < gE; ++e) {
        const int v[3] = { m->elem_nodes[3*e+0], m->elem_nodes[3*e+1], m->elem_nodes[3*e+2] };
        signed char dmin = 127;
        for (int k = 0; k < 3; ++k) if (gdist[(size_t)v[k]] < dmin) dmin = gdist[(size_t)v[k]];
        if (dmin > (signed char)(R - 1)) continue;
        const int le = elem_g2l[e];
        if (le >= 0 && le < p->myDim_elem2D) continue;          /* owned/shared: kernels have it */
        int slots[3];
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            if (node_g2l[v[k]] >= 0) slots[k] = node_g2l[v[k]];
            else {
                auto it = g2x.find(v[k]);
                if (it == g2x.end()) { ok = false; break; }     /* vertex beyond ring R */
                slots[k] = N + it->second;
            }
        }
        FESOM_CHECK(ok, "evpwide: ghost elem gid=%d has a vertex beyond ring R=%d "
                        "(element-ring/node-ring inconsistency)", e + 1, R);
        egid1.push_back(e + 1);
        eown.push_back(own[(size_t)v[0]]);
        for (int k = 0; k < 3; ++k) evert.push_back(slots[k]);
    }
    g_w.Eg = (int)egid1.size();
    g_w.eg_gid1  = std::move(egid1);
    g_w.eg_vert  = std::move(evert);
    g_w.eg_owner = std::move(eown);
    g_w.hook_done = true;

    long loc[4] = { (long)next, (long)g_w.Eg, extra, bfs1 };
    long mx[4];
    MPI_Reduce(loc, mx, 4, MPI_LONG, MPI_MAX, 0, p->MPI_COMM_FESOM);
    if (p->mype == 0)
        fprintf(stderr, "[evpwide] hook: K=%d R=%d ext-nodes(max)=%ld ghost-elems(max)=%ld "
                        "eDim-extras(max)=%ld ring1-bfs(max)=%ld\n",
                K, R, mx[0], mx[1], mx[2], mx[3]);
}

/* ---- stage B: lazy collective build ---------------------------------------- */
static void evpw_build(struct fesom_ice *ice, struct fesom_partit *p, struct fesom_mesh *m)
{
    (void)ice;
    EvpwState &S = g_w;
    const int myDim = S.myDim, N = S.N, next = S.next, E = S.E, Eg = S.Eg, K = S.K, R = S.R;
    MPI_Comm comm = p->MPI_COMM_FESOM;
    const int npes = p->npes;

    FESOM_CHECK(m->myDim_nod2D == myDim && m->eDim_nod2D == S.eDim && m->myDim_elem2D == E,
                "evpwide: mesh dims changed between hook and build");

    /* ulevels==1 audit (this port is non-cavity, fesom_mesh.cpp:547; the ghost kernels omit
     * the cavity guards on that basis — abort loudly if that ever changes). */
    for (int e = 0; e < E; ++e)
        FESOM_CHECK(m->ulevels[e] == 1, "evpwide: ulevels!=1 (cavity?) — ghost kernels assume 1");
    for (int n = 0; n < N; ++n)
        FESOM_CHECK(m->ulevels_nod2D[n] == 1, "evpwide: ulevels_nod2D!=1 — unsupported");

    /* ring-1 owner cross-check: com graph vs the owner vector. */
    {
        const fesom_com_struct *cs = &p->com_nod2D;
        std::vector<int> owner_l((size_t)S.eDim, -1);
        for (int k = 0; k < cs->rPEnum; ++k)
            for (int j = cs->rptr[k] - 1; j < cs->rptr[k + 1] - 1; ++j)
                owner_l[(size_t)(cs->rlist[j] - 1 - myDim)] = cs->rPE[k];
        for (int i = 0; i < S.eDim; ++i)
            FESOM_CHECK(owner_l[(size_t)i] == S.halo_owner[(size_t)i],
                        "evpwide: ring-1 owner mismatch com=%d vec=%d (slot %d)",
                        owner_l[(size_t)i], S.halo_owner[(size_t)i], myDim + i);
    }

    /* coastal-mask parity on OWNED nodes: global-edge mask == local owned-boundary-edge mask
     * (the evp_coastal_mask semantics, fesom_ice_evp.cpp:493). */
    {
        std::vector<char> lmask((size_t)N, 0);
        for (int ed = 0; ed < m->myDim_edge2D; ++ed) {
            if (p->myList_edge2D[ed] <= m->edge2D_in) continue;
            lmask[(size_t)m->edges[ed*2 + 0]] = 1;
            lmask[(size_t)m->edges[ed*2 + 1]] = 1;
        }
        for (int n = 0; n < myDim; ++n)
            FESOM_CHECK(lmask[(size_t)n] == S.gcoast[(size_t)n],
                        "evpwide: coastal mask parity broken at owned node %d (local=%d global=%d)",
                        n, (int)lmask[(size_t)n], (int)S.gcoast[(size_t)n]);
    }

    /* unified coord accessor: slot < N -> local mesh coords (same global doubles), else stash. */
    auto slot_xy = [&](int s, real_t *x, real_t *y) {
        if (s < N) { *x = m->coord_nod2D[2*s + 0]; *y = m->coord_nod2D[2*s + 1]; }
        else       { *x = S.ext_coord[(size_t)2*(s - N) + 0]; *y = S.ext_coord[(size_t)2*(s - N) + 1]; }
    };

    /* orient_cw REPLAY (fesom_mesh.cpp:460): the mesh orients its LOCAL elements CW AFTER
     * scatter_mesh, but the hook stashed the PRE-orientation global vertex order. Replicate
     * the exact swap rule (pure function of vertex coords) on the ghost elements BEFORE any
     * geometry / kk-slot mapping, so eps/sigma arithmetic matches the owner's post-swap
     * order. (The first ice-active smoke caught this: ~half the ghost elements had inverted
     * orientation => wrong-signed gradients => sigma diverged from window 1.) */
    {
        const real_t cyc      = FESOM_CYCLIC_LENGTH_RAD;
        const real_t half_cyc = 0.5 * cyc;
        long nsw = 0;
        for (int eg = 0; eg < Eg; ++eg) {
            int *v = &S.eg_vert[(size_t)3*eg];
            real_t ax, ay, bx1, by1, cx1, cy1;
            slot_xy(v[0], &ax,  &ay);
            slot_xy(v[1], &bx1, &by1);
            slot_xy(v[2], &cx1, &cy1);
            real_t bx = bx1 - ax, by = by1 - ay;
            real_t cx = cx1 - ax, cy = cy1 - ay;
            if (bx >  half_cyc) bx -= cyc;
            if (bx < -half_cyc) bx += cyc;
            if (cx >  half_cyc) cx -= cyc;
            if (cx < -half_cyc) cx += cyc;
            if (bx * cy - by * cx > 0.0) { const int t = v[1]; v[1] = v[2]; v[2] = t; ++nsw; }
        }
        long mxsw = 0;
        MPI_Reduce(&nsw, &mxsw, 1, MPI_LONG, MPI_MAX, 0, comm);
        if (p->mype == 0)
            fprintf(stderr, "[evpwide] ghost orient_cw replay: swapped(max) %ld ghost elems\n", mxsw);
    }

    /* ghost geometry is SHIPPED (see the banner at the top of this file: the libmvec
     * vectorization lesson) — filled from the element-owner reply (tag 2206) below. */
    std::vector<real_t> gs_g((size_t)Eg * 6), ea_g((size_t)Eg), mf_g((size_t)Eg);

    /* -------- handshake: wants grouped by owner; replies carry (area0, coriolis_node) +
     * owner-order incident-element gid lists for updatable slots. Tags 2201/2202/2203. */
    const int nGhost = S.eDim + next;                 /* all refreshed slots */
    std::vector<int> slot_of((size_t)nGhost), owner_of((size_t)nGhost);
    for (int i = 0; i < S.eDim; ++i) { slot_of[(size_t)i] = myDim + i; owner_of[(size_t)i] = S.halo_owner[(size_t)i]; }
    for (int t = 0; t < next; ++t)   { slot_of[(size_t)(S.eDim + t)] = N + t; owner_of[(size_t)(S.eDim + t)] = S.ext_owner[(size_t)t]; }

    auto upd_of_slot = [&](int s) { return S.dist[(size_t)s] >= 1 && S.dist[(size_t)s] <= (signed char)(R - 1); };

    std::vector<std::vector<int>> want((size_t)npes);        /* per owner: [gid1, need]* */
    std::vector<std::vector<int>> wslot((size_t)npes);       /* my slots, same order */
    for (int i = 0; i < nGhost; ++i) {                       /* ascending slot => deterministic */
        const int s = slot_of[(size_t)i], o = owner_of[(size_t)i];
        const int gid1 = (s < N) ? p->myList_nod2D[s] : S.ext_gid1[(size_t)(s - N)];
        want[(size_t)o].push_back(gid1);
        want[(size_t)o].push_back(upd_of_slot(s) ? 1 : 0);
        wslot[(size_t)o].push_back(s);
    }

    std::vector<int> scnt((size_t)npes, 0), rcnt((size_t)npes, 0);
    for (int q = 0; q < npes; ++q) scnt[(size_t)q] = (int)want[(size_t)q].size();
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, comm);

    std::vector<std::vector<int>> wantin((size_t)npes);
    std::vector<MPI_Request> rq;
    for (int q = 0; q < npes; ++q) {
        if (rcnt[(size_t)q] > 0) {
            wantin[(size_t)q].resize((size_t)rcnt[(size_t)q]);
            rq.push_back(MPI_Request());
            MPI_Irecv(wantin[(size_t)q].data(), rcnt[(size_t)q], MPI_INT, q, 2201, comm, &rq.back());
        }
        if (scnt[(size_t)q] > 0) {
            rq.push_back(MPI_Request());
            MPI_Isend(want[(size_t)q].data(), scnt[(size_t)q], MPI_INT, q, 2201, comm, &rq.back());
        }
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    rq.clear();

    /* owner side: owned gid1 -> local; node->elem adjacency in LOCAL ELEMENT ORDER
     * (== the owner's Serial scatter order — the whole point). */
    std::unordered_map<int, int> own_g2l;
    own_g2l.reserve((size_t)myDim * 2);
    for (int i = 0; i < myDim; ++i) own_g2l.emplace(p->myList_nod2D[i], i);
    std::vector<int> adj_ptr((size_t)myDim + 1, 0);
    for (int e = 0; e < E; ++e)
        for (int k = 0; k < 3; ++k) {
            const int n = m->elem_nodes[3*e + k];
            if (n < myDim) ++adj_ptr[(size_t)n + 1];
        }
    for (int n = 0; n < myDim; ++n) adj_ptr[(size_t)n + 1] += adj_ptr[(size_t)n];
    std::vector<int> adj_e(adj_ptr[(size_t)myDim]);
    {
        std::vector<int> cur(adj_ptr.begin(), adj_ptr.end() - 1);
        for (int e = 0; e < E; ++e)                       /* ascending e => owner order */
            for (int k = 0; k < 3; ++k) {
                const int n = m->elem_nodes[3*e + k];
                if (n < myDim) adj_e[(size_t)cur[(size_t)n]++] = e;
            }
    }

    /* replies + the runtime SEND lists (owned indices, requester's want order).
     * M9: the per-want double reply is (area0, coriolis, bc_index) — THREE doubles, was two.
     * bc_index_nod2D joined it because the mEVP node solve multiplies its determinant by it
     * (trap 9) and it is built from LOCAL edges (fesom_ice.cpp:314), so a ghost slot cannot
     * compute its own. It is time-invariant, so shipping it here — once, at build — costs one
     * double per ghost slot forever instead of one per step. */
    std::vector<std::vector<real_t>> rep_d((size_t)npes);    /* (area0, cor, bc_index) per want */
    std::vector<std::vector<int>>    rep_i((size_t)npes);    /* per flagged want: cnt, egid1... */
    std::vector<std::vector<int>>    slist((size_t)npes);
    const int nl = m->nl;
    /* Built unconditionally in fesom_ice_init (fesom_ice.cpp:308), i.e. long before this lazy
     * build runs. If it is ever NULL we would ship garbage into the ghost determinant and the
     * damage would be invisible on Serial; abort instead of defaulting. */
    FESOM_CHECK(m->bc_index_nod2D != NULL,
                "evpwide: bc_index_nod2D not built before the wide-halo build");
    for (int q = 0; q < npes; ++q) {
        const std::vector<int> &win = wantin[(size_t)q];
        for (size_t j = 0; j + 1 < win.size(); j += 2) {
            const int gid1 = win[j], need = win[j + 1];
            auto it = own_g2l.find(gid1);
            FESOM_CHECK(it != own_g2l.end(), "evpwide: rank %d wants gid %d I do not own",
                        q, gid1);
            const int n = it->second;
            slist[(size_t)q].push_back(n);
            rep_d[(size_t)q].push_back(m->area[(size_t)n * (size_t)nl + 0]);
            rep_d[(size_t)q].push_back(m->coriolis_node[n]);   /* shipped, not recomputed */
            rep_d[(size_t)q].push_back(m->bc_index_nod2D[n]);
            if (need) {
                rep_i[(size_t)q].push_back(adj_ptr[(size_t)n + 1] - adj_ptr[(size_t)n]);
                for (int a = adj_ptr[(size_t)n]; a < adj_ptr[(size_t)n + 1]; ++a)
                    rep_i[(size_t)q].push_back(p->myList_elem2D[adj_e[(size_t)a]]);
            }
        }
    }

    /* reply counts via Alltoall (int payload length; the dbl length = my want count). */
    std::vector<int> sicnt((size_t)npes, 0), ricnt((size_t)npes, 0);
    for (int q = 0; q < npes; ++q) sicnt[(size_t)q] = (int)rep_i[(size_t)q].size();
    MPI_Alltoall(sicnt.data(), 1, MPI_INT, ricnt.data(), 1, MPI_INT, comm);

    std::vector<std::vector<real_t>> ind((size_t)npes);
    std::vector<std::vector<int>>    ini((size_t)npes);
    for (int q = 0; q < npes; ++q) {
        const int nw = (int)wslot[(size_t)q].size();
        if (nw > 0) {
            ind[(size_t)q].resize((size_t)nw * 3);            /* (area0, cor, bc_index) per want */
            rq.push_back(MPI_Request());
            MPI_Irecv(ind[(size_t)q].data(), nw * 3, MPI_DOUBLE, q, 2202, comm, &rq.back());
        }
        if (ricnt[(size_t)q] > 0) {
            ini[(size_t)q].resize((size_t)ricnt[(size_t)q]);
            rq.push_back(MPI_Request());
            MPI_Irecv(ini[(size_t)q].data(), ricnt[(size_t)q], MPI_INT, q, 2203, comm, &rq.back());
        }
        if (!rep_d[(size_t)q].empty()) {
            rq.push_back(MPI_Request());
            MPI_Isend(rep_d[(size_t)q].data(), (int)rep_d[(size_t)q].size(), MPI_DOUBLE, q, 2202, comm, &rq.back());
        }
        if (!rep_i[(size_t)q].empty()) {
            rq.push_back(MPI_Request());
            MPI_Isend(rep_i[(size_t)q].data(), (int)rep_i[(size_t)q].size(), MPI_INT, q, 2203, comm, &rq.back());
        }
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    rq.clear();

    /* -------- element (sigma-refresh) handshake: want ghost-element sigma from each
     * element's deterministic owner (first-vertex owner — holds it maxring<=1, sigma clean).
     * Tags 2204. Order per owner: ascending my ghost index => deterministic both sides. */
    std::vector<std::vector<int>> wantE((size_t)npes), ewslot((size_t)npes);
    for (int eg = 0; eg < Eg; ++eg) {
        const int o = S.eg_owner[(size_t)eg];
        wantE[(size_t)o].push_back(S.eg_gid1[(size_t)eg]);
        ewslot[(size_t)o].push_back(eg);
    }
    std::vector<int> escnt((size_t)npes, 0), ercnt((size_t)npes, 0);
    for (int q = 0; q < npes; ++q) escnt[(size_t)q] = (int)wantE[(size_t)q].size();
    MPI_Alltoall(escnt.data(), 1, MPI_INT, ercnt.data(), 1, MPI_INT, comm);
    std::vector<std::vector<int>> wantEin((size_t)npes);
    for (int q = 0; q < npes; ++q) {
        if (ercnt[(size_t)q] > 0) {
            wantEin[(size_t)q].resize((size_t)ercnt[(size_t)q]);
            rq.push_back(MPI_Request());
            MPI_Irecv(wantEin[(size_t)q].data(), ercnt[(size_t)q], MPI_INT, q, 2204, comm, &rq.back());
        }
        if (escnt[(size_t)q] > 0) {
            rq.push_back(MPI_Request());
            MPI_Isend(wantE[(size_t)q].data(), escnt[(size_t)q], MPI_INT, q, 2204, comm, &rq.back());
        }
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    rq.clear();
    std::vector<std::vector<int>> eslist((size_t)npes);      /* my local elem idx, requester order */
    {
        std::unordered_map<int, int> own_e2l;
        own_e2l.reserve((size_t)E * 2);
        for (int e = 0; e < E; ++e) own_e2l.emplace(p->myList_elem2D[e], e);
        for (int q = 0; q < npes; ++q) {
            eslist[(size_t)q].reserve(wantEin[(size_t)q].size());
            for (int gid1 : wantEin[(size_t)q]) {
                auto it = own_e2l.find(gid1);
                FESOM_CHECK(it != own_e2l.end(),
                            "evpwide: rank %d wants sigma of elem gid %d that is not my "
                            "myDim_elem2D (first-vertex-owner invariant broken)", q, gid1);
                eslist[(size_t)q].push_back(it->second);
            }
        }
    }
    /* one-time GEOMETRY reply per wanted element (tag 2206): [gs0..gs5, elem_area,
     * metric_factor] — the OWNER's mesh bytes, exactly what its kernels use. */
    {
        std::vector<std::vector<real_t>> rep_e((size_t)npes), ine((size_t)npes);
        for (int q = 0; q < npes; ++q) {
            rep_e[(size_t)q].reserve(eslist[(size_t)q].size() * 8);
            for (int e : eslist[(size_t)q]) {
                for (int g = 0; g < 6; ++g) rep_e[(size_t)q].push_back(m->gradient_sca[6*e + g]);
                rep_e[(size_t)q].push_back(m->elem_area[e]);
                rep_e[(size_t)q].push_back(m->metric_factor[e]);
            }
        }
        for (int q = 0; q < npes; ++q) {
            const int nwE = (int)ewslot[(size_t)q].size();
            if (nwE > 0) {
                ine[(size_t)q].resize((size_t)nwE * 8);
                rq.push_back(MPI_Request());
                MPI_Irecv(ine[(size_t)q].data(), nwE * 8, MPI_DOUBLE, q, 2206, comm, &rq.back());
            }
            if (!rep_e[(size_t)q].empty()) {
                rq.push_back(MPI_Request());
                MPI_Isend(rep_e[(size_t)q].data(), (int)rep_e[(size_t)q].size(), MPI_DOUBLE, q, 2206,
                          comm, &rq.back());
            }
        }
        MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
        rq.clear();
        for (int q = 0; q < npes; ++q)
            for (size_t j = 0; j < ewslot[(size_t)q].size(); ++j) {
                const int eg = ewslot[(size_t)q][j];
                for (int g = 0; g < 6; ++g) gs_g[(size_t)6*eg + g] = ine[(size_t)q][8*j + g];
                ea_g[(size_t)eg] = ine[(size_t)q][8*j + 6];
                mf_g[(size_t)eg] = ine[(size_t)q][8*j + 7];
                FESOM_CHECK(ea_g[(size_t)eg] > 0.0, "evpwide: shipped elem_area <= 0 (ghost %d)", eg);
            }
    }

    /* -------- unified statics + gather CSR. */
    std::vector<real_t> area0x((size_t)(N + next), 0.0), corx((size_t)(N + next), 0.0);
    std::vector<real_t> bcx((size_t)(N + next), 0.0);
    std::vector<int>    coastx((size_t)(N + next), 0);
    for (int n = 0; n < N; ++n) {
        area0x[(size_t)n] = m->area[(size_t)n * (size_t)nl + 0];
        corx  [(size_t)n] = m->coriolis_node[n];
        bcx   [(size_t)n] = m->bc_index_nod2D[n];
        coastx[(size_t)n] = S.gcoast[(size_t)n];
    }
    for (int t = 0; t < next; ++t)
        coastx[(size_t)(N + t)] = S.gcoast[(size_t)(N + t)];
    /* corx for ALL ghost slots (>= myDim) is overwritten by the shipped owner bytes in the
     * reply-consumption walk below — never recomputed (libmvec lesson). */

    std::unordered_map<int, int> eg2u;                 /* elem gid1 -> unified idx */
    eg2u.reserve((size_t)(E + Eg) * 2);
    for (int e = 0; e < E; ++e) eg2u.emplace(p->myList_elem2D[e], e);
    for (int eg = 0; eg < Eg; ++eg) eg2u.emplace(S.eg_gid1[(size_t)eg], E + eg);

    std::vector<int> gath_ptr((size_t)(N + next) + 1, 0);
    std::vector<int> gath_elem, gath_k;
    std::vector<int> upd_slots;
    {
        /* Walk my wants per owner in want order, consuming the area0 + adjacency streams.
         * Adjacency is staged PER SLOT first — the owner-walk order interleaves slots of
         * different owners, so pushing straight into a prefix-summed CSR would scramble rows. */
        double area0_maxdiff = 0.0;
        long   bc_mismatch = 0;   /* ring-1 slots where MY local bc_index != the owner's */
        std::vector<std::vector<int>> rowE((size_t)(N + next));
        for (int q = 0; q < npes; ++q) {
            size_t ip = 0;
            const std::vector<int> &sl = wslot[(size_t)q];
            for (size_t j = 0; j < sl.size(); ++j) {
                const int s = sl[j];
                /* OWNER bytes win at EVERY ghost slot — including ring 1: the local halo
                 * mesh->area was summed in MY nod_in_elem2D order (compute_node_areas runs
                 * over interior+halo locally), so it differs from the owner's in last bits.
                 * The replay must use what the OWNER's Step 2 used. (First smoke caught
                 * exactly this: max ulp-level diffs on ring-1 cluster areas.) */
                if (s < N && ind[(size_t)q][3*j] != area0x[(size_t)s]) {
                    const double d = fabs(ind[(size_t)q][3*j] - area0x[(size_t)s]);
                    if (d > area0_maxdiff) area0_maxdiff = d;
                }
                area0x[(size_t)s] = ind[(size_t)q][3*j];
                corx  [(size_t)s] = ind[(size_t)q][3*j + 1];   /* owner bytes (libmvec lesson) */
                if (s < N && ind[(size_t)q][3*j + 2] != bcx[(size_t)s]) ++bc_mismatch;
                bcx   [(size_t)s] = ind[(size_t)q][3*j + 2];   /* owner bytes (local-edge derived) */
                if (!upd_of_slot(s)) continue;
                FESOM_CHECK(ip < ini[(size_t)q].size(), "evpwide: adjacency stream underrun (rank %d)", q);
                const int cnt = ini[(size_t)q][ip++];
                FESOM_CHECK(cnt > 0, "evpwide: owner reports 0 incident elements for slot %d", s);
                rowE[(size_t)s].reserve((size_t)cnt);
                for (int c = 0; c < cnt; ++c) {
                    const int egid1 = ini[(size_t)q][ip++];
                    auto it = eg2u.find(egid1);
                    FESOM_CHECK(it != eg2u.end(),
                                "evpwide: owner-order elem gid %d not in my unified zone (slot %d)",
                                egid1, s);
                    rowE[(size_t)s].push_back(it->second);   /* OWNER's element order kept */
                }
                upd_slots.push_back(s);
            }
            FESOM_CHECK(ip == ini[(size_t)q].size(), "evpwide: adjacency stream overrun (rank %d)", q);
        }
        for (size_t s = 0; s < (size_t)(N + next); ++s)
            gath_ptr[s + 1] = gath_ptr[s] + (int)rowE[s].size();
        gath_elem.reserve((size_t)gath_ptr[(size_t)(N + next)]);
        gath_k.reserve((size_t)gath_ptr[(size_t)(N + next)]);
        for (size_t s = 0; s < (size_t)(N + next); ++s) {
            for (int ue : rowE[s]) {
                int ks = -1;
                for (int k = 0; k < 3; ++k) {
                    const int vs = (ue < E) ? m->elem_nodes[3*ue + k]
                                            : S.eg_vert[(size_t)3*(ue - E) + k];
                    if (vs == (int)s) { ks = k; break; }
                }
                FESOM_CHECK(ks >= 0, "evpwide: slot %zu not a vertex of unified elem %d", s, ue);
                gath_elem.push_back(ue);
                gath_k.push_back(ks);
            }
        }
        std::sort(upd_slots.begin(), upd_slots.end());
        double a0mx = 0.0;
        MPI_Reduce(&area0_maxdiff, &a0mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        long bcsum = 0;
        MPI_Reduce(&bc_mismatch, &bcsum, 1, MPI_LONG, MPI_SUM, 0, comm);
        if (p->mype == 0) {
            fprintf(stderr, "[evpwide] ring-1 area0: owner-vs-local max|diff| = %.3e "
                            "(expected ulp-level; owner bytes adopted for the replay)\n", a0mx);
            /* NOT an error either way. bc_index is derived from LOCAL edges, so a rank that does
             * not own the boundary edge cannot see the mask — a nonzero count is the positive
             * evidence that shipping it was necessary; a zero count on a partition whose ring 1
             * happens to miss every coast is equally fine. Printed so the decision is auditable
             * rather than assumed. */
            fprintf(stderr, "[evpwide] ring-1 bc_index: %ld slot(s) where local != owner "
                            "(owner bytes adopted; ship-once, time-invariant)\n", bcsum);
        }
    }

    /* -------- runtime exchange lists (cgpipe shape): ascending partner ranks; the elem
     * (sigma) segments ride the SAME partner set (union). */
    S.partner.clear(); S.soff.assign(1, 0); S.roff.assign(1, 0);
    S.esoff.assign(1, 0); S.eroff.assign(1, 0);
    std::vector<int> sidx, ridx, esidx, eridx;
    std::vector<char> selff;
    for (int q = 0; q < npes; ++q) {
        const bool has = !slist[(size_t)q].empty() || !wslot[(size_t)q].empty()
                      || !eslist[(size_t)q].empty() || !ewslot[(size_t)q].empty();
        if (!has) continue;
        S.partner.push_back(q);
        for (int n : slist[(size_t)q]) sidx.push_back(n);
        for (int s : wslot[(size_t)q]) {
            ridx.push_back(s);
            char fl = S.dist[(size_t)s] <= (signed char)(S.R - K) ? 1 : 0;   /* end-clean set */
            if (fl) {   /* debug split: 2 = the gather row touches a GHOST element */
                for (int a = gath_ptr[(size_t)s]; a < gath_ptr[(size_t)s + 1]; ++a)
                    if (gath_elem[(size_t)a] >= E) { fl = 2; break; }
            }
            selff.push_back(fl);
        }
        for (int e : eslist[(size_t)q]) esidx.push_back(e);
        for (int eg : ewslot[(size_t)q]) eridx.push_back(eg);
        S.soff.push_back((int)sidx.size());
        S.roff.push_back((int)ridx.size());
        S.esoff.push_back((int)esidx.size());
        S.eroff.push_back((int)eridx.size());
    }
    S.nsend = (int)sidx.size();
    S.nrecv = (int)ridx.size();
    S.nesend = (int)esidx.size();
    S.nerecv = (int)eridx.size();
    S.reqs.assign(4 * S.partner.size(), MPI_Request());

    /* -------- device pushes. */
    S.sidx_d  = push_dev("evpw.sidx", sidx);
    S.ridx_d  = push_dev("evpw.ridx", ridx);
    S.esidx_d = push_dev("evpw.esidx", esidx);
    S.eridx_d = push_dev("evpw.eridx", eridx);
    S.selff_d = push_dev("evpw.selff", selff);
    S.sbuf_d  = Kokkos::View<real_t*>("evpw.sbuf", (size_t)S.nsend * 11);
    S.rbuf_d  = Kokkos::View<real_t*>("evpw.rbuf", (size_t)S.nrecv * 11);
    S.esbuf_d = Kokkos::View<real_t*>("evpw.esbuf", (size_t)S.nesend * 3);
    S.erbuf_d = Kokkos::View<real_t*>("evpw.erbuf", (size_t)S.nerecv * 3);

    /* -------- D1 / P0b: the fused-transport maps. Built ONLY when the knob is on, so the
     * default form carries no extra memory and no extra device arrays whose mere existence
     * could be mistaken for the variant running. Layout per partner q, send side:
     *     base(q)      = soff[q]*nf + esoff[q]*3
     *     node entry i = base(q) + (i - soff[q])*nf + f
     *     elem entry j = base(q) + (soff[q+1]-soff[q])*nf + (j - esoff[q])*3 + c
     * i.e. exactly the unfused segments, concatenated per partner. nf is a plan property, so
     * the offsets are recomputed per exchange in the kernel from soff/esoff rather than
     * baked in — one int multiply per entry, and it keeps the maps valid for every plan. */
    S.fuse = (fesom_evpwide_env_fuse() != 0);
    if (S.fuse) {
        std::vector<int> spart((size_t)S.nsend), rpart((size_t)S.nrecv);
        std::vector<int> espart((size_t)S.nesend), erpart((size_t)S.nerecv);
        for (size_t q = 0; q < S.partner.size(); ++q) {
            for (int i = S.soff[q];  i < S.soff[q + 1];  ++i) spart[(size_t)i]  = (int)q;
            for (int i = S.roff[q];  i < S.roff[q + 1];  ++i) rpart[(size_t)i]  = (int)q;
            for (int i = S.esoff[q]; i < S.esoff[q + 1]; ++i) espart[(size_t)i] = (int)q;
            for (int i = S.eroff[q]; i < S.eroff[q + 1]; ++i) erpart[(size_t)i] = (int)q;
        }
        S.spart_d  = push_dev("evpw.spart", spart);
        S.rpart_d  = push_dev("evpw.rpart", rpart);
        S.espart_d = push_dev("evpw.espart", espart);
        S.erpart_d = push_dev("evpw.erpart", erpart);
        S.soff_d   = push_dev("evpw.soffd", S.soff);
        S.roff_d   = push_dev("evpw.roffd", S.roff);
        S.esoff_d  = push_dev("evpw.esoffd", S.esoff);
        S.eroff_d  = push_dev("evpw.eroffd", S.eroff);
        S.fsbuf_d  = Kokkos::View<real_t*>("evpw.fsbuf",
                                           (size_t)S.nsend * 11 + (size_t)S.nesend * 3);
        S.frbuf_d  = Kokkos::View<real_t*>("evpw.frbuf",
                                           (size_t)S.nrecv * 11 + (size_t)S.nerecv * 3);
    }

    FesomEvpwideDev &D = S.dev;
    D.K = K; D.R = R; D.next = next; D.Eg = Eg; D.nUpd = (int)upd_slots.size();
    D.upd_slots = push_dev("evpw.upd", upd_slots);
    D.gath_ptr  = push_dev("evpw.gptr", gath_ptr);
    D.gath_elem = push_dev("evpw.gelem", gath_elem);
    D.gath_k    = push_dev("evpw.gk", gath_k);
    D.en_g      = push_dev("evpw.eng", S.eg_vert);
    D.gs_g      = push_dev("evpw.gsg", gs_g);
    D.ea_g      = push_dev("evpw.eag", ea_g);
    D.mf_g      = push_dev("evpw.mfg", mf_g);
    D.istr_g    = Kokkos::View<real_t*>("evpw.istrg", (size_t)Eg);
    D.s11_g     = Kokkos::View<real_t*>("evpw.s11g", (size_t)Eg);   /* zero-init = cold start */
    D.s12_g     = Kokkos::View<real_t*>("evpw.s12g", (size_t)Eg);
    D.s22_g     = Kokkos::View<real_t*>("evpw.s22g", (size_t)Eg);
    D.area0x    = push_dev("evpw.area0x", area0x);
    D.corx      = push_dev("evpw.corx", corx);
    D.coastx    = push_dev("evpw.coastx", coastx);
    /* M9 mEVP: per-step ghost element maps (twins of istr_g) + the shipped boundary mask.
     * Allocated unconditionally rather than behind a mEVP-mode flag: pfac_g/ice_el_g are [Eg]
     * and bcx is [N+next], i.e. ghost-zone sized and negligible next to what is already here,
     * and a mode flag would add a branch whose OFF side is exactly the state a bug would fake. */
    D.pfac_g    = Kokkos::View<real_t*>("evpw.pfacg", (size_t)Eg);
    D.ice_el_g  = Kokkos::View<int*>   ("evpw.iceelg", (size_t)Eg);
    D.bcx       = push_dev("evpw.bcx", bcx);

    S.selfcheck = 0;
    if (const char *sc = getenv("FESOM_EVPWIDE_SELFCHECK")) S.selfcheck = atoi(sc);
    D.dbg = (S.selfcheck >= 2) ? 1 : 0;
    S.gs_h = m->gradient_sca; S.ea_h = m->elem_area; S.mf_h = m->metric_factor;

    long loc[6] = { (long)S.partner.size(), (long)S.nrecv, (long)S.nsend, (long)D.nUpd,
                    (long)(S.roff.empty() ? 0 : *std::max_element(S.roff.begin(), S.roff.end())),
                    (long)Eg };
    long mx[6];
    MPI_Reduce(loc, mx, 6, MPI_LONG, MPI_MAX, 0, comm);
    if (p->mype == 0)
        fprintf(stderr, "[evpwide] built: partners(max)=%ld recv-slots(max)=%ld send-slots(max)=%ld "
                        "upd-slots(max)=%ld ghost-elems(max)=%ld — wide %s ACTIVE (K=%d, R=%d, "
                        "msgs/step %d -> %d+1, widest msg ~%ld KB @2f)\n",
                mx[0], mx[1], mx[2], mx[3], mx[5],
                ice->whichEVP == 1 ? "mEVP" : "EVP", K, R,
                ice->evp_rheol_steps, ice->evp_rheol_steps / K,
                (long)(mx[4] * 2 * 8 / 1024));
    S.built = true;
}

/* ---- runtime resolve ------------------------------------------------------- */
int fesom_evpwide_K(struct fesom_ice *ice, struct fesom_partit *p, struct fesom_mesh *m)
{
    const int K = fesom_evpwide_env_K();
    if (K <= 0) {
        /* FUSE is a modifier of EVPWIDE, so on its own it does nothing — and would do nothing
         * SILENTLY, which is the one failure mode this project keeps paying for (L80). */
        static bool warned = false;
        if (!warned && fesom_evpwide_env_fuse() && p->mype == 0) {
            warned = true;
            fprintf(stderr, "[fesom_speed] !! FESOM_SPEED_EVPWIDE_FUSE was requested but "
                            "FESOM_SPEED_EVPWIDE is off — FUSE only changes how the wide-halo "
                            "window ship is transported, so it is NOT running.\n");
            fflush(stderr);
        }
        return 0;
    }

    /* M9: whichEVP==1 (mEVP) is now SUPPORTED — cells ②/④, the exact wide halo. The refusal
     * that used to sit here was correct only while the ghost kernel bodies existed for std EVP
     * alone; fesom_ice_maevp.cpp now carries its own. Anything else still refuses. */
    if (ice->whichEVP != 0 && ice->whichEVP != 1) {
        if (!g_w.announced && p->mype == 0) {
            fprintf(stderr, "[fesom_speed] !! FESOM_SPEED_EVPWIDE requested but whichEVP=%d "
                            "(no ghost kernel bodies exist for that rheology) — the lever is "
                            "NOT running.\n", ice->whichEVP);
            fflush(stderr);
        }
        g_w.announced = true;
        return 0;
    }
    if (!g_w.hook_done) {
        if (!g_w.announced && p->mype == 0) {
            fprintf(stderr, "[fesom_speed] !! FESOM_SPEED_EVPWIDE requested but no extended zone "
                            "was built (npes==1?) — the lever is NOT running.\n");
            fflush(stderr);
        }
        g_w.announced = true;
        return 0;
    }
    FESOM_CHECK(ice->evp_rheol_steps % K == 0,
                "evpwide: K=%d must divide evp_rheol_steps=%d", K, ice->evp_rheol_steps);
    {
        const char *v = getenv("FESOM_KK_VERIFY");
        FESOM_CHECK(!(v && strcmp(v, "evp") == 0),
                    "evpwide: FESOM_KK_VERIFY=evp is incompatible with FESOM_SPEED_EVPWIDE "
                    "(the host twin is not wide-aware)");
    }
    if (!g_w.built) evpw_build(ice, p, m);
    if (!g_w.announced) {
        if (p->mype == 0) {
            const int ns = ice->evp_rheol_steps;
            fprintf(stderr, "[fesom_speed] FESOM_SPEED_EVPWIDE = %d (%s, R=K=%d rings, exact "
                            "replay w/ per-window refresh, subcycle exchanges/step %d -> %d+1)\n",
                    K, ice->whichEVP == 1 ? "mEVP: cell ② classic-form / ④ divergence-form"
                                          : "std EVP",
                    g_w.R, ns, ns / K);
            /* D1 / P0b — say WHICH transport ran. Fused and unfused ship identical bytes, so
             * nothing downstream can tell them apart; this line and the message counter are the
             * only evidence, and a silent default would make the A/B unauditable (L80). */
            fprintf(stderr, "[fesom_speed] FESOM_SPEED_EVPWIDE_FUSE = %d (two-segment window ship "
                            "= %s; no-op for any ONE-segment ship, i.e. the prestep 11-field "
                            "ship and cell ④)\n",
                    g_w.fuse ? 1 : 0,
                    g_w.fuse ? "FUSED, 1 msg/partner"
                             : "unfused, 2 msgs/partner (tags 2200 + 2205)");
            fflush(stderr);
        }
        g_w.announced = true;
    }
    return K;
}

const FesomEvpwideDev &fesom_evpwide_dev(void) { return g_w.dev; }

/* ---- the nf-field extended exchange ----------------------------------------- */
namespace {

struct PackViews { DevV v[11]; };

/*
 * M9 — what ONE extended exchange ships and what it diagnoses.
 *
 * Before M9 all of this was implied by the single integer `nf`, and EVERY diagnostic was keyed
 * on its VALUE: `sig = (nf<=6)`, the selfcheck on `nf<=6`, the debug split on `nf==6`, the
 * window print on `nf==2`, the prestep echo on `nf==11`. Handing the field list to the caller
 * (which is what lets mEVP reuse this machinery) silently re-keys every one of them — and the
 * FORCE_SERIAL byte proof passes just as happily with all the diagnostics DEAD, because a dead
 * diagnostic changes no bytes. That is the L80 class, and it is why intent is now stated
 * explicitly instead of being inferred from a count.
 *
 * The std-EVP callers below construct exactly the three plans that used to be nf = 11 / 2 / 6,
 * so their behaviour is unchanged by construction.
 */
struct EvpwPlan {
    PackViews F;
    int  nf  = 0;              /* node fields shipped */
    int  nun = 0;              /* how many are UNPACKED; the rest are compare-only */
    bool sig = false;          /* also ship the element sigma segment (tag 2205) */
    enum Diag { NONE, WINDOW, WINDOW_DBG, PRESTEP_ECHO } diag = NONE;
};

/*
 * Buffer index of one exchange ENTRY, in either transport form. This is the whole of the fused
 * variant on the kernel side: same entries, same order, same values — a different address.
 *
 *   unfused: the node segment is its own buffer (stride nf) and the element segment is its own
 *            (stride 3), so the index is just entry*stride.
 *   fused:   one buffer holds, per partner q, [that partner's node entries][its elem entries],
 *            so an entry needs its partner's base offset. `part` gives the partner INDEX of the
 *            entry; `noff`/`eoff` are the per-partner entry offsets already used to post MPI.
 *
 * Making BOTH forms go through this one object is deliberate: every diagnostic in the exchange
 * (selfcheck drift, prestep echo, the level-3 dump) reads the recv buffer by hand, and a fused
 * form that quietly left them indexing the unfused buffer would read stale/garbage bytes while
 * a byte gate — which only compares model output — passed. That is the L80 shape.
 */
struct SegIdx {
    Kokkos::View<const int*> part, noff, eoff;
    int nf = 0;                /* node fields of the ACTIVE plan (element stride is always 3) */
    int fused = 0;
    int elem = 0;              /* 0 = node segment, 1 = element (sigma) segment */
    KOKKOS_INLINE_FUNCTION int operator()(const int i) const
    {
        if (!fused) return elem ? i * 3 : i * nf;
        const int q = part(i);
        const int base = noff(q) * nf + eoff(q) * 3;
        if (!elem) return base + (i - noff(q)) * nf;
        return base + (noff(q + 1) - noff(q)) * nf + (i - eoff(q)) * 3;
    }
};

void evpw_exchange(struct fesom_ice *ice, struct fesom_partit *p, const EvpwPlan &P)
{
    EvpwState &S = g_w;
    FESOM_CHECK(S.built, "evpwide: exchange before build");
    /* the send/recv buffers are sized nsend*11 / nrecv*11 in evpw_build — a plan wider than
     * that would overrun them silently. Widen the sizing first if a caller ever needs it. */
    FESOM_CHECK(P.nf >= 1 && P.nf <= 11,
                "evpwide: plan ships %d node fields; buffers are sized for 11", P.nf);
    FESOM_CHECK(P.nun >= 0 && P.nun <= P.nf, "evpwide: plan unpacks %d of %d", P.nun, P.nf);

    const PackViews F  = P.F;
    const int       nf = P.nf;

    fesom_halo_prof_barrier(p);                 /* M5.17 split-instrumentation parity */

    const bool sig = P.sig;       /* the wide refresh re-baselines ghost SIGMA every window
                                   * (design §2 corrected: velocity-only refresh is inexact
                                   * for any finite R); the per-step 11-field ship does not
                                   * (sigma is untouched on both sides between EVP calls). */
    /* D1 / P0b. Fusing only means anything when there ARE two segments, so a one-segment plan
     * (the 11-field prestep ship, cell ④'s window) is untouched by the knob BY CONSTRUCTION —
     * which is also why cell ④ is the right control for this measurement: it already ships one
     * message and the knob cannot change it. */
    const bool fused = S.fuse && sig;
    SegIdx NI, EI;
    NI.part = S.spart_d; NI.noff = S.soff_d; NI.eoff = S.esoff_d;
    NI.nf = nf; NI.fused = fused ? 1 : 0; NI.elem = 0;
    EI = NI; EI.part = S.espart_d; EI.elem = 1;
    SegIdx RI = NI, ERI = EI;
    RI.part  = S.rpart_d;  RI.noff  = S.roff_d; RI.eoff  = S.eroff_d;
    ERI.part = S.erpart_d; ERI.noff = S.roff_d; ERI.eoff = S.eroff_d;

    {   /* pack owned values */
        auto sidx = S.sidx_d; const PackViews FF = F; const int nfl = nf;
        auto sbuf = fused ? S.fsbuf_d : S.sbuf_d;
        const SegIdx IX = NI;
        if (S.nsend > 0)
            Kokkos::parallel_for("fesom_evpwide_pack", Kokkos::RangePolicy<>(0, S.nsend),
                KOKKOS_LAMBDA(const int i) {
                    const int n = sidx(i);
                    const int b = IX(i);
                    for (int f = 0; f < nfl; ++f) sbuf(b + f) = FF.v[f](n);
                });
    }
    if (sig && S.nesend > 0) {   /* pack my (owner-clean) sigma for the requesters */
        auto esidx = S.esidx_d;
        auto esbuf = fused ? S.fsbuf_d : S.esbuf_d;
        const SegIdx IX = EI;
        auto s11 = ice->work.sigma11_fld.d();
        auto s12 = ice->work.sigma12_fld.d();
        auto s22 = ice->work.sigma22_fld.d();
        Kokkos::parallel_for("fesom_evpwide_packsig", Kokkos::RangePolicy<>(0, S.nesend),
            KOKKOS_LAMBDA(const int i) {
                const int e = esidx(i);
                const int b = IX(i);
                esbuf(b + 0) = s11(e);
                esbuf(b + 1) = s12(e);
                esbuf(b + 2) = s22(e);
            });
    }
    Kokkos::fence();   /* MANDATORY pre-MPI: MPI reads sbuf + re-posts rbuf (drains prev unpack) */

    int nreq = 0;
    real_t *sp = fused ? S.fsbuf_d.data() : S.sbuf_d.data();
    real_t *rp = fused ? S.frbuf_d.data() : S.rbuf_d.data();
    double bytes = 0.0;
    if (fused) {
        /* ONE message per partner: [node entries][elem entries], contiguous. Same bytes as the
         * two-segment form, half the messages — which is the whole content of the variant. */
        for (size_t q = 0; q < S.partner.size(); ++q) {
            const int rc = (S.roff[q + 1] - S.roff[q]) * nf + (S.eroff[q + 1] - S.eroff[q]) * 3;
            if (rc > 0) {
                MPI_Irecv(rp + ((size_t)S.roff[q] * nf + (size_t)S.eroff[q] * 3), rc, MPI_DOUBLE,
                          S.partner[q], 2200, p->MPI_COMM_FESOM, &S.reqs[(size_t)nreq++]);
                bytes += (double)rc * sizeof(real_t);
            }
        }
        for (size_t q = 0; q < S.partner.size(); ++q) {
            const int sc = (S.soff[q + 1] - S.soff[q]) * nf + (S.esoff[q + 1] - S.esoff[q]) * 3;
            if (sc > 0)
                MPI_Isend(sp + ((size_t)S.soff[q] * nf + (size_t)S.esoff[q] * 3), sc, MPI_DOUBLE,
                          S.partner[q], 2200, p->MPI_COMM_FESOM, &S.reqs[(size_t)nreq++]);
        }
    } else {
        for (size_t q = 0; q < S.partner.size(); ++q) {
            const int rc = (S.roff[q + 1] - S.roff[q]) * nf;
            if (rc > 0) {
                MPI_Irecv(rp + (size_t)S.roff[q] * nf, rc, MPI_DOUBLE, S.partner[q], 2200,
                          p->MPI_COMM_FESOM, &S.reqs[(size_t)nreq++]);
                bytes += (double)rc * sizeof(real_t);
            }
        }
        for (size_t q = 0; q < S.partner.size(); ++q) {
            const int sc = (S.soff[q + 1] - S.soff[q]) * nf;
            if (sc > 0)
                MPI_Isend(sp + (size_t)S.soff[q] * nf, sc, MPI_DOUBLE, S.partner[q], 2200,
                          p->MPI_COMM_FESOM, &S.reqs[(size_t)nreq++]);
        }
        if (sig) {   /* the sigma segment: second message per partner, tag 2205 */
            real_t *esp = S.esbuf_d.data();
            real_t *erp = S.erbuf_d.data();
            for (size_t q = 0; q < S.partner.size(); ++q) {
                const int rc = (S.eroff[q + 1] - S.eroff[q]) * 3;
                if (rc > 0) {
                    MPI_Irecv(erp + (size_t)S.eroff[q] * 3, rc, MPI_DOUBLE, S.partner[q], 2205,
                              p->MPI_COMM_FESOM, &S.reqs[(size_t)nreq++]);
                    bytes += (double)rc * sizeof(real_t);
                }
            }
            for (size_t q = 0; q < S.partner.size(); ++q) {
                const int sc = (S.esoff[q + 1] - S.esoff[q]) * 3;
                if (sc > 0)
                    MPI_Isend(esp + (size_t)S.esoff[q] * 3, sc, MPI_DOUBLE, S.partner[q], 2205,
                              p->MPI_COMM_FESOM, &S.reqs[(size_t)nreq++]);
            }
        }
    }
    S.msg_count += nreq;                  /* the ONLY observable that separates the two forms */
    S.dbl_count += bytes / (double)sizeof(real_t);
    if (P.diag == EvpwPlan::PRESTEP_ECHO) { S.msg_pre += nreq; S.dbl_pre += bytes / (double)sizeof(real_t); }
    else                                  { S.msg_win += nreq; S.dbl_win += bytes / (double)sizeof(real_t); }
    fesom_halo_prof_waitall(nreq, S.reqs.data());
    fesom_halo_prof_bytes(bytes);

    /* selfcheck, wide-refresh path (nf<=4): refresh-vs-local BEFORE unpack. With R=K +
     * per-window sigma refresh, ghost state at window END is legitimately dirty (the dirt
     * never reaches owned reads — design §2 corrected) => this is a DRIFT DIAGNOSTIC,
     * expected small-but-NONZERO. The exact-zero certification is the FORCE_SERIAL
     * diff_snap gate on owned bytes, plus the pre-step echo check below. */
    if ((P.diag == EvpwPlan::WINDOW || P.diag == EvpwPlan::WINDOW_DBG) && S.selfcheck) {
        auto ridx = S.ridx_d; auto flag = S.selff_d;
        auto rbuf = fused ? S.frbuf_d : S.rbuf_d;      /* both forms, one index object */
        const SegIdx IX = RI;
        const PackViews FF = F; const int Nlim = S.N;
        double mx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int f = 0; f < nf; ++f) {
            double m1 = 0.0; const int fl = f;
            Kokkos::parallel_reduce("fesom_evpwide_self", Kokkos::RangePolicy<>(0, S.nrecv),
                KOKKOS_LAMBDA(const int i, double &acc) {
                    if (!flag(i)) return;
                    const int s = ridx(i);
                    if (fl >= 2 && s >= Nlim) return;      /* u_rhs is not extended */
                    const double d = Kokkos::fabs((double)rbuf(IX(i) + fl) - (double)FF.v[fl](s));
                    if (d > acc) acc = d;
                }, Kokkos::Max<double>(m1));
            mx[f] = m1;
        }
        if (P.diag == EvpwPlan::WINDOW_DBG) {   /* debug split of the urhs mismatch by gather composition */
            for (int cls = 1; cls <= 2; ++cls) {
                double m1 = 0.0; const char want = (char)cls;
                Kokkos::parallel_reduce("fesom_evpwide_self_cls", Kokkos::RangePolicy<>(0, S.nrecv),
                    KOKKOS_LAMBDA(const int i, double &acc) {
                        if (flag(i) != want) return;
                        const int s = ridx(i);
                        if (s >= Nlim) return;
                        const double d = Kokkos::fabs((double)rbuf(IX(i) + 2) - (double)FF.v[2](s));
                        if (d > acc) acc = d;
                    }, Kokkos::Max<double>(m1));
                mx[5 + cls] = m1;
            }
        }
        double gmx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        MPI_Reduce(mx, gmx, 8, MPI_DOUBLE, MPI_MAX, 0, p->MPI_COMM_FESOM);
        if (p->mype == 0) {
            if (P.diag == EvpwPlan::WINDOW)
                printf("[evpwide-self] exch %ld  ghost drift (end-clean set, expected 0 only if R>K) = %.3e\n",
                       g_w.exch_count, gmx[0] > gmx[1] ? gmx[0] : gmx[1]);
            else
                printf("[evpwide-self] exch %ld  drift u=%.3e v=%.3e urhs=%.3e vrhs=%.3e "
                       "invam=%.3e invm=%.3e | urhs[own-only]=%.3e urhs[with-ghost]=%.3e\n",
                       g_w.exch_count, gmx[0], gmx[1], gmx[2], gmx[3], gmx[4], gmx[5], gmx[6], gmx[7]);
            fflush(stdout);
        }
    }
#ifndef KOKKOS_ENABLE_CUDA
    /* selfcheck level 3 (Serial-only): dump the WORST urhs slot's full gather composition.
     * All Views are host-resident on Serial — direct indexing. */
    static bool s_dumped = false;
    if (P.diag == EvpwPlan::WINDOW_DBG && S.selfcheck >= 3 && !s_dumped) {
        s_dumped = true;
        auto rbuf  = fused ? S.frbuf_d : S.rbuf_d;
        auto erbuf = fused ? S.frbuf_d : S.erbuf_d;   /* fused: same buffer, elem offset */
        auto ridx = S.ridx_d; auto flag = S.selff_d;
        const SegIdx IX = RI, EIX = ERI;
        int ndust = 0;
        for (int i2 = 0; i2 < S.nrecv && ndust < 4; ++i2) {
            if (!flag(i2)) continue;
            const int s2 = ridx(i2);
            if (s2 >= S.N) continue;
            double dd = 0.0;
            for (int f2 = 0; f2 < 4; ++f2) {
                const double d2 = fabs(rbuf(IX(i2) + f2) - (double)F.v[f2](s2));
                if (d2 > dd) dd = d2;
            }
            if (dd <= 0.0) continue;
            ++ndust;
            const int wi = i2; const double wd = dd;
        {
            const int s = ridx(wi);
            const FesomEvpwideDev &D = S.dev;
            fprintf(stderr, "[evpw-dump] rank %d slot %d dist %d owner-urhs %.17g "
                            "diff %.3e  invam %.17g rhs_a %.17g row [%d..%d):\n",
                    p->mype, s, (int)S.dist[(size_t)s],
                    rbuf(IX(wi) + 2), wd,
                    (double)ice->work.inv_areamass_fld.d()(s),
                    (double)ice->data[FESOM_ICE_AICE].values_rhs_fld.d()(s),
                    (int)D.gath_ptr(s), (int)D.gath_ptr(s + 1));
            auto s11o = ice->work.sigma11_fld.d(); auto s12o = ice->work.sigma12_fld.d();
            auto s22o = ice->work.sigma22_fld.d();
            auto istro = ice->work.ice_strength_fld.d();
            double sum = 0.0;
            const real_t val3d = 1.0/3.0;
            for (int q = D.gath_ptr(s); q < D.gath_ptr(s + 1); ++q) {
                const int ue = D.gath_elem(q); const int kk = D.gath_k(q);
                double istr_v, s11v, s12v, s22v, gsa, gsb, mfv, eav, cu = 0.0;
                if (ue < S.E) {
                    istr_v = istro(ue); s11v = s11o(ue); s12v = s12o(ue); s22v = s22o(ue);
                    gsa = S.gs_h[6*ue + kk]; gsb = S.gs_h[6*ue + kk + 3];
                    mfv = S.mf_h[ue]; eav = S.ea_h[ue];
                } else {
                    const int eg = ue - S.E;
                    istr_v = D.istr_g(eg); s11v = D.s11_g(eg); s12v = D.s12_g(eg); s22v = D.s22_g(eg);
                    gsa = D.gs_g(6*eg + kk); gsb = D.gs_g(6*eg + kk + 3);
                    mfv = D.mf_g(eg); eav = D.ea_g(eg);
                }
                if (istr_v > 0.0)
                    cu = -(eav*(s11v*gsa + s12v*gsb + s12v*val3d*mfv));
                sum += cu;
                double o11 = 0, o12 = 0, o22 = 0;   /* the OWNER's sigma (incoming refresh) */
                if (ue >= S.E) {
                    const int eg = ue - S.E;
                    for (int z = 0; z < S.nerecv; ++z)
                        if (S.eridx_d(z) == eg) {
                            const int eb = EIX(z);
                            o11 = erbuf(eb + 0); o12 = erbuf(eb + 1); o22 = erbuf(eb + 2);
                            break;
                        }
                }
                fprintf(stderr, "  [%c] ue %d kk %d istr %.6g s11 %.10g s12 %.10g s22 %.10g "
                                "gs %.10g/%.10g mf %.6g ea %.6g cu %.17g | own-sig %.10g %.10g %.10g\n",
                        ue < S.E ? 'S' : 'G', ue, kk, istr_v, s11v, s12v, s22v, gsa, gsb, mfv, eav, cu,
                        o11, o12, o22);
                for (int z = 0; z < 3; ++z) {
                    const int vs = (ue < S.E) ? -1 : (int)S.dev.en_g(3*(ue - S.E) + z);
                    if (vs < 0) continue;
                    const int vgid = (vs < S.N) ? p->myList_nod2D[vs] : S.ext_gid1[(size_t)(vs - S.N)];
                    fprintf(stderr, "      v%d slot %d gid %d dist %d u %.10g v %.10g\n",
                            z, vs, vgid, (int)S.dist[(size_t)vs],
                            (double)F.v[0](vs), (double)F.v[1](vs));
                }
            }
            const double invam = ice->work.inv_areamass_fld.d()(s);
            const double rhsa  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d()(s);
            fprintf(stderr, "[evpw-dump] sum %.17g -> final %.17g vs owner %.17g | my u/v %.17g %.17g "
                            "own u/v %.17g %.17g coastx %d aice %.17g\n",
                    sum, invam > 0.0 ? sum*invam + rhsa : 0.0, rbuf(IX(wi) + 2),
                    (double)F.v[0](s), (double)F.v[1](s), rbuf(IX(wi) + 0), rbuf(IX(wi) + 1),
                    (int)S.dev.coastx(s), (double)ice->data[FESOM_ICE_AICE].values_fld.d()(s));
            {   /* permutation probe: does ANY summation order of my contributions reproduce
                 * the owner's finalized u_rhs? (order bug vs value bug discriminator) */
                std::vector<double> cus;
                for (int q = D.gath_ptr(s); q < D.gath_ptr(s + 1); ++q) {
                    const int ue = D.gath_elem(q); const int kk = D.gath_k(q);
                    double istr_v, s11v, s12v, gsa, gsb, mfv, eav;
                    if (ue < S.E) {
                        istr_v = ice->work.ice_strength_fld.d()(ue);
                        s11v = ice->work.sigma11_fld.d()(ue); s12v = ice->work.sigma12_fld.d()(ue);
                        gsa = S.gs_h[6*ue + kk]; gsb = S.gs_h[6*ue + kk + 3];
                        mfv = S.mf_h[ue]; eav = S.ea_h[ue];
                    } else {
                        const int eg = ue - S.E;
                        istr_v = D.istr_g(eg); s11v = D.s11_g(eg); s12v = D.s12_g(eg);
                        gsa = D.gs_g(6*eg + kk); gsb = D.gs_g(6*eg + kk + 3);
                        mfv = D.mf_g(eg); eav = D.ea_g(eg);
                    }
                    if (istr_v > 0.0)
                        cus.push_back(-(eav*(s11v*gsa + s12v*gsb + s12v*val3d*mfv)));
                }
                const double target = rbuf(IX(wi) + 2);
                int hit = -1;
                if (cus.size() <= 8) {
                    std::vector<int> pi(cus.size());
                    for (size_t z = 0; z < pi.size(); ++z) pi[z] = (int)z;
                    int pidx = 0;
                    do {
                        double sp2 = 0.0;
                        for (size_t z = 0; z < pi.size(); ++z) sp2 += cus[(size_t)pi[(size_t)z]];
                        const double fin = invam > 0.0 ? sp2*invam + rhsa : 0.0;
                        if (fin == target) { hit = pidx; break; }
                        ++pidx;
                    } while (std::next_permutation(pi.begin(), pi.end()));
                }
                fprintf(stderr, "[evpw-dump] permutation probe: n=%zu hit=%d (0=my order; -1=NO order matches)\n",
                        cus.size(), hit);
            }
        }
        }
    }
#endif
    /* pre-step echo check (nf==11): uice/vice are modified ONLY by the EVP window loop, and
     * the last window ends with a refresh => the incoming ghost bytes MUST equal what we
     * already hold, EXACTLY, on every backend. A nonzero here = plumbing or an unexpected
     * outside writer of uice/vice ghosts. */
    if (P.diag == EvpwPlan::PRESTEP_ECHO && S.selfcheck) {
        auto ridx = S.ridx_d;
        auto rbuf = fused ? S.frbuf_d : S.rbuf_d;
        const SegIdx IX = RI;
        auto u = F.v[0]; auto v = F.v[1];
        double mx0 = 0.0;
        Kokkos::parallel_reduce("fesom_evpwide_echo", Kokkos::RangePolicy<>(0, S.nrecv),
            KOKKOS_LAMBDA(const int i, double &acc) {
                const int s = ridx(i);
                double d = Kokkos::fabs((double)rbuf(IX(i) + 0) - (double)u(s));
                const double d2 = Kokkos::fabs((double)rbuf(IX(i) + 1) - (double)v(s));
                if (d2 > d) d = d2;
                if (d > acc) acc = d;
            }, Kokkos::Max<double>(mx0));
        double gmx0 = 0.0;
        MPI_Reduce(&mx0, &gmx0, 1, MPI_DOUBLE, MPI_MAX, 0, p->MPI_COMM_FESOM);
        if (p->mype == 0) {
            printf("[evpwide-self] step-ship uv echo = %.3e (MUST be 0.000e+00)\n", gmx0);
            fflush(stdout);
        }
    }
    ++g_w.exch_count;

    {   /* unpack into ghost slots (fields >= 2 of the dbg nf==4 mode are compare-only) */
        auto ridx = S.ridx_d; const PackViews FF = F;
        auto rbuf = fused ? S.frbuf_d : S.rbuf_d;
        const SegIdx IX = RI;
        const int nun = P.nun;                   /* compare-only fields are not unpacked */
        if (S.nrecv > 0)
            Kokkos::parallel_for("fesom_evpwide_unpack", Kokkos::RangePolicy<>(0, S.nrecv),
                KOKKOS_LAMBDA(const int i) {
                    const int s = ridx(i);
                    const int b = IX(i);
                    for (int f = 0; f < nun; ++f) FF.v[f](s) = rbuf(b + f);
                });
    }
    if (sig && S.nerecv > 0) {   /* re-baseline ghost sigma to owner bytes */
        auto eridx = S.eridx_d;
        auto erbuf = fused ? S.frbuf_d : S.erbuf_d;
        const SegIdx IX = ERI;
        auto s11g = S.dev.s11_g; auto s12g = S.dev.s12_g; auto s22g = S.dev.s22_g;
        Kokkos::parallel_for("fesom_evpwide_unpacksig", Kokkos::RangePolicy<>(0, S.nerecv),
            KOKKOS_LAMBDA(const int i) {
                const int eg = eridx(i);
                const int b = IX(i);
                s11g(eg) = erbuf(b + 0);
                s12g(eg) = erbuf(b + 1);
                s22g(eg) = erbuf(b + 2);
            });
    }
    /* no post-unpack fence: consumers are same-stream kernels; the pre-MPI fence of the NEXT
     * exchange drains this unpack before rbuf is re-posted (the cgpipe/NOFENCE2 audit). */
}

} /* anonymous namespace */

/* ── std-EVP plans. These reproduce the pre-M9 nf = 11 / 2 / 6 cases EXACTLY, which is what
 *    keeps the certified std-EVP behaviour unchanged by construction. ────────────────────── */

void fesom_evpwide_prestep_exchange(struct fesom_ice *ice, struct fesom_partit *p)
{
    EvpwPlan P;                       /* was nf = 11 */
    P.F.v[0]  = ice->uice_fld.d();
    P.F.v[1]  = ice->vice_fld.d();
    P.F.v[2]  = ice->data[FESOM_ICE_AICE].values_fld.d();
    P.F.v[3]  = ice->data[FESOM_ICE_MICE].values_fld.d();
    P.F.v[4]  = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    P.F.v[5]  = ice->srfoce_u_fld.d();
    P.F.v[6]  = ice->srfoce_v_fld.d();
    P.F.v[7]  = ice->stress_atmice_x_fld.d();
    P.F.v[8]  = ice->stress_atmice_y_fld.d();
    P.F.v[9]  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    P.F.v[10] = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    P.nf = 11; P.nun = 11;
    P.sig = false;                    /* sigma is untouched on both sides between EVP calls */
    P.diag = EvpwPlan::PRESTEP_ECHO;  /* uice/vice ghosts MUST already equal the incoming bytes */
    evpw_exchange(ice, p, P);
}

void fesom_evpwide_subcycle_exchange(struct fesom_ice *ice, struct fesom_partit *p)
{
    EvpwPlan P;
    P.F.v[0] = ice->uice_fld.d();
    P.F.v[1] = ice->vice_fld.d();
    P.nf = 2; P.nun = 2;
    P.sig = true;                     /* re-baseline ghost sigma every window (design §2 corr.) */
    P.diag = EvpwPlan::WINDOW;
    if (g_w.selfcheck >= 2) {         /* was nf = 6: ship the owner's finalized u_rhs/v_rhs +
                                       * inv_am/inv_m for COMPARISON only, never unpacked —
                                       * splits gather bugs from finalize-input bugs. */
        P.F.v[2] = ice->uice_rhs_fld.d();
        P.F.v[3] = ice->vice_rhs_fld.d();
        P.F.v[4] = ice->work.inv_areamass_fld.d();
        P.F.v[5] = ice->work.inv_mass_fld.d();
        P.nf = 6; P.nun = 2;
        P.diag = EvpwPlan::WINDOW_DBG;
    }
    evpw_exchange(ice, p, P);
}

/* ── M9 mEVP plans (cells ②/④) ─────────────────────────────────────────────────────────────
 * These are the reason evpw_exchange takes a caller-supplied plan at all (Task 11). Note what
 * is NOT here: no new transport, no new handshake, no new buffers. The ring zone, the owner
 * vector, the owner-order gather CSR and the message machinery are all shared with std EVP —
 * mEVP differs only in WHICH fields ride and WHICH ghost kernels consume them. */

void fesom_evpwide_mevp_prestep_exchange(struct fesom_ice *ice, struct fesom_partit *p)
{
    EvpwPlan P;
    P.F.v[0]  = ice->uice_fld.d();
    P.F.v[1]  = ice->vice_fld.d();
    P.F.v[2]  = ice->data[FESOM_ICE_AICE].values_fld.d();
    P.F.v[3]  = ice->data[FESOM_ICE_MICE].values_fld.d();
    P.F.v[4]  = ice->data[FESOM_ICE_MSNOW].values_fld.d();
    P.F.v[5]  = ice->srfoce_u_fld.d();
    P.F.v[6]  = ice->srfoce_v_fld.d();
    P.F.v[7]  = ice->stress_atmice_x_fld.d();
    P.F.v[8]  = ice->stress_atmice_y_fld.d();
    /* ⚠️ TRAP 7 — these two are shipped AFTER maevp_node_pre has divided them by area, so the
     * ghost node_pre must NOT scale them again. mEVP scales rhs INSIDE its ice branch, i.e.
     * only at nodes with a_ice >= 0.01; shipping the owner's post-scaling value is right in
     * both cases because the ghost solve reads rhs only where it is scaled. */
    P.F.v[9]  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
    P.F.v[10] = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    P.nf = 11; P.nun = 11;
    P.sig = false;                    /* sigma travels on the WINDOW ship (cell ②) or not at all */
    /* The echo is exact here for the same reason it is under std EVP: the last subcycle of the
     * previous step always refreshes (K divides evp_rheol_steps), and maevp_final_copy then
     * publishes u_aux into u_ice over the FULL extended range — so the incoming ghost uice/vice
     * bytes must equal what we already hold, on every backend. A nonzero is plumbing. */
    P.diag = EvpwPlan::PRESTEP_ECHO;
    evpw_exchange(ice, p, P);
}

void fesom_evpwide_mevp_window_exchange(struct fesom_ice *ice, struct fesom_partit *p,
                                        int div_on)
{
    EvpwPlan P;
    P.F.v[0] = ice->uice_aux_fld.d();   /* mEVP subcycles u_aux/v_aux, not u_ice/v_ice */
    P.F.v[1] = ice->vice_aux_fld.d();
    if (div_on) {
        /* Cell ④ — everything carried is a NODE field, so one segment carries the whole state:
         * 4*Ng doubles and no element message. This is what the reformulation buys; it is not
         * faster per element (H1 falsified) but it is half the wide-halo traffic of ②. */
        P.F.v[2] = ice->work.mevp_Ru_fld.d();
        P.F.v[3] = ice->work.mevp_Rv_fld.d();
        P.nf = 4; P.nun = 4;
        P.sig = false;
    } else {
        /* Cell ② — the carried state is element sigma, so it needs the second (element) segment:
         * 2*Ng node doubles + 3 per ghost element (~2*Ng elements) = 8*Ng doubles. */
        P.nf = 2; P.nun = 2;
        P.sig = true;
    }
    P.diag = EvpwPlan::WINDOW;
    evpw_exchange(ice, p, P);
}

void fesom_evpwide_msg_report(int timed_steps, struct fesom_partit *p)
{
    if (!p || timed_steps <= 0) return;
    int on = g_w.built ? 1 : 0, any = 0;
    MPI_Allreduce(&on, &any, 1, MPI_INT, MPI_MAX, p->MPI_COMM_FESOM);
    if (!any) return;                     /* lever off everywhere: say nothing */
    /* msg_count spans the WHOLE run, timed_steps only the timed window; the ratio is what is
     * comparable across legs, and it is exact because the exchange cadence is per-step. */
    double loc[6] = { (double)g_w.msg_count, g_w.dbl_count,
                      (double)g_w.msg_pre, g_w.dbl_pre,
                      (double)g_w.msg_win, g_w.dbl_win }, mx[6], sum[6];
    MPI_Reduce(loc, mx,  6, MPI_DOUBLE, MPI_MAX, 0, p->MPI_COMM_FESOM);
    MPI_Reduce(loc, sum, 6, MPI_DOUBLE, MPI_SUM, 0, p->MPI_COMM_FESOM);
    if (p->mype != 0) return;
    const double st = (double)timed_steps;
    printf("[evpwide-wire] transport=%s  msgs/step: max-rank %.1f, all-ranks %.1f | "
           "doubles recv/step: max-rank %.0f, all-ranks %.0f  (bytes MUST match between the "
           "fused and unfused forms; only the message count may differ)\n",
           g_w.fuse ? "FUSED" : "unfused", mx[0]/st, sum[0]/st, mx[1]/st, sum[1]/st);
    printf("[evpwide-wire] split per step (all-ranks): PRESTEP 11-field ship %.1f msgs / %.0f "
           "doubles (%.1f%% of msgs, %.1f%% of doubles) | WINDOW ships %.1f msgs / %.0f doubles\n",
           sum[2]/st, sum[3]/st,
           sum[0] > 0 ? 100.0*sum[2]/sum[0] : 0.0, sum[1] > 0 ? 100.0*sum[3]/sum[1] : 0.0,
           sum[4]/st, sum[5]/st);
    fflush(stdout);
}

void fesom_evpwide_free(void)
{
    EvpwState &S = g_w;
    S.dev = FesomEvpwideDev();
    S.sidx_d = Kokkos::View<int*>();  S.ridx_d = Kokkos::View<int*>();
    S.esidx_d = Kokkos::View<int*>(); S.eridx_d = Kokkos::View<int*>();
    S.selff_d = Kokkos::View<char*>();
    S.sbuf_d = Kokkos::View<real_t*>(); S.rbuf_d = Kokkos::View<real_t*>();
    S.esbuf_d = Kokkos::View<real_t*>(); S.erbuf_d = Kokkos::View<real_t*>();
    S.fsbuf_d = Kokkos::View<real_t*>(); S.frbuf_d = Kokkos::View<real_t*>();
    S.spart_d = Kokkos::View<int*>();  S.rpart_d = Kokkos::View<int*>();
    S.espart_d = Kokkos::View<int*>(); S.erpart_d = Kokkos::View<int*>();
    S.soff_d = Kokkos::View<int*>();   S.roff_d = Kokkos::View<int*>();
    S.esoff_d = Kokkos::View<int*>();  S.eroff_d = Kokkos::View<int*>();
    S.built = false;
}
