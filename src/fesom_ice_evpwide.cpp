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
    long ring1_bfs = 0, ring1_extra = 0;        /* diagnostics: BFS ring-1 vs eDim */

    /* stage B: lazy build */
    bool built = false;
    FesomEvpwideDev dev;
    std::vector<int> partner;                   /* ascending ranks */
    std::vector<int> soff, roff;                /* [P+1] slot offsets */
    int nsend = 0, nrecv = 0;
    Kokkos::View<int*>    sidx_d, ridx_d;       /* owned idx / slot idx */
    Kokkos::View<char*>   selff_d;              /* [nrecv] selfcheck flag: dist(slot) <= K-1 */
    Kokkos::View<real_t*> sbuf_d, rbuf_d;       /* [nsend*11] / [nrecv*11] */
    std::vector<MPI_Request> reqs;
    int selfcheck = -1;                         /* env FESOM_EVPWIDE_SELFCHECK */
    long exch_count = 0;
};
EvpwState g_w;

/* ---- verbatim geometry formulas (fesom_mesh.cpp mirrors; see :763 elem_area,
 * :843 elem_center_xy, :909 elem_cos/metric_factor, :1019 gradient_sca).
 * Pure per-element functions of the 3 vertex coords => byte-equal to the mesh build for
 * identical input doubles. Cross-checked at build against mesh values on owned elements. */
struct XY { const real_t *c; };   /* unified coord accessor built at build time */

static real_t f_elem_area(const real_t xy[6])
{
    const real_t cyc      = FESOM_CYCLIC_LENGTH_RAD;
    const real_t half_cyc = 0.5 * cyc;
    const real_t r2       = (real_t)FESOM_R_EARTH * (real_t)FESOM_R_EARTH;
    real_t ay = (xy[1] + xy[3] + xy[5]) / 3.0;
    ay = cos(ay);
    real_t ax  = xy[2] - xy[0];
    real_t aly = xy[3] - xy[1];
    real_t bx  = xy[4] - xy[0];
    real_t bly = xy[5] - xy[1];
    if (ax >  half_cyc) ax -= cyc;
    if (ax < -half_cyc) ax += cyc;
    if (bx >  half_cyc) bx -= cyc;
    if (bx < -half_cyc) bx += cyc;
    ax *= ay;
    bx *= ay;
    real_t cross = ax * bly - bx * aly;
    if (cross < 0) cross = -cross;
    return (0.5 * cross) * r2;                 /* area computed in rad², then *r2 — matches
                                                  the two-loop sequence at :798/:801 */
}

static void f_elem_center(const real_t xy[6], real_t *cx, real_t *cy)
{
    const real_t cyc      = FESOM_CYCLIC_LENGTH_RAD;
    const real_t half_cyc = 0.5 * cyc;
    real_t ax[3] = { xy[0], xy[2], xy[4] };
    real_t amin = ax[0];
    if (ax[1] < amin) amin = ax[1];
    if (ax[2] < amin) amin = ax[2];
    for (int k = 0; k < 3; ++k) {
        if (ax[k] - amin >=  half_cyc) ax[k] -= cyc;
        if (ax[k] - amin <  -half_cyc) ax[k] += cyc;
    }
    *cx = (ax[0] + ax[1] + ax[2]) / 3.0;
    *cy = (xy[1] + xy[3] + xy[5]) / 3.0;
}

static void f_gradient_sca(const real_t xy[6], real_t elem_cos, real_t elem_area, real_t gs[6])
{
    const real_t cyc      = FESOM_CYCLIC_LENGTH_RAD;
    const real_t half_cyc = 0.5 * cyc;
    const real_t r_earth  = (real_t)FESOM_R_EARTH;
    real_t dX31 = xy[4] - xy[0];
    if (dX31 >  half_cyc) dX31 -= cyc;
    if (dX31 < -half_cyc) dX31 += cyc;
    dX31 *= elem_cos;
    real_t dX21 = xy[2] - xy[0];
    if (dX21 >  half_cyc) dX21 -= cyc;
    if (dX21 < -half_cyc) dX21 += cyc;
    dX21 *= elem_cos;
    real_t dY31 = xy[5] - xy[1];
    real_t dY21 = xy[3] - xy[1];
    real_t dfactor = -0.5 * r_earth / elem_area;
    gs[0] = (-dY31 + dY21) * dfactor;
    gs[1] = ( dY31)        * dfactor;
    gs[2] = (-dY21)        * dfactor;
    gs[3] = ( dX31 - dX21) * dfactor;
    gs[4] = (-dX31)        * dfactor;
    gs[5] = ( dX21)        * dfactor;
}

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

int fesom_evpwide_next(void) { return g_w.hook_done ? g_w.next : 0; }

/* ---- stage A: scatter-time discovery -------------------------------------- */
void fesom_evpwide_mesh_hook(struct fesom_mesh *m, struct fesom_partit *p,
                             const int *node_g2l, const int *elem_g2l)
{
    const int K = fesom_evpwide_env_K();
    if (K <= 0 || p->npes <= 1) return;
    FESOM_CHECK(K <= 60, "evpwide: K=%d unreasonable (rings 2K-1)", K);
    const int R = 2 * K - 1;

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
     * refs contain -1.) Vertex slots resolved to the unified space now. */
    std::vector<int> egid1, evert;
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
        for (int k = 0; k < 3; ++k) evert.push_back(slots[k]);
    }
    g_w.Eg = (int)egid1.size();
    g_w.eg_gid1 = std::move(egid1);
    g_w.eg_vert = std::move(evert);
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

    /* ghost geometry (verbatim formulas) + the formula cross-check on owned elements. */
    {
        for (int e = 0; e < E && e < 64; ++e) {
            real_t xy[6];
            for (int k = 0; k < 3; ++k) {
                const int s = m->elem_nodes[3*e + k];
                slot_xy(s, &xy[2*k], &xy[2*k + 1]);
            }
            const real_t ea = f_elem_area(xy);
            real_t cx, cy;  f_elem_center(xy, &cx, &cy);
            const real_t ec = cos(cy);
            real_t gs[6];   f_gradient_sca(xy, ec, ea, gs);
            FESOM_CHECK(ea == m->elem_area[e] && ec == m->elem_cos[e],
                        "evpwide: geometry formula drift at owned elem %d (area %.17g vs %.17g)",
                        e, ea, m->elem_area[e]);
            for (int q = 0; q < 6; ++q)
                FESOM_CHECK(gs[q] == m->gradient_sca[6*e + q],
                            "evpwide: gradient_sca formula drift at owned elem %d[%d]", e, q);
        }
    }
    std::vector<real_t> gs_g((size_t)Eg * 6), ea_g((size_t)Eg), mf_g((size_t)Eg);
    {
        const real_t r_earth = (real_t)FESOM_R_EARTH;
        for (int eg = 0; eg < Eg; ++eg) {
            real_t xy[6];
            for (int k = 0; k < 3; ++k)
                slot_xy(S.eg_vert[(size_t)3*eg + k], &xy[2*k], &xy[2*k + 1]);
            const real_t ea = f_elem_area(xy);
            real_t cx, cy;  f_elem_center(xy, &cx, &cy);
            const real_t ec = cos(cy);
            ea_g[(size_t)eg] = ea;
            mf_g[(size_t)eg] = tan(cy) / r_earth;   /* metric_factor, fesom_mesh.cpp:882 */
            f_gradient_sca(xy, ec, ea, &gs_g[(size_t)6*eg]);
        }
    }

    /* -------- handshake: wants grouped by owner; replies carry area0 + owner-order
     * incident-element gid lists for updatable slots. Tags 2201/2202/2203. */
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

    /* replies + the runtime SEND lists (owned indices, requester's want order). */
    std::vector<std::vector<real_t>> rep_d((size_t)npes);    /* area0 per want */
    std::vector<std::vector<int>>    rep_i((size_t)npes);    /* per flagged want: cnt, egid1... */
    std::vector<std::vector<int>>    slist((size_t)npes);
    const int nl = m->nl;
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
            ind[(size_t)q].resize((size_t)nw);
            rq.push_back(MPI_Request());
            MPI_Irecv(ind[(size_t)q].data(), nw, MPI_DOUBLE, q, 2202, comm, &rq.back());
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

    /* -------- unified statics + gather CSR. */
    std::vector<real_t> area0x((size_t)(N + next), 0.0), corx((size_t)(N + next), 0.0);
    std::vector<int>    coastx((size_t)(N + next), 0);
    for (int n = 0; n < N; ++n) {
        area0x[(size_t)n] = m->area[(size_t)n * (size_t)nl + 0];
        corx  [(size_t)n] = m->coriolis_node[n];
        coastx[(size_t)n] = S.gcoast[(size_t)n];
    }
    for (int t = 0; t < next; ++t) {
        corx  [(size_t)(N + t)] = 2.0 * (real_t)FESOM_OMEGA * sin(S.ext_geolat[(size_t)t]);
        coastx[(size_t)(N + t)] = S.gcoast[(size_t)(N + t)];
    }

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
                if (s < N && ind[(size_t)q][j] != area0x[(size_t)s]) {
                    const double d = fabs(ind[(size_t)q][j] - area0x[(size_t)s]);
                    if (d > area0_maxdiff) area0_maxdiff = d;
                }
                area0x[(size_t)s] = ind[(size_t)q][j];
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
        if (p->mype == 0)
            fprintf(stderr, "[evpwide] ring-1 area0: owner-vs-local max|diff| = %.3e "
                            "(expected ulp-level; owner bytes adopted for the replay)\n", a0mx);
    }

    /* -------- runtime exchange lists (cgpipe shape): ascending partner ranks. */
    S.partner.clear(); S.soff.assign(1, 0); S.roff.assign(1, 0);
    std::vector<int> sidx, ridx;
    std::vector<char> selff;
    for (int q = 0; q < npes; ++q) {
        const bool has = !slist[(size_t)q].empty() || !wslot[(size_t)q].empty();
        if (!has) continue;
        S.partner.push_back(q);
        for (int n : slist[(size_t)q]) sidx.push_back(n);
        for (int s : wslot[(size_t)q]) {
            ridx.push_back(s);
            selff.push_back(S.dist[(size_t)s] <= (signed char)(K - 1) ? 1 : 0);
        }
        S.soff.push_back((int)sidx.size());
        S.roff.push_back((int)ridx.size());
    }
    S.nsend = (int)sidx.size();
    S.nrecv = (int)ridx.size();
    S.reqs.assign(2 * S.partner.size(), MPI_Request());

    /* -------- device pushes. */
    S.sidx_d  = push_dev("evpw.sidx", sidx);
    S.ridx_d  = push_dev("evpw.ridx", ridx);
    S.selff_d = push_dev("evpw.selff", selff);
    S.sbuf_d  = Kokkos::View<real_t*>("evpw.sbuf", (size_t)S.nsend * 11);
    S.rbuf_d  = Kokkos::View<real_t*>("evpw.rbuf", (size_t)S.nrecv * 11);

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

    S.selfcheck = 0;
    if (const char *sc = getenv("FESOM_EVPWIDE_SELFCHECK")) S.selfcheck = atoi(sc) != 0;

    long loc[5] = { (long)S.partner.size(), (long)S.nrecv, (long)S.nsend, (long)D.nUpd,
                    (long)(S.roff.empty() ? 0 : *std::max_element(S.roff.begin(), S.roff.end())) };
    long mx[5];
    MPI_Reduce(loc, mx, 5, MPI_LONG, MPI_MAX, 0, comm);
    if (p->mype == 0)
        fprintf(stderr, "[evpwide] built: partners(max)=%ld recv-slots(max)=%ld send-slots(max)=%ld "
                        "upd-slots(max)=%ld — wide EVP ACTIVE (K=%d, R=%d, msgs/step 120 -> %d+1, "
                        "widest msg ~%ld KB @2f)\n",
                mx[0], mx[1], mx[2], mx[3], K, R, 120 / K,
                (long)(mx[4] * 2 * 8 / 1024));
    S.built = true;
}

/* ---- runtime resolve ------------------------------------------------------- */
int fesom_evpwide_K(struct fesom_ice *ice, struct fesom_partit *p, struct fesom_mesh *m)
{
    const int K = fesom_evpwide_env_K();
    if (K <= 0) return 0;

    if (ice->whichEVP != 0) {
        if (!g_w.announced && p->mype == 0) {
            fprintf(stderr, "[fesom_speed] !! FESOM_SPEED_EVPWIDE requested but whichEVP=%d "
                            "(mEVP/aEVP has its own subcycle exchange) — the lever is NOT running.\n",
                    ice->whichEVP);
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
            fprintf(stderr, "[fesom_speed] FESOM_SPEED_EVPWIDE = %d (R=%d rings, exact replay, "
                            "EVP exchanges/step 120 -> %d+1)\n", K, g_w.R, 120 / K);
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

void evpw_exchange(struct fesom_ice *ice, struct fesom_partit *p, int nf)
{
    EvpwState &S = g_w;
    FESOM_CHECK(S.built, "evpwide: exchange before build");

    PackViews F;
    F.v[0]  = ice->uice_fld.d();
    F.v[1]  = ice->vice_fld.d();
    if (nf > 2) {
        F.v[2]  = ice->data[FESOM_ICE_AICE].values_fld.d();
        F.v[3]  = ice->data[FESOM_ICE_MICE].values_fld.d();
        F.v[4]  = ice->data[FESOM_ICE_MSNOW].values_fld.d();
        F.v[5]  = ice->srfoce_u_fld.d();
        F.v[6]  = ice->srfoce_v_fld.d();
        F.v[7]  = ice->stress_atmice_x_fld.d();
        F.v[8]  = ice->stress_atmice_y_fld.d();
        F.v[9]  = ice->data[FESOM_ICE_AICE].values_rhs_fld.d();
        F.v[10] = ice->data[FESOM_ICE_MICE].values_rhs_fld.d();
    }

    fesom_halo_prof_barrier(p);                 /* M5.17 split-instrumentation parity */

    {   /* pack owned values */
        auto sidx = S.sidx_d; auto sbuf = S.sbuf_d; const PackViews FF = F; const int nfl = nf;
        if (S.nsend > 0)
            Kokkos::parallel_for("fesom_evpwide_pack", Kokkos::RangePolicy<>(0, S.nsend),
                KOKKOS_LAMBDA(const int i) {
                    const int n = sidx(i);
                    for (int f = 0; f < nfl; ++f) sbuf(i*nfl + f) = FF.v[f](n);
                });
    }
    Kokkos::fence();   /* MANDATORY pre-MPI: MPI reads sbuf + re-posts rbuf (drains prev unpack) */

    int nreq = 0;
    real_t *sp = S.sbuf_d.data();
    real_t *rp = S.rbuf_d.data();
    double bytes = 0.0;
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
    fesom_halo_prof_waitall(nreq, S.reqs.data());
    fesom_halo_prof_bytes(bytes);

    /* selfcheck (nf==2 path): BEFORE unpack, refresh-vs-local on rings <= K-1.
     * MUST be exactly 0 on Serial (the exact-replay contract); ~rounding on CUDA. */
    if (nf == 2 && S.selfcheck) {
        auto ridx = S.ridx_d; auto rbuf = S.rbuf_d; auto flag = S.selff_d;
        auto u = F.v[0]; auto v = F.v[1];
        double mx = 0.0;
        Kokkos::parallel_reduce("fesom_evpwide_self", Kokkos::RangePolicy<>(0, S.nrecv),
            KOKKOS_LAMBDA(const int i, double &acc) {
                if (!flag(i)) return;
                const int s = ridx(i);
                double d = Kokkos::fabs((double)rbuf(i*2 + 0) - (double)u(s));
                const double d2 = Kokkos::fabs((double)rbuf(i*2 + 1) - (double)v(s));
                if (d2 > d) d = d2;
                if (d > acc) acc = d;
            }, Kokkos::Max<double>(mx));
        double gmx = 0.0;
        MPI_Reduce(&mx, &gmx, 1, MPI_DOUBLE, MPI_MAX, 0, p->MPI_COMM_FESOM);
        if (p->mype == 0) {
            printf("[evpwide-self] exch %ld  max|refresh-local| (rings<=K-1) = %.3e\n",
                   g_w.exch_count, gmx);
            fflush(stdout);
        }
    }
    ++g_w.exch_count;

    {   /* unpack into ghost slots */
        auto ridx = S.ridx_d; auto rbuf = S.rbuf_d; const PackViews FF = F; const int nfl = nf;
        if (S.nrecv > 0)
            Kokkos::parallel_for("fesom_evpwide_unpack", Kokkos::RangePolicy<>(0, S.nrecv),
                KOKKOS_LAMBDA(const int i) {
                    const int s = ridx(i);
                    for (int f = 0; f < nfl; ++f) FF.v[f](s) = rbuf(i*nfl + f);
                });
    }
    /* no post-unpack fence: consumers are same-stream kernels; the pre-MPI fence of the NEXT
     * exchange drains this unpack before rbuf is re-posted (the cgpipe/NOFENCE2 audit). */
}

} /* anonymous namespace */

void fesom_evpwide_prestep_exchange(struct fesom_ice *ice, struct fesom_partit *p)
{ evpw_exchange(ice, p, 11); }

void fesom_evpwide_subcycle_exchange(struct fesom_ice *ice, struct fesom_partit *p)
{ evpw_exchange(ice, p, 2); }

void fesom_evpwide_free(void)
{
    EvpwState &S = g_w;
    S.dev = FesomEvpwideDev();
    S.sidx_d = Kokkos::View<int*>();  S.ridx_d = Kokkos::View<int*>();
    S.selff_d = Kokkos::View<char*>();
    S.sbuf_d = Kokkos::View<real_t*>(); S.rbuf_d = Kokkos::View<real_t*>();
    S.built = false;
}
