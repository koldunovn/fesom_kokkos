/*
 * Simple per-rank binary restart — see fesom_restart.h. Each rank writes/reads its
 * own local prognostic arrays (no gather). Fields are matched by NAME on read, so
 * adding/removing a field across versions degrades gracefully (skip / warn).
 */
#include "fesom_restart.h"
#include "fesom_types.h"        /* FESOM_DIE */
#include "fesom_field.hpp"
#include "fesom_mesh.h"
#include "fesom_dyn.h"
#include "fesom_tracers.h"
#include "fesom_ice_types.h"

#include <mpi.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace {

struct FEntry { std::string name; fesom::Field *f; };

static void add(std::vector<FEntry> &v, const std::string &nm, fesom::Field &f) {
    if (f.allocated()) v.push_back({nm, &f});   /* skip un-allocated optional fields */
}

/* The prognostic state needed to continue the run. EVP sigma is intentionally NOT
 * saved (re-relaxed each step by the subcycling); add it here if the continuity
 * check shows an ice-dynamics transient at restart. */
static std::vector<FEntry> build_list(fesom_mesh *mesh, fesom_dyn *dyn,
                                      fesom_tracers *tr, fesom_ice *ice) {
    std::vector<FEntry> v;
    /* ALE layer thicknesses — carried across the step boundary (a step's substeps 1-13
     * read the thickness committed by the PREVIOUS step; in linfs the surface layer
     * carries eta). Without these, resume reads the IC (eta=0) thickness -> transient. */
    add(v, "mesh.hnode",     mesh->hnode_fld);
    add(v, "mesh.hnode_new", mesh->hnode_new_fld);
    add(v, "mesh.helem",     mesh->helem_fld);
    add(v, "mesh.hbar",      mesh->hbar_fld);
    add(v, "mesh.hbar_old",  mesh->hbar_old_fld);
    add(v, "dyn.uv",          dyn->uv_fld);
    add(v, "dyn.uvnode",      dyn->uvnode_fld);   /* node-vel used in next step's forcing before recompute */
    add(v, "dyn.uv_rhsAB",    dyn->uv_rhsAB_fld);
    add(v, "dyn.eta_n",       dyn->eta_n_fld);
    add(v, "dyn.d_eta",       dyn->d_eta_fld);
    add(v, "dyn.ssh_rhs",     dyn->ssh_rhs_fld);
    add(v, "dyn.ssh_rhs_old", dyn->ssh_rhs_old_fld);
    for (int t = 0; t < tr->num_tracers; ++t) {
        char p[16]; snprintf(p, sizeof p, "tr%d.", t);
        add(v, std::string(p) + "values",    tr->data[t].values_fld);
        add(v, std::string(p) + "valuesAB",  tr->data[t].valuesAB_fld);
        add(v, std::string(p) + "valuesold", tr->data[t].valuesold_fld);
    }
    if (ice) {
        for (int t = 0; t < FESOM_NUM_ICE_TRACERS; ++t) {
            char p[16]; snprintf(p, sizeof p, "ice%d.", t);
            add(v, std::string(p) + "values",     ice->data[t].values_fld);
            add(v, std::string(p) + "values_old", ice->data[t].values_old_fld);
        }
        add(v, "ice.uice",     ice->uice_fld);
        add(v, "ice.uice_old", ice->uice_old_fld);
        add(v, "ice.vice",     ice->vice_fld);
        add(v, "ice.vice_old", ice->vice_old_fld);
        /* Thermodynamic state carried across the boundary (thdgr_old = prev growth rate,
         * feeds the ice->ocean salt/heat flux; t_skin = skin temperature). */
        add(v, "ice.thdgr",     ice->thermo.thdgr_fld);
        add(v, "ice.thdgr_old", ice->thermo.thdgr_old_fld);
        add(v, "ice.thdgrsn",   ice->thermo.thdgrsn_fld);
        add(v, "ice.thdgra",    ice->thermo.thdgra_fld);
        add(v, "ice.t_skin",    ice->thermo.t_skin_fld);
    }
    return v;
}

static const uint32_t MAGIC   = 0xFE50FE50u;
static const uint32_t VERSION = 1u;

static void path_for(char *buf, size_t n, const char *dir, int rank) {
    snprintf(buf, n, "%s/restart_%06d.bin", dir, rank);
}

} /* namespace */

int fesom_restart_write(const char *dir, int step, const fesom_calendar_t *cal,
                        fesom_mesh *mesh, fesom_dyn *dyn, fesom_tracers *tr, fesom_ice *ice,
                        int mype, int npes)
{
    char path[2048]; path_for(path, sizeof path, dir, mype);
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "[restart] rank %d: cannot open %s for write\n", mype, path); return 1; }

    std::vector<FEntry> list = build_list(mesh, dyn, tr, ice);
    uint32_t magic = MAGIC, version = VERSION;
    int32_t npes_file = npes, st = step, nf = (int32_t)list.size();
    fwrite(&magic,     sizeof magic,     1, fp);
    fwrite(&version,   sizeof version,   1, fp);
    fwrite(&npes_file, sizeof npes_file, 1, fp);
    fwrite(&st,      sizeof st,      1, fp);
    fwrite(cal,      sizeof *cal,    1, fp);
    fwrite(&nf,      sizeof nf,      1, fp);
    for (auto &e : list) {
        e.f->sync_host();                          /* device -> host (no-op on CPU backends) */
        int32_t  L   = (int32_t)e.name.size();
        uint64_t cnt = (uint64_t)e.f->size();
        fwrite(&L, sizeof L, 1, fp);
        fwrite(e.name.data(), 1, (size_t)L, fp);
        fwrite(&cnt, sizeof cnt, 1, fp);
        fwrite(e.f->h(), sizeof(double), (size_t)cnt, fp);
    }
    fclose(fp);
    if (mype == 0)
        printf("[restart] wrote step %d (%04d-%02d-%02d %02d:%02d), %d fields/rank -> %s/restart_*.bin\n",
               step, cal->year, cal->month, cal->day, cal->hour, cal->minute, (int)list.size(), dir);
    return 0;
}

int fesom_restart_read(const char *dir, int *step, fesom_calendar_t *cal,
                       fesom_mesh *mesh, fesom_dyn *dyn, fesom_tracers *tr, fesom_ice *ice,
                       int mype, int npes)
{
    char path[2048]; path_for(path, sizeof path, dir, mype);
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;                              /* no checkpoint -> fresh start */

    uint32_t magic = 0, version = 0; int32_t npes_file = 0, st = 0, nf = 0;
    if (fread(&magic, sizeof magic, 1, fp) != 1 || magic != MAGIC)
        FESOM_DIE("restart: %s bad magic (corrupt checkpoint)", path);
    if (fread(&version, sizeof version, 1, fp) != 1 || version != VERSION)
        FESOM_DIE("restart: %s version mismatch (got %u, expected %u)", path, version, VERSION);
    if (fread(&npes_file, sizeof npes_file, 1, fp) != 1 || npes_file != npes)
        FESOM_DIE("restart: %s written for npes=%d, resuming on npes=%d (must match)", path, npes_file, npes);
    if (fread(&st, sizeof st, 1, fp) != 1) FESOM_DIE("restart: %s short read (step)", path);
    if (fread(cal, sizeof *cal, 1, fp) != 1) FESOM_DIE("restart: %s short read (calendar)", path);
    if (fread(&nf, sizeof nf, 1, fp) != 1) FESOM_DIE("restart: %s short read (nfields)", path);

    std::vector<FEntry> list = build_list(mesh, dyn, tr, ice);
    int loaded = 0;
    for (int i = 0; i < nf; ++i) {
        int32_t L = 0; uint64_t cnt = 0;
        char nm[256];
        if (fread(&L, sizeof L, 1, fp) != 1 || L < 0 || L >= (int32_t)sizeof nm)
            FESOM_DIE("restart: %s bad field-name length", path);
        if (fread(nm, 1, (size_t)L, fp) != (size_t)L) FESOM_DIE("restart: %s short read (name)", path);
        nm[L] = '\0';
        if (fread(&cnt, sizeof cnt, 1, fp) != 1) FESOM_DIE("restart: %s short read (count)", path);

        fesom::Field *f = nullptr;
        for (auto &e : list) if (e.name == nm) { f = e.f; break; }
        if (!f) {                                   /* field in file but not in model: skip */
            fseek(fp, (long)(cnt * sizeof(double)), SEEK_CUR);
            if (mype == 0)
                fprintf(stderr, "[restart] warning: field '%s' in file not used by model — skipped\n", nm);
            continue;
        }
        if (cnt != (uint64_t)f->size())
            FESOM_DIE("restart: field '%s' count %llu != model size %zu (mesh/partition changed?)",
                      nm, (unsigned long long)cnt, f->size());
        if (fread(f->h(), sizeof(double), (size_t)cnt, fp) != cnt)
            FESOM_DIE("restart: %s short read on field '%s'", path, nm);
        f->modify_host(); f->sync_device();         /* push restored host -> device */
        ++loaded;
    }
    fclose(fp);
    *step = st;
    if (mype == 0)
        printf("[restart] RESUMED from step %d (%04d-%02d-%02d %02d:%02d), %d fields/rank from %s/restart_*.bin\n",
               st, cal->year, cal->month, cal->day, cal->hour, cal->minute, loaded, dir);
    return 1;
}
