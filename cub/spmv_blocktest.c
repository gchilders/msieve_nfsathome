/* Standalone correctness test for cub/spmv_engine.so multi-block usage.
 *
 * Verifies that splitting a sparse GF(2) matrix into column slices and
 * XOR-accumulating per-slice SpMV launches (exactly how msieve's
 * mul_packed_gpu drives the engine when block_nnz forces multiple blocks)
 * produces bitwise-identical output to (a) a single full-matrix launch and
 * (b) a CPU reference.
 *
 * Build: gcc -O2 -o spmv_blocktest spmv_blocktest.c -I/usr/local/cuda/include -lcuda -ldl
 * Run from the msieve tree root (dlopens ./cub/spmv_engine.so), GPU must be free.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda.h>

typedef struct { unsigned long long w[1]; } v_t;   /* VBITS=64 */

typedef struct {
    int num_rows;
    uint32_t num_col_entries;
    CUdeviceptr vector_in;
    CUdeviceptr vector_out;
    CUdeviceptr col_entries;
    CUdeviceptr row_entries;
} spmv_data_t;

typedef void* (*init_f)(int*);
typedef void  (*free_f)(void*);
typedef void  (*run_f)(void*, spmv_data_t*);

#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *s = "?"; cuGetErrorString(r, &s); \
    fprintf(stderr, "CUDA error %d (%s) at line %d\n", r, s, __LINE__); exit(2); } } while (0)

/* xorshift PRNG for reproducibility */
static uint64_t rng_state;
static uint64_t rnd64(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}
static uint32_t rnd32(uint32_t bound) { return (uint32_t)(rnd64() % bound); }

/* one matrix in COO-ish form: per-row sorted column lists */
typedef struct {
    uint32_t nrows, ncols;
    uint32_t nnz;
    uint32_t *rowptr;   /* nrows+1 */
    uint32_t *colidx;   /* nnz, sorted within row */
} csr_t;

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/* Generate a matrix mimicking the NFS normal direction: most rows light,
 * a band of very heavy rows (like ideal rows just below the dense cutoff),
 * and some globally empty rows. */
static csr_t gen_matrix(uint32_t nrows, uint32_t ncols,
                        uint32_t light_max, uint32_t num_heavy,
                        uint32_t heavy_weight) {
    csr_t m;
    m.nrows = nrows; m.ncols = ncols;
    uint32_t *weights = calloc(nrows, sizeof(uint32_t));
    uint64_t total = 0;
    for (uint32_t r = 0; r < nrows; r++) {
        uint32_t w;
        uint32_t roll = rnd32(100);
        if (roll < 5)       w = 0;                       /* empty row */
        else if (roll < 95) w = 1 + rnd32(light_max);    /* light row */
        else                w = light_max + rnd32(4 * light_max);
        weights[r] = w;
        total += w;
    }
    for (uint32_t h = 0; h < num_heavy; h++) {   /* heavy rows up front, like NFS */
        uint32_t r = h;
        total -= weights[r];
        weights[r] = heavy_weight;
        total += heavy_weight;
    }
    m.nnz = (uint32_t)total;
    m.rowptr = malloc((nrows + 1) * sizeof(uint32_t));
    m.colidx = malloc(total * sizeof(uint32_t));
    uint64_t pos = 0;
    for (uint32_t r = 0; r < nrows; r++) {
        m.rowptr[r] = (uint32_t)pos;
        for (uint32_t j = 0; j < weights[r]; j++)
            m.colidx[pos + j] = rnd32(ncols);
        qsort(m.colidx + pos, weights[r], sizeof(uint32_t), cmp_u32);
        pos += weights[r];
    }
    m.rowptr[nrows] = (uint32_t)pos;
    free(weights);
    return m;
}

/* CPU reference: y[r] = XOR over row entries of x[col] */
static void cpu_ref(const csr_t *m, const v_t *x, v_t *y) {
    memset(y, 0, m->nrows * sizeof(v_t));
    for (uint32_t r = 0; r < m->nrows; r++)
        for (uint32_t j = m->rowptr[r]; j < m->rowptr[r + 1]; j++)
            y[r].w[0] ^= x[m->colidx[j]].w[0];
}

/* Extract the column slice [c0, c1) of m as a self-contained CSR with
 * rebased column indices — mirrors msieve's extract_block/pack_matrix_block */
static csr_t col_slice(const csr_t *m, uint32_t c0, uint32_t c1) {
    csr_t s;
    s.nrows = m->nrows; s.ncols = c1 - c0;
    s.rowptr = malloc((m->nrows + 1) * sizeof(uint32_t));
    uint64_t cnt = 0;
    for (uint32_t r = 0; r < m->nrows; r++) {
        s.rowptr[r] = (uint32_t)cnt;
        for (uint32_t j = m->rowptr[r]; j < m->rowptr[r + 1]; j++)
            if (m->colidx[j] >= c0 && m->colidx[j] < c1) cnt++;
    }
    s.rowptr[m->nrows] = (uint32_t)cnt;
    s.nnz = (uint32_t)cnt;
    s.colidx = malloc(cnt * sizeof(uint32_t));
    cnt = 0;
    for (uint32_t r = 0; r < m->nrows; r++)
        for (uint32_t j = m->rowptr[r]; j < m->rowptr[r + 1]; j++)
            if (m->colidx[j] >= c0 && m->colidx[j] < c1)
                s.colidx[cnt++] = m->colidx[j] - c0;
    return s;
}

static void free_csr(csr_t *m) { free(m->rowptr); free(m->colidx); }

/* upload a CSR and run the engine once, XOR-accumulating into d_y */
static void run_block(run_f run, void *eng, const csr_t *m,
                      CUdeviceptr d_x_off, CUdeviceptr d_y) {
    spmv_data_t sd;
    CUdeviceptr d_rp, d_ci = 0;
    CU(cuMemAlloc(&d_rp, (m->nrows + 1) * sizeof(uint32_t)));
    CU(cuMemcpyHtoD(d_rp, m->rowptr, (m->nrows + 1) * sizeof(uint32_t)));
    if (m->nnz) {
        CU(cuMemAlloc(&d_ci, m->nnz * sizeof(uint32_t)));
        CU(cuMemcpyHtoD(d_ci, m->colidx, m->nnz * sizeof(uint32_t)));
    }
    sd.num_rows = (int)m->nrows;
    sd.num_col_entries = m->nnz;
    sd.vector_in = d_x_off;
    sd.vector_out = d_y;
    sd.col_entries = d_ci;
    sd.row_entries = d_rp;
    run(eng, &sd);
    CU(cuCtxSynchronize());
    CU(cuMemFree(d_rp));
    if (d_ci) CU(cuMemFree(d_ci));
}

static int compare(const char *what, const v_t *a, const v_t *b, uint32_t n) {
    uint32_t bad = 0, first = 0;
    for (uint32_t i = 0; i < n; i++)
        if (a[i].w[0] != b[i].w[0]) { if (!bad) first = i; bad++; }
    if (bad)
        printf("FAIL  %-40s %u/%u rows differ (first at row %u)\n", what, bad, n, first);
    else
        printf("pass  %-40s\n", what);
    return bad != 0;
}

int main(int argc, char **argv) {
    const char *so_path = (argc > 1) ? argv[1] : "./cub/spmv_engine.so";
    void *dso = dlopen(so_path, RTLD_NOW);
    if (!dso) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    init_f einit = (init_f)dlsym(dso, "spmv_engine_init");
    free_f efree = (free_f)dlsym(dso, "spmv_engine_free");
    run_f  erun  = (run_f)dlsym(dso, "spmv_engine_run");
    if (!einit || !efree || !erun) { fprintf(stderr, "dlsym failed\n"); return 2; }

    CU(cuInit(0));
    CUdevice dev; CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CU(cuCtxCreate(&ctx, NULL, 0, dev));

    int failures = 0;

    /* two shapes to push autotune into different warp_items variants */
    struct { const char *name; uint32_t nrows, ncols, light, heavy_n, heavy_w; } shapes[] = {
        { "light-rows",  1500000, 1500000, 20,  64, 200000 },
        { "heavy-rows",  400000,  400000,  300, 64, 400000 },
    };

    for (int s = 0; s < 2; s++) {
        rng_state = 0x9E3779B97F4A7C15ULL + s;
        csr_t m = gen_matrix(shapes[s].nrows, shapes[s].ncols,
                             shapes[s].light, shapes[s].heavy_n, shapes[s].heavy_w);
        uint32_t nr = m.nrows, nc = m.ncols;
        printf("== shape %s: %u x %u, nnz=%u\n", shapes[s].name, nr, nc, m.nnz);

        v_t *x = malloc(nc * sizeof(v_t));
        for (uint32_t i = 0; i < nc; i++) x[i].w[0] = rnd64();
        v_t *yref = malloc(nr * sizeof(v_t));
        v_t *ygpu = malloc(nr * sizeof(v_t));
        cpu_ref(&m, x, yref);

        CUdeviceptr d_x, d_y;
        CU(cuMemAlloc(&d_x, nc * sizeof(v_t)));
        CU(cuMemcpyHtoD(d_x, x, nc * sizeof(v_t)));
        CU(cuMemAlloc(&d_y, nr * sizeof(v_t)));

        /* fresh engine per shape so each shape's stats drive the autotune */
        int vbits = 0;
        void *eng = einit(&vbits);
        if (vbits != 64) { fprintf(stderr, "engine VBITS=%d, expected 64\n", vbits); return 2; }

        /* single full-matrix launch */
        CU(cuMemsetD8(d_y, 0, nr * sizeof(v_t)));
        run_block(erun, eng, &m, d_x, d_y);
        CU(cuMemcpyDtoH(ygpu, d_y, nr * sizeof(v_t)));
        failures += compare("single block vs CPU", ygpu, yref, nr);

        /* multi-block: K column slices, uneven boundaries, accumulate */
        uint32_t ks[] = { 2, 4, 7, 14 };
        for (int ki = 0; ki < 4; ki++) {
            uint32_t K = ks[ki];
            CU(cuMemsetD8(d_y, 0, nr * sizeof(v_t)));
            uint32_t c0 = 0;
            for (uint32_t k = 0; k < K; k++) {
                uint32_t c1 = (k == K - 1) ? nc :
                    c0 + (nc / K) + (rnd32(nc / (4 * K)) - nc / (8 * K));
                if (c1 > nc) c1 = nc;
                if (c1 <= c0) c1 = c0 + 1;
                csr_t sl = col_slice(&m, c0, c1);
                run_block(erun, eng, &sl,
                          (CUdeviceptr)((v_t *)d_x + c0), d_y);
                free_csr(&sl);
                c0 = c1;
            }
            CU(cuMemcpyDtoH(ygpu, d_y, nr * sizeof(v_t)));
            char what[64];
            snprintf(what, sizeof(what), "%u column slices vs CPU", K);
            failures += compare(what, ygpu, yref, nr);
        }

        efree(eng);
        CU(cuMemFree(d_x)); CU(cuMemFree(d_y));
        free(x); free(yref); free(ygpu); free_csr(&m);
    }

    printf(failures ? "\nRESULT: %d FAILURES\n" : "\nRESULT: all tests passed\n", failures);
    return failures ? 1 : 0;
}
