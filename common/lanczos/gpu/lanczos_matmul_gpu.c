/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id$
--------------------------------------------------------------------*/

#include "lanczos_gpu.h"
#if !defined(WIN32) && !defined(_WIN64)
#include <unistd.h>
#endif
#ifdef MSIEVE_CUDA_SINGLE_BINARY
#include <cuda_embedded.h>

/* Built-in CUB engine entry points. These are linked directly into msieve
   in single-binary CUDA builds; keep the public DSO header unchanged. */
extern void *spmv_engine_init(int *vbits);
extern void spmv_engine_free(void *engine);
extern void spmv_engine_run(void *engine, spmv_data_t *spmv_data);
extern void spmv_engine_run_trans(void *engine, spmv_data_t *spmv_data);
extern void spmv_engine_set_kernel(void *engine, int kernel);
#endif
#include "lanczos_gpu_core.h"
#include "lanczos_nvtx.h"

static const char * gpu_kernel_names[] = 
{
	"lanczos_kernel_mask",
	"lanczos_kernel_xor",
	"lanczos_kernel_inner_prod",
	"lanczos_kernel_outer_prod",
};
 
typedef struct {
	uint32 row_off;
	uint32 col_off;
} entry_idx_t;

/* Largest block_nnz we accept. One block's nonzero count has to fit in a
   uint32 all the way to the kernels (block_row_t::num_col_entries and
   spmv_data_t::num_col_entries), and the CSR row pointers are uint32 too,
   so 2^32-1 is the hard ceiling. We stop ~295M short of it because
   extract_block() never splits a column, and so overshoots the request by
   up to one column's weight; no real column comes anywhere near that much
   slack. Every host-side nonzero count below is 64-bit, so nothing wraps
   while a block is being assembled -- that, not the kernels, was what
   limited this to 1750000000 before (11 * (nnz/10) overflowed a uint32).

   Note that a block this large is expensive to build on the host: the CSR
   column array costs 4 bytes per nonzero and radix_sort() another 16, so
   a full 4e9-nonzero block needs ~80GB of host memory in transit, even
   though only 16GB of it lands on the card. */

#define MAX_BLOCK_NNZ 4000000000u

/* Largest block that actually gets built. The SpMV kernels index
   nonzeros in uint32 and step past a block's end by up to one warp
   segment (TWarpItems <= 2048) plus a warp-strided load before they
   compare, so a block right at 2^32-1 would wrap; keep 2^20 clear */

#define MAX_CSR_BLOCK_NNZ ((uint64)UINT32_MAX - (1u << 20))

#if 0
// Tried using compressible memory on an A100. Did not help 
static CUresult setProp(CUmemAllocationProp *prop, int UseCompressibleMemory)
{
    CUdevice currentDevice;
    CUDA_TRY(cuCtxGetDevice(&currentDevice))

    memset(prop, 0, sizeof(CUmemAllocationProp));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = currentDevice;

    if (UseCompressibleMemory)
        prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    return CUDA_SUCCESS;
}

CUresult allocateCompressible(void **adr, size_t size, int UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    setProp(&prop, UseCompressibleMemory);

    size_t granularity = 0;
    CUDA_TRY(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    size = ((size - 1) / granularity + 1) * granularity;
    CUdeviceptr dptr;
    CUDA_TRY(cuMemAddressReserve(&dptr, size, 0, 0, 0))
    
	CUmemGenericAllocationHandle allocationHandle;
    CUDA_TRY(cuMemCreate(&allocationHandle, size, &prop, 0))

    // Check if cuMemCreate was able to allocate compressible memory.
    if (UseCompressibleMemory) {
        CUmemAllocationProp allocationProp = {};
        cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
        if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
            printf("Could not allocate compressible memory...\n");
            exit(-1);
        }
    }

    CUDA_TRY(cuMemMap(dptr, size, 0, allocationHandle, 0))
    CUDA_TRY(cuMemRelease(allocationHandle))

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUDA_TRY(cuMemSetAccess(dptr, size, &accessDescriptor, 1))

    *adr = (void *)dptr;
    return CUDA_SUCCESS;
}

CUresult freeCompressible(void *ptr, size_t size, int UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    setProp(&prop, UseCompressibleMemory);

    size_t granularity = 0;
    CUDA_TRY(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    size = ((size - 1) / granularity + 1) * granularity;

    if (ptr == NULL) return CUDA_SUCCESS;
    if (cuMemUnmap((CUdeviceptr)ptr, size) != CUDA_SUCCESS ||
        cuMemAddressFree((CUdeviceptr)ptr, size) != CUDA_SUCCESS)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}
#endif

/*-------------------------------------------------------------------*/
static void copy_dense(packed_matrix_t *p) 
{
	/* copy the dense arrays to device memory */

	uint32 i, j, k;
	uint32 ncols = p->ncols;
	gpudata_t *d = (gpudata_t *)p->extra;
	uint32 num_dense_blocks = (p->num_dense_rows + VBITS - 1) / VBITS;
	v_t *tmp = (v_t *)xmalloc(ncols * sizeof(v_t));

	d->dense_blocks = (CUdeviceptr *)xmalloc(num_dense_blocks *
						sizeof(CUdeviceptr));

	for (i = 0; i < num_dense_blocks; i++) {

		for (j = 0; j < ncols; j++) {
			la_col_t *col = p->unpacked_cols + j;
			uint32 *src = col->data + col->weight;
			for (k = 0; k < VWORDS; k++) {
				uint32 t = i * VWORDS + k;
				tmp[j].w[k] = (uint64)src[2 * t + 1] << 32 |
					(uint64)src[2 * t];
			}
		}

		if (d->use_cudamanaged) {
			CUDA_TRY(cuMemAllocManaged(&d->dense_blocks[i],
				ncols * sizeof(v_t),
				CU_MEM_ATTACH_GLOBAL))
			CUDA_TRY(cuMemcpy(d->dense_blocks[i],
				(CUdeviceptr) tmp,
				ncols * sizeof(v_t)))
			CUDA_TRY(my_cuMemAdvise(d->dense_blocks[i],
				ncols * sizeof(v_t),
				CU_MEM_ADVISE_SET_READ_MOSTLY,
				d->gpu_info->device_handle))
		} else {
			/* CUDA_TRY(allocateCompressible((void **)&d->dense_blocks[i], ncols * sizeof(v_t), 1)) */
			CUDA_TRY(cuMemAlloc(&d->dense_blocks[i],
					ncols * sizeof(v_t)))
			CUDA_TRY(cuMemcpyHtoD(d->dense_blocks[i], tmp,
					ncols * sizeof(v_t)))
		}
	}

	free(tmp);
}

/*-------------------------------------------------------------------*/
static uint64 extract_block(la_col_t *cols,
			uint32 row_min, uint32 row_max,
			uint32 col_min, uint32 col_max,
			uint64 nnz, uint32 *blocksize,
			entry_idx_t **entries_in,
			uint64 *max_entries_in)
{
	uint32 i, j;
	uint64 num_entries = 0;
	entry_idx_t *entries = *entries_in;
	uint64 max_entries = *max_entries_in;

	for (i = col_min; (i < col_max) && (num_entries < nnz); i++) {

		la_col_t *col = cols + i;

		for (j = 0; j < col->weight; j++) {
			uint32 idx = col->data[j];

			if (idx >= row_max)
				break;

			if (idx >= row_min) {

				entry_idx_t *e;

				if (num_entries == max_entries) {
					max_entries *= 2;
					entries = (entry_idx_t *)xrealloc(
							entries, 
							max_entries *
							sizeof(entry_idx_t));
				}

				e = entries + num_entries++;
				e->row_off = idx;
				e->col_off = i;
			}
		}
	}

	*blocksize = i - col_min;
	*entries_in = entries;
	*max_entries_in = max_entries;
	return num_entries;
}

/*-------------------------------------------------------------------*/
static uint64 extract_rows(la_col_t *cols, uint32 ncols,
			uint32 row_min, uint32 row_max,
			entry_idx_t **entries_in,
			uint64 *max_entries_in)
{
	/* collect the nonzeros in rows [row_min, row_max) of every
	   column, for one block of the transpose. Where its rows start
	   and end is decided beforehand, by plan_trans_blocks */

	uint32 i, j;
	uint64 num_entries = 0;
	entry_idx_t *entries = *entries_in;
	uint64 max_entries = *max_entries_in;

	for (i = 0; i < ncols; i++) {

		la_col_t *col = cols + i;

		for (j = 0; j < col->weight; j++) {
			uint32 idx = col->data[j];

			if (idx >= row_max)
				break;

			if (idx >= row_min) {

				entry_idx_t *e;

				if (num_entries == max_entries) {
					max_entries *= 2;
					entries = (entry_idx_t *)xrealloc(
							entries, 
							max_entries *
							sizeof(entry_idx_t));
				}

				e = entries + num_entries++;
				e->row_off = idx;
				e->col_off = i;
			}
		}
	}

	*entries_in = entries;
	*max_entries_in = max_entries;
	return num_entries;
}

/*-------------------------------------------------------------------*/
static int compare_row_off(const void *x, const void *y) {
	entry_idx_t *xx = (entry_idx_t *)x;
	entry_idx_t *yy = (entry_idx_t *)y;

	if (xx->row_off > yy->row_off)
		return 1;
	if (xx->row_off < yy->row_off)
		return -1;

	return (int)xx->col_off - (int)yy->col_off;
}

/*-------------------------------------------------------------------*/
static void radix_sort(entry_idx_t *arr, uint64 n) {

	/* simple radix sort, much faster than qsort() */

	uint64 i;
	uint32 pass, skip;
	uint64 *a, *b, *from, *to, *temp;

	/* xmalloc, not malloc: these are 8n bytes each, so at the top of
	   the block_nnz range they are tens of GB and a failure here is
	   entirely plausible -- better to say so than to segfault */

	a = (uint64 *) xmalloc(n * sizeof(uint64));
	b = (uint64 *) xmalloc(n * sizeof(uint64));

	for (i = 0; i < n; i++) {
		entry_idx_t *e = arr + i;
		a[i] = ((uint64)(e->row_off) << 32) | (uint64)(e->col_off);
	}

	from = a;
	to = b;
	skip = 0;
	for (pass = 0; pass < 8; pass++)  {
		uint64 box[256] = { 0 };

		for (i = 0; i < n; i++) box[ (from[i] >> (8*pass)) & 255]++;
		if (box[0] == n) { /* this word is all 0's, don't need to sort */
			skip++;
			continue;
		}
		for (i = 1; i < 256; i++) box[i] += box[i-1];
		for (i = n - 1; i != (uint64)(-1); i--) to[--box[(from[i] >> (8*pass)) & 255]] = from[i];

		temp = from;
		from = to;
		to = temp;
	}

	if (skip & 1) to = b;
	else to = a;
	for (i = 0; i < n; i++) {
		entry_idx_t *e = arr + i;
		e->row_off = (uint32)(to[i] >> 32);
		e->col_off = (uint32)(to[i]);
	}

	free(a);
	free(b);
}

/*-------------------------------------------------------------------*/
static size_t block_bytes(uint64 num_entries, uint32 num_rows) {

	/* a packed block: column indices padded to 256 bytes, then
	   the row pointers */

	return ((((size_t)num_entries + 63) & ~(size_t)63) +
			num_rows + 1) * sizeof(uint32);
}

/*-------------------------------------------------------------------*/
static void pack_matrix_block(block_row_t *b,
			entry_idx_t *entries, uint64 num_entries,
			uint32 row_min, uint32 row_max,
			uint32 col_min, uint32 col_max,
			uint32 is_trans, uint32 streamed)
{

	uint64 i, j;
	uint32 num_rows = row_max - row_min;
	uint32 *col_entries;
	uint32 *row_entries;

	/* the CSR arrays below, and everything downstream of them, index
	   nonzeros with a uint32; the block extractors are capped so this
	   cannot happen, but a wrong answer here would be silent */

	if (num_entries > MAX_CSR_BLOCK_NNZ) {
		printf("error: matrix block holds %" PRIu64 " nonzeros, above "
			"the %" PRIu64 " a CSR block can hold; lower block_nnz\n",
			num_entries, MAX_CSR_BLOCK_NNZ);
		exit(-1);
	}

	/* convert a block of matrix rows from COO to CSR format, in one
	   host buffer: column indices, then the row pointers starting on
	   a 256-byte boundary. A block that will be streamed goes
	   straight into pinned memory. If the system won't pin any more
	   (WSL limits pinned memory), it stays in ordinary memory: its
	   copies then run several times slower, but the solve goes on */

	b->row_offset = ((size_t)num_entries + 63) & ~(size_t)63;
	b->bytes = block_bytes(num_entries, num_rows);
	b->streamed = streamed;
	if (streamed &&
	    cuMemHostAlloc((void **)&b->host_data, b->bytes, 0) ==
	    						CUDA_SUCCESS) {
		b->pinned = 1;
	}
	else {
		static uint32 warned = 0;

		if (streamed && !warned) {
			printf("warning: cannot pin host memory for a "
				"streamed block; copying it from ordinary "
				"memory instead, which is slower\n");
			warned = 1;
		}
		b->host_data = (uint32 *)xmalloc(b->bytes);
	}
	col_entries = b->host_data;
	row_entries = b->host_data + b->row_offset;
	memset(row_entries, 0, (num_rows + 1) * sizeof(uint32));

	if (is_trans) {
		for (i = 0; i < num_entries; i++) {
			entry_idx_t *e = entries + i;
			j = e->row_off;
			e->row_off = e->col_off;
			e->col_off = j;
		}
	}
	else {
		/* qsort(entries, num_entries, sizeof(entry_idx_t),
				compare_row_off); */
		radix_sort(entries, num_entries);
	}

	for (i = j = 0; i < num_entries; i++, j++) {

		entry_idx_t *e = entries + i;

		col_entries[i] = e[0].col_off - col_min;

		if (i > 0 && e[0].row_off != e[-1].row_off) {
			row_entries[e[-1].row_off - row_min] = (uint32)j;
			j = 0;
		}
	}
	if (num_entries > 0)
		row_entries[entries[i-1].row_off - row_min] = (uint32)j;

	/* the running total ends at num_entries, which the check above
	   has already bounded to what a uint32 can hold */

	for (i = j = 0; i < num_rows; i++) {
		uint32 t = row_entries[i];
		row_entries[i] = (uint32)j;
		j += t;
	}
	row_entries[num_rows] = (uint32)num_entries;

	b->num_rows = num_rows;
	b->num_cols = col_max - col_min;
	b->num_col_entries = (uint32)num_entries;
	b->col_entries = b->row_entries = 0;
	printf("%" PRIu64 " %u %u\n", num_entries, num_rows, b->blocksize);
}

/*-------------------------------------------------------------------*/
static void upload_block(gpudata_t *d, block_row_t *b) {

	/* copy a block onto the card and drop its host copy. A block
	   with no nonzeros still gets a (tiny) column array, since CUDA
	   refuses zero-byte allocations */

	size_t col_bytes = b->num_col_entries * sizeof(uint32);
	size_t col_alloc = MAX(col_bytes, sizeof(uint32));
	size_t row_bytes = (b->num_rows + 1) * sizeof(uint32);
	uint32 *row_entries = b->host_data + b->row_offset;

	if (d->use_cudamanaged) {
		CUDA_TRY(cuMemAllocManaged(&b->col_entries,
				col_alloc, CU_MEM_ATTACH_GLOBAL))
		if (col_bytes > 0) {
			CUDA_TRY(cuMemcpy(b->col_entries,
				(CUdeviceptr) b->host_data, col_bytes))
		}
		CUDA_TRY(my_cuMemAdvise(b->col_entries, col_alloc,
				CU_MEM_ADVISE_SET_READ_MOSTLY,
				d->gpu_info->device_handle))

		CUDA_TRY(cuMemAllocManaged(&b->row_entries,
				row_bytes, CU_MEM_ATTACH_GLOBAL))
		CUDA_TRY(cuMemcpy(b->row_entries,
				(CUdeviceptr) row_entries, row_bytes))
		CUDA_TRY(my_cuMemAdvise(b->row_entries, row_bytes,
				CU_MEM_ADVISE_SET_READ_MOSTLY,
				d->gpu_info->device_handle))
	} else {
		/* CUDA_TRY(allocateCompressible((void **)&b->col_entries, col_bytes, 1)) */
		CUDA_TRY(cuMemAlloc(&b->col_entries, col_alloc))
		if (col_bytes > 0) {
			CUDA_TRY(cuMemcpyHtoD(b->col_entries,
				b->host_data, col_bytes))
		}

		/* CUDA_TRY(allocateCompressible((void **)&b->row_entries, row_bytes, 1)) */
		CUDA_TRY(cuMemAlloc(&b->row_entries, row_bytes))
		CUDA_TRY(cuMemcpyHtoD(b->row_entries,
				row_entries, row_bytes))
	}

	free(b->host_data);
	b->host_data = NULL;
}

/*-------------------------------------------------------------------*/
static size_t vector_mem_bytes(packed_matrix_t *p) {

	/* the vectors used in the lanczos iteration, and the vv kernel
	   scratch array */

#ifdef HAVE_MPI
	return (6 * (size_t)p->nsubcols +
		2 * (size_t)MAX(p->nrows, p->ncols)) * sizeof(v_t) +
		VBITS * sizeof(v_t);
#else
	return 7 * (size_t)p->max_ncols * sizeof(v_t) +
			VBITS * sizeof(v_t);
#endif
}

/*-------------------------------------------------------------------*/
/* Streaming matrix blocks from the host.

   Every Lanczos iteration multiplies by the whole matrix, so blocks
   that don't fit on the card are copied in again every iteration,
   through a few staging buffers on a separate stream while the SpMV
   works on earlier blocks. The streamed blocks are spread evenly
   through the order the iteration uses the blocks in, to keep the
   copy engine busy the whole time. In single-copy mode the transpose
   product runs through the blocks backwards, so the last streamed
   blocks of the forward product are still in their buffers.

   Pinned host-to-device copies ran at ~48 GB/s on an RTX 5070 (PCIe 5,
   WSL), which hides the copies as long as the streamed bytes per
   iteration take no longer to copy than the iteration takes. On the
   C189 matrices in LANCZOS_OPTIMIZATION_NOTES.md they were the
   limit, at 30-35 GB/s */

#define NUM_STREAM_SLOTS 3

/* room left on the card for the CUDA runtime, whatever the steps after
   the iteration allocate, and, under the Windows driver model (native
   Windows or WSL), for other programs using the card. It only decides
   how much to stream once streaming is unavoidable: a matrix that fits
   without it is loaded whole, as it always was. On WSL a C189 run that
   streamed to within 256MB of full got 3x slower and later hit driver
   faults (LANCZOS_OPTIMIZATION_NOTES.md), so WDDM gets more room */
#define STREAM_MARGIN ((size_t)256 << 20)
#define STREAM_MARGIN_WDDM ((size_t)1536 << 20)

/* cuMemAlloc rounds large allocations up to 2MB, and a resident
   block makes two of them: 4MB in the worst case, which a streaming
   layout is budgeted with, and 2MB on average, which is used to
   decide whether a matrix fits at all */
#define BLOCK_ALLOC_SLACK ((size_t)4 << 20)
#define BLOCK_ALLOC_SLACK_AVG ((size_t)2 << 20)

enum {
	PLACE_RESIDENT,  /* each block goes onto the card as it is built */
	PLACE_STREAM     /* the blocks to stream were chosen from their
			    sizes before building; those are packed
			    straight into (pinned) host memory */
};

/* The layout of the sparse matrix, decided before any block is built:
   every block, forward then transpose (the order one iteration uses
   them), with the columns (forward) or rows (transpose) it covers,
   its nonzeros and its packed size, plus which ones to stream */

typedef struct {
	size_t budget;          /* card memory the sparse blocks may use */
	double frac;            /* stream at least this fraction */
	uint32 max_slots;       /* staging buffers asked for */
	uint32 mode;

	uint32 num_blocks;
	uint32 num_forward;
	uint32 alloc;
	uint32 *start;
	uint32 *size;
	uint64 *nnz;
	size_t *bytes;

	uint8 *streamed;        /* PLACE_STREAM only */
	uint32 num_slots;
	size_t slot_bytes;
} block_plan_t;

/*-------------------------------------------------------------------*/
static void plan_add_block(block_plan_t *plan, uint32 start, uint32 size,
			uint64 nnz, uint32 num_rows) {

	uint32 n = plan->num_blocks;

	/* caught here rather than in pack_matrix_block, before hours of
	   building blocks; block_nnz stops at MAX_BLOCK_NNZ so that one
	   column's overshoot still fits */

	if (nnz > MAX_CSR_BLOCK_NNZ) {
		printf("error: matrix block %u would hold %" PRIu64 " nonzeros, "
			"above the %" PRIu64 " a CSR block can hold; lower "
			"block_nnz\n", n, nnz, MAX_CSR_BLOCK_NNZ);
		exit(-1);
	}

	if (n == plan->alloc) {
		plan->alloc = MAX(100, 2 * plan->alloc);
		plan->start = (uint32 *)xrealloc(plan->start,
					plan->alloc * sizeof(uint32));
		plan->size = (uint32 *)xrealloc(plan->size,
					plan->alloc * sizeof(uint32));
		plan->nnz = (uint64 *)xrealloc(plan->nnz,
					plan->alloc * sizeof(uint64));
		plan->bytes = (size_t *)xrealloc(plan->bytes,
					plan->alloc * sizeof(size_t));
	}
	plan->start[n] = start;
	plan->size[n] = size;
	plan->nnz[n] = nnz;
	plan->bytes[n] = block_bytes(nnz, num_rows);
	plan->num_blocks++;
}

static void plan_free(block_plan_t *plan) {

	free(plan->start);
	free(plan->size);
	free(plan->nnz);
	free(plan->bytes);
	free(plan->streamed);
	memset(plan, 0, sizeof(block_plan_t));
}

/*-------------------------------------------------------------------*/
static uint32 col_sparse_entries(la_col_t *col, uint32 nrows) {

	/* how many of the column's row indices extract_block keeps */

	uint32 j;

	if (col->weight == 0 || col->data[col->weight - 1] < nrows)
		return col->weight;
	for (j = 0; j < col->weight && col->data[j] < nrows; j++)
		;
	return j;
}

/*-------------------------------------------------------------------*/
static void plan_forward_blocks(packed_matrix_t *p, block_plan_t *plan) {

	/* the forward product's blocks: whole columns until a block has
	   block_nnz nonzeros. Every block spans all the rows */

	uint32 i;
	uint32 start = 0;

	while (start < p->ncols) {
		uint64 num = 0;

		for (i = start; i < p->ncols && num < p->block_nnz; i++)
			num += col_sparse_entries(p->unpacked_cols + i,
						p->nrows);
		plan_add_block(plan, start, i - start, num, p->nrows);
		start = i;
	}
	plan->num_forward = plan->num_blocks;
}

/*-------------------------------------------------------------------*/
static void plan_trans_blocks(packed_matrix_t *p, block_plan_t *plan) {

	/* the transpose's blocks: ranges of rows holding about block_nnz
	   nonzeros, across all the columns. The first rows are the
	   heaviest, so the search starts from a small range and grows or
	   shrinks it by 5/4 steps, accepting 90-110% of block_nnz. That
	   only ever needs the nonzeros in a range of rows, which prefix
	   sums of the per-row counts give directly: one pass over the
	   matrix instead of one per search step. The steps are those of
	   the extract_block_trans this replaced, so the blocks are
	   unchanged */

	uint32 i, j;
	uint32 nrows = p->nrows;
	uint32 start_row = 0;
	uint32 blocksize = p->block_nnz / 10000;
	uint64 *below = (uint64 *)xcalloc((size_t)nrows + 1, sizeof(uint64));

	for (i = 0; i < p->ncols; i++) {
		la_col_t *col = p->unpacked_cols + i;
		for (j = 0; j < col->weight && col->data[j] < nrows; j++)
			below[col->data[j] + 1]++;
	}
	for (i = 0; i < nrows; i++)
		below[i + 1] += below[i];

#define ROWS_NNZ(lo, hi) \
	(below[MIN(hi, nrows)] - below[MIN(lo, nrows)])

	while (start_row < nrows) {

		uint32 row_min = start_row;
		uint32 row_max = nrows;
		uint64 nnz = p->block_nnz;
		uint64 min_nnz = 9 * (nnz / 10);
		/* the +10% search window has to stay inside what a
		   block can hold */
		uint64 max_nnz = MIN(11 * (nnz / 10), (uint64)MAX_BLOCK_NNZ);
		uint32 my_blocksize = blocksize;
		uint32 my_row_max = row_min + my_blocksize;
		uint64 num_entries;

		if (my_row_max > row_max) {
			my_row_max = row_max;
			my_blocksize = row_max - row_min;
		}

		while (1) {
			num_entries = ROWS_NNZ(row_min, my_row_max);
			if (num_entries > max_nnz) {
				my_blocksize = 4 * (my_blocksize / 5);
				if ((row_min == 0) &&
				    (my_blocksize <= p->num_dense_rows)) {
					/* just grab a few and continue */
					my_row_max = p->num_dense_rows + 10;
					break;
				}
				my_row_max = row_min + my_blocksize;
				if (my_blocksize == 2) break;
				min_nnz = 0;
				continue;
			}
			if (num_entries < min_nnz) {
				my_blocksize = 5 * (my_blocksize / 4);
				my_row_max = row_min + my_blocksize;
				if (my_row_max >= row_max) {
					my_row_max = row_max;
					break;
				}
				max_nnz = (uint64)MAX_BLOCK_NNZ;
				continue;
			}
			break;
		}

		/* num_entries is the count from the last search step,
		   which after a shrink is not the final range's */

		if (num_entries == 0) /* shouldn't happen */
			my_row_max = MIN(my_row_max + 10, row_max);

		blocksize = my_row_max - row_min;
		plan_add_block(plan, row_min, blocksize,
				ROWS_NNZ(row_min, my_row_max), p->ncols);
		start_row += blocksize;
	}
#undef ROWS_NNZ

	free(below);
}

/*-------------------------------------------------------------------*/
static uint32 spaced_block(uint32 i, uint32 m, uint32 n) {

	/* the i-th of m blocks spread evenly over n */

	return (uint32)(((2 * (uint64)i + 1) * n) / (2 * (uint64)m));
}

static uint32 choose_streamed(block_plan_t *plan) {

	/* Choose the fewest blocks m to stream, spread evenly through
	   the order an iteration uses them, so that the resident blocks
	   plus the staging buffers (as many as asked for, fewer if those
	   don't fit, and never more than m, each the size of the largest
	   streamed block) fit in the budget and at least a fraction frac
	   of the matrix is streamed. Fills in plan->streamed, num_slots
	   and slot_bytes; returns m, or 0 if nothing fits */

	uint32 i, m, m0, slots;
	uint32 n = plan->num_blocks;
	const size_t *bytes = plan->bytes;
	size_t total = 0, cost = 0, largest_all = 0;

	for (i = 0; i < n; i++) {
		total += bytes[i];
		cost += bytes[i] + BLOCK_ALLOC_SLACK;
		largest_all = MAX(largest_all, bytes[i]);
	}

	/* a streamed block frees at most the largest block's cost, and
	   supplies at most the largest block's bytes, so fewer than m0
	   blocks can never be enough */

	m0 = 1;
	if (cost > plan->budget)
		m0 = MAX(m0, (uint32)((cost - plan->budget) /
				(largest_all + BLOCK_ALLOC_SLACK)));
	m0 = MAX(m0, (uint32)(plan->frac * total / largest_all));
	m0 = MIN(m0, n);

	for (m = m0; m <= n; m++) {
		size_t sbytes = 0, scost = 0, largest = 0;

		for (i = 0; i < m; i++) {
			uint32 k = spaced_block(i, m, n);
			sbytes += bytes[k];
			scost += bytes[k] + BLOCK_ALLOC_SLACK;
			largest = MAX(largest, bytes[k]);
		}
		if ((double)sbytes < plan->frac * total)
			continue;

		for (slots = MIN(plan->max_slots, m);
				slots >= MIN(2, m); slots--) {
			if (cost - scost + slots * (largest +
					BLOCK_ALLOC_SLACK / 2) > plan->budget)
				continue;

			memset(plan->streamed, 0, n);
			for (i = 0; i < m; i++)
				plan->streamed[spaced_block(i, m, n)] = 1;
			plan->num_slots = slots;
			plan->slot_bytes = largest;
			return m;
		}
	}
	return 0;
}

/*-------------------------------------------------------------------*/
static uint32 gpu_under_wddm(gpudata_t *d) {

	/* whether the card is driven by the Windows display driver model,
	   natively (unless in TCC mode) or through WSL */

#if defined(WIN32) || defined(_WIN64)
	int tcc = 0;

	CUDA_TRY(cuDeviceGetAttribute(&tcc, CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
				d->gpu_info->device_handle))
	return !tcc;
#else
	(void)d;
	return access("/dev/dxg", F_OK) == 0;
#endif
}

/*-------------------------------------------------------------------*/
static void plan_matrix(msieve_obj *obj, packed_matrix_t *p,
			block_plan_t *plan) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;
	size_t free_mem, total_mem, vectors, dense, avail;
	size_t fit_cost = 0;
	size_t margin = gpu_under_wddm(d) ? STREAM_MARGIN_WDDM :
						STREAM_MARGIN;
	const char *tmp;

	memset(plan, 0, sizeof(block_plan_t));
	plan->mode = PLACE_RESIDENT;
	plan->max_slots = NUM_STREAM_SLOTS;

	/* every block's extent and size, forward and (two-copy)
	   transpose; gpu_matrix_init builds exactly these */

	plan_forward_blocks(p, plan);
	if (!d->single_copy)
		plan_trans_blocks(p, plan);

	if (d->use_cudamanaged) {
		if (obj->nfs_args != NULL &&
		    (strstr(obj->nfs_args, "max_gpu_mem=") ||
		     strstr(obj->nfs_args, "stream_frac=") ||
		     strstr(obj->nfs_args, "stream_slots=")))
			logprintf(obj, "note: max_gpu_mem, stream_frac and "
					"stream_slots are ignored with "
					"use_managed=1\n");
		return;
	}

	/* what the sparse blocks may use: the free memory now, less the
	   vectors and dense rows (allocated later), and when streaming,
	   less a margin. max_gpu_mem=MB is used instead of the free
	   memory, with no margin, as the memory the matrix and vectors
	   may take; stream_frac=F streams at least that fraction of the
	   sparse matrix and stream_slots=N sets the number of staging
	   buffers */

	CUDA_TRY(cuMemGetInfo(&free_mem, &total_mem))
	vectors = vector_mem_bytes(p);
	dense = ((p->num_dense_rows + VBITS - 1) / VBITS) *
			(size_t)p->ncols * sizeof(v_t);
	avail = free_mem > vectors + dense ? free_mem - vectors - dense : 0;

	if (obj->nfs_args != NULL &&
	    (tmp = strstr(obj->nfs_args, "max_gpu_mem=")) != NULL) {
		size_t limit = (size_t)strtoull(tmp + 12, NULL, 10) << 20;
		avail = limit > vectors + dense ? limit - vectors - dense : 0;
		margin = 0;
	}
	plan->budget = avail > margin ? avail - margin : 0;

	if (obj->nfs_args != NULL &&
	    (tmp = strstr(obj->nfs_args, "stream_frac=")) != NULL) {
		plan->frac = atof(tmp + 12);
		if (!(plan->frac > 0))		/* also catches NaN */
			plan->frac = 0;
		if (plan->frac > 1)
			plan->frac = 1;
	}
	if (obj->nfs_args != NULL &&
	    (tmp = strstr(obj->nfs_args, "stream_slots=")) != NULL)
		plan->max_slots = MIN(16, MAX(2, atoi(tmp + 13)));

	for (i = 0; i < plan->num_blocks; i++)
		fit_cost += plan->bytes[i] + BLOCK_ALLOC_SLACK_AVG;

	logprintf(obj, "GPU memory: %.0f MB free, %.0f MB for vectors, "
			"%.0f MB for sparse matrix blocks (%.0f MB needed)\n",
			(double)free_mem / 1048576, (double)vectors / 1048576,
			(double)avail / 1048576, (double)fit_cost / 1048576);

	if (plan->frac == 0 && fit_cost <= avail)
		return;

	plan->streamed = (uint8 *)xmalloc(plan->num_blocks);
	if (choose_streamed(plan) == 0) {
		logprintf(obj, "warning: cannot stream the matrix within "
				"%.0f MB (a smaller block_nnz may help); "
				"trying to load all of it onto the card\n",
				(double)plan->budget / 1048576);
		free(plan->streamed);
		plan->streamed = NULL;
		return;
	}
	if (margin > 0)
		logprintf(obj, "leaving %u MB of GPU memory free while "
				"streaming (use max_gpu_mem=N to change)\n",
				(uint32)(margin >> 20));
	plan->mode = PLACE_STREAM;
}

/*-------------------------------------------------------------------*/
static void alloc_staging(gpudata_t *d, uint32 num_slots, size_t bytes) {

	/* the staging buffers streamed blocks are copied into. They are
	   the busiest memory on a nearly full card, so they are allocated
	   before the resident blocks in case the last allocations are
	   the ones WDDM would put in system memory; on the C189 TD=90
	   matrix the order made no measurable difference */

	uint32 i;

	d->num_slots = num_slots;
	d->slots = (stream_slot_t *)xcalloc(num_slots, sizeof(stream_slot_t));
	for (i = 0; i < num_slots; i++) {
		stream_slot_t *s = d->slots + i;
		CUDA_TRY(cuMemAlloc(&s->buf, bytes))
		CUDA_TRY(cuEventCreate(&s->ready, CU_EVENT_DISABLE_TIMING))
		CUDA_TRY(cuEventCreate(&s->freed, CU_EVENT_DISABLE_TIMING))
	}
	CUDA_TRY(cuStreamCreate(&d->copy_stream, CU_STREAM_NON_BLOCKING))
	d->staging_bytes = num_slots * bytes;
}

/*-------------------------------------------------------------------*/
static void check_block(block_plan_t *plan, uint32 k, uint32 size,
			uint64 num_entries) {

	/* the planned counts come from the same column data, so this
	   only fails on a bug */

	if (size != plan->size[k] || num_entries != plan->nnz[k]) {
		printf("error: matrix block %u has %u lines and %" PRIu64
			" nonzeros, planned %u and %" PRIu64 "\n",
			k, size, num_entries,
			plan->size[k], plan->nnz[k]);
		exit(-1);
	}
}

static void gpu_matrix_init(packed_matrix_t *p, block_plan_t *plan) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;
	uint32 streaming = (plan->mode == PLACE_STREAM);
	uint32 num_trans = plan->num_blocks - plan->num_forward;
	uint64 num_entries_alloc = 10000;
	entry_idx_t *entries = (entry_idx_t *)xmalloc(
					num_entries_alloc *
					sizeof(entry_idx_t));

	if (streaming)
		alloc_staging(d, plan->num_slots, plan->slot_bytes);

	/* deal with the dense rows */

	copy_dense(p);

	/* deal with the sparse rows, block by block as planned. Blocks
	   go onto the card as they are built, except those the plan
	   streams, which are packed into host memory */

	printf("converting matrix to CSR\n");

	d->num_block_rows = plan->num_forward;
	d->block_rows = (block_row_t *)xcalloc(MAX(1, plan->num_forward),
					sizeof(block_row_t));

	for (i = 0; i < plan->num_forward; i++) {

		block_row_t *b = d->block_rows + i;
		uint32 blocksize;
		uint64 num_entries;

		num_entries = extract_block(p->unpacked_cols,
					0, p->nrows,
					plan->start[i],
					plan->start[i] + plan->size[i],
					(uint64)(-1),
					&blocksize,
					&entries,
					&num_entries_alloc);
		check_block(plan, i, blocksize, num_entries);

		b->blocksize = blocksize;
		b->start = plan->start[i];
		pack_matrix_block(b, entries, num_entries,
				0, p->nrows,
				b->start, b->start + blocksize,
				0, streaming && plan->streamed[i]);
		if (!b->streamed)
			upload_block(d, b);
	}

	/* the transpose of the matrix; in single-copy mode the
	   transpose multiply reuses the blocks above instead */

	d->num_trans_block_rows = num_trans;
	d->trans_block_rows = (block_row_t *)xcalloc(MAX(1, num_trans),
					sizeof(block_row_t));

	for (i = 0; i < num_trans; i++) {

		uint32 k = plan->num_forward + i;
		block_row_t *b = d->trans_block_rows + i;
		uint64 num_entries;

		num_entries = extract_rows(p->unpacked_cols, p->ncols,
					plan->start[k],
					plan->start[k] + plan->size[k],
					&entries,
					&num_entries_alloc);
		check_block(plan, k, plan->size[k], num_entries);

		b->blocksize = plan->size[k];
		b->start = plan->start[k];
		pack_matrix_block(b, entries, num_entries,
				0, p->ncols,
				b->start, b->start + b->blocksize,
				1, streaming && plan->streamed[k]);
		if (!b->streamed)
			upload_block(d, b);
	}

	free(entries);
}

/*-------------------------------------------------------------------*/
static void free_block(block_row_t *b) {

	if (b->pinned) {
		CUDA_TRY(cuMemFreeHost(b->host_data))
	}
	else if (b->streamed) {
		free(b->host_data);
	}
	else {
		CUDA_TRY(cuMemFree(b->row_entries))
		CUDA_TRY(cuMemFree(b->col_entries))
	}
}

static void gpu_matrix_free(packed_matrix_t *p) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;

	/* no copies may still be reading pinned host memory */
	CUDA_TRY(cuCtxSynchronize())

	for (i = 0; i < d->num_block_rows; i++)
		free_block(d->block_rows + i);
	free(d->block_rows);

	for (i = 0; i < d->num_trans_block_rows; i++)
		free_block(d->trans_block_rows + i);
	free(d->trans_block_rows);

	for (i = 0; i < d->num_slots; i++) {
		stream_slot_t *s = d->slots + i;
		CUDA_TRY(cuMemFree(s->buf))
		CUDA_TRY(cuEventDestroy(s->ready))
		CUDA_TRY(cuEventDestroy(s->freed))
	}
	free(d->slots);
	free(d->sched);
	if (d->copy_stream != NULL)
		CUDA_TRY(cuStreamDestroy(d->copy_stream))

	for (i = 0; i < (p->num_dense_rows + VBITS - 1) / VBITS; i++)
		CUDA_TRY(cuMemFree(d->dense_blocks[i]))
	free(d->dense_blocks);
}

/*-------------------------------------------------------------------*/
static void setup_schedule(msieve_obj *obj, packed_matrix_t *p,
			block_plan_t *plan) {

	/* the schedule: the streamed blocks in the order one iteration
	   uses them. In single-copy mode each is used twice, forward
	   and then (in reverse block order) by the transpose product */

	uint32 i, m = 0, len = 0;
	gpudata_t *d = (gpudata_t *)p->extra;
	uint32 n = d->num_block_rows + d->num_trans_block_rows;
	block_row_t **order;
	size_t streamed_bytes = 0;

	if (plan->mode != PLACE_STREAM)
		return;

	order = (block_row_t **)xmalloc(n * sizeof(block_row_t *));
	for (i = 0; i < d->num_block_rows; i++)
		order[i] = d->block_rows + i;
	for (i = 0; i < d->num_trans_block_rows; i++)
		order[d->num_block_rows + i] = d->trans_block_rows + i;

	for (i = 0; i < n; i++) {
		if (order[i]->streamed) {
			m++;
			streamed_bytes += order[i]->bytes;
		}
	}

	d->sched = (block_row_t **)xmalloc(2 * m * sizeof(block_row_t *));
	for (i = 0; i < n; i++) {
		if (order[i]->streamed) {
			order[i]->sched_idx[0] = len;
			order[i]->sched_idx[1] = len;
			d->sched[len++] = order[i];
		}
	}
	if (d->single_copy) {
		for (i = n; i-- > 0; ) {
			if (order[i]->streamed) {
				order[i]->sched_idx[1] = len;
				d->sched[len++] = order[i];
			}
		}
	}
	d->sched_len = len;

	/* with no more streamed blocks than buffers, each block gets
	   copied in once and stays; nothing is copied per iteration */

	d->streamed_bytes = (m > d->num_slots) ? streamed_bytes : 0;
	if (d->streamed_bytes > 0)
		logprintf(obj, "streaming %u of %u matrix blocks (%.0f MB) "
				"from pinned host memory through %u %.0f MB "
				"buffers\n", m, n,
				(double)streamed_bytes / 1048576,
				d->num_slots, (double)plan->slot_bytes / 1048576);
	else
		logprintf(obj, "%u of %u matrix blocks (%.0f MB) are held in "
				"their own staging buffers; nothing is copied "
				"per iteration\n", m, n,
				(double)streamed_bytes / 1048576);

	free(order);
}

/*-------------------------------------------------------------------*/
static uint32 sched_distance(gpudata_t *d, block_row_t *b, uint32 pos) {

	/* how many schedule entries from pos until b is used again */

	uint32 d0 = (b->sched_idx[0] + d->sched_len - pos) % d->sched_len;
	uint32 d1 = (b->sched_idx[1] + d->sched_len - pos) % d->sched_len;
	return MIN(d0, d1);
}

/*-------------------------------------------------------------------*/
static stream_slot_t * stream_load(gpudata_t *d, block_row_t *b,
				uint32 pos) {

	/* make sure block b is in a staging buffer or being copied into
	   one. Otherwise evict the block that the schedule, from position
	   pos, uses furthest in the future */

	uint32 i;
	uint32 victim = 0;
	uint32 victim_dist = 0;
	stream_slot_t *s;

	for (i = 0; i < d->num_slots; i++) {
		uint32 dist;

		s = d->slots + i;
		if (s->blk == b)
			return s;

		dist = (s->blk == NULL) ? d->sched_len :
				sched_distance(d, s->blk, pos);
		if (i == 0 || dist > victim_dist) {
			victim = i;
			victim_dist = dist;
		}
	}

	s = d->slots + victim;
	CUDA_TRY(cuStreamWaitEvent(d->copy_stream, s->freed, 0))
	CUDA_TRY(cuMemcpyHtoDAsync(s->buf, b->host_data, b->bytes,
				d->copy_stream))
	CUDA_TRY(cuEventRecord(s->ready, d->copy_stream))
	s->blk = b;
	return s;
}

/*------------------------------------------------------------------------*/
#ifdef MSIEVE_CUDA_SINGLE_BINARY
static void
load_spmv_engine(msieve_obj *obj, gpudata_t *d)
{
	char libname[256];
	char *tmp = NULL;

	if (d->gpu_info->compute_version_major < 2) {
		printf("error: GPU compute capability >= 2.0 required\n");
		exit(-1);
	}

	/* The normal build links the CUB SpMV engine directly into msieve. */
	d->spmv_engine_handle = NULL;
	d->spmv_engine_init = spmv_engine_init;
	d->spmv_engine_free = spmv_engine_free;
	d->spmv_engine_run = spmv_engine_run;
	d->spmv_engine_run_trans = spmv_engine_run_trans;
	d->spmv_engine_set_kernel = spmv_engine_set_kernel;

	/* Preserve the historical spmvlib= override for testing/custom engines. */
	if (obj->nfs_args != NULL)
		tmp = strstr(obj->nfs_args, "spmvlib=");
	if (tmp == NULL)
		return;

	{
		uint32 i;
		for (i = 0, tmp += 8; i < sizeof(libname) - 1; i++) {
			if (*tmp == 0 || isspace(*tmp))
				break;
			libname[i] = *tmp++;
		}
		libname[i] = 0;
	}

	d->spmv_engine_handle = load_dynamic_lib(libname);
	if (d->spmv_engine_handle == NULL) {
		printf("error: failed to load GPU matrix multiply engine override "
		       "from \"%s\"\n", libname);
		exit(-1);
	}

	d->spmv_engine_init = get_lib_symbol(d->spmv_engine_handle,
					"spmv_engine_init");
	d->spmv_engine_free = get_lib_symbol(d->spmv_engine_handle,
					"spmv_engine_free");
	d->spmv_engine_run = get_lib_symbol(d->spmv_engine_handle,
					"spmv_engine_run");
	/* the override library replaces the built-in engine completely, so
	   the optional entry points come from it too (NULL if it has none) */
	d->spmv_engine_run_trans = get_lib_symbol(d->spmv_engine_handle,
					"spmv_engine_run_trans");
	d->spmv_engine_set_kernel = get_lib_symbol(d->spmv_engine_handle,
					"spmv_engine_set_kernel");
	if (d->spmv_engine_init == NULL ||
	    d->spmv_engine_free == NULL ||
	    d->spmv_engine_run == NULL) {
		printf("error: cannot find GPU matrix multiply function in \"%s\"\n",
			libname);
		exit(-1);
	}
}
#else
static void
load_spmv_engine(msieve_obj *obj, gpudata_t *d)
{
	char libname[256];
	#if defined(WIN32) || defined(_WIN64)
	const char *suffix = ".dll";
	#else
	const char *suffix = ".so";
	#endif

	if (d->gpu_info->compute_version_major < 2) {
		printf("error: GPU compute capability >= 2.0 required\n");
		exit(-1);
	}

	sprintf(libname, "cub/spmv_engine%s", suffix);

	/* override from input args */

	if (obj->nfs_args != NULL) {
		char *tmp = strstr(obj->nfs_args, "spmvlib=");

		if (tmp != NULL) {
			uint32 i;
			for (i = 0, tmp += 8; i < sizeof(libname) - 1; i++) {
				if (*tmp == 0 || isspace(*tmp))
					break;

				libname[i] = *tmp++;
			}
			libname[i] = 0;
		}
	}

	d->spmv_engine_handle = load_dynamic_lib(libname);
	if (d->spmv_engine_handle == NULL) {
		printf("error: failed to load GPU matrix multiply engine\n");
		exit(-1);
	}

	/* the spmv engine uses the same CUDA context */

	d->spmv_engine_init = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_init");
	d->spmv_engine_free = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_free");
	d->spmv_engine_run = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_run");
	d->spmv_engine_run_trans = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_run_trans");
	d->spmv_engine_set_kernel = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_set_kernel");
	if (d->spmv_engine_init == NULL ||
	    d->spmv_engine_free == NULL ||
	    d->spmv_engine_run == NULL) {
		printf("error: cannot find GPU matrix multiply function\n");
		exit(-1);
	}
}
#endif


/*-------------------------------------------------------------------*/
void matrix_extra_init(msieve_obj *obj, packed_matrix_t *p,
			uint32 first_block_size) {

	uint32 i;
	int check_vbits;
	gpudata_t *d;
	gpu_config_t gpu_config;
	gpu_info_t *gpu_info;
	CUresult status;
	block_plan_t plan;
	uint32 floor_binds = 0;

	/* select card, save info struct */

	gpu_init(&gpu_config);
	if (gpu_config.num_gpu == 0) {
		printf("error: no CUDA-enabled GPUs found\n");
		exit(-1);
	}
	if (obj->which_gpu >= (uint32)gpu_config.num_gpu) {
		printf("error: GPU %u does not exist "
			"or is not CUDA-enabled\n", obj->which_gpu);
		exit(-1);
	}

	p->extra = d = (gpudata_t *)xcalloc(1, sizeof(gpudata_t));

	d->gpu_info = gpu_info = (gpu_info_t *)xmalloc(sizeof(gpu_info_t));
	memcpy(gpu_info, gpu_config.info + obj->which_gpu,
			sizeof(gpu_info_t)); 

	logprintf(obj, "using GPU %u (%s)\n", obj->which_gpu, gpu_info->name);
	logprintf(obj, "selected card has CUDA arch %d.%d\n",
			gpu_info->compute_version_major,
			gpu_info->compute_version_minor);

 	/* CUDA_TRY(cuDevicePrimaryCtxSetFlags(d->gpu_info->device_handle,
			CU_CTX_SCHED_BLOCKING_SYNC)) */

	/* initialize context */

	CUDA_TRY(my_cuCtxCreate(&d->gpu_context,
			CU_CTX_SCHED_BLOCKING_SYNC,
			d->gpu_info->device_handle))

	/* CUDA_TRY(cuDevicePrimaryCtxRetain(&d->gpu_context,
			d->gpu_info->device_handle)) */

	load_spmv_engine(obj, d);
	d->spmv_engine = d->spmv_engine_init(&check_vbits);
	if (check_vbits != VBITS) {
                printf("error: SpMV library compiled for VBITS=%d\n", check_vbits);
                exit(-1);
	}

#ifdef MSIEVE_CUDA_SINGLE_BINARY
	/* Prefer native code from the embedded multi-architecture fatbin.
	   Keep a separately embedded PTX image as a fallback for driver/toolkit
	   combinations that reject the fatbin container. */
	status = cuda_load_embedded_module(&d->gpu_module,
			msieve_lanczos_kernel_fatbin,
			(const char *)msieve_lanczos_kernel_ptx,
			"lanczos_kernel");
	CUDA_TRY(status)
#else
	/* load kernels */

	status = cuModuleLoad(&d->gpu_module, "lanczos_kernel.ptx");				\
	if (status != CUDA_SUCCESS) {
		printf("Error loading ptx. Trying fatbin.\n");
		CUDA_TRY(cuModuleLoad(&d->gpu_module, "lanczos_kernel.fatbin"))
	}
#endif

	d->launch = (gpu_launch_t *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(gpu_launch_t));

	for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
		gpu_launch_t *launch = d->launch + i;

		gpu_launch_init(d->gpu_module, gpu_kernel_names[i],
				launch);

		launch->threads_per_block = MIN(256, 
				launch->threads_per_block);
	}

	/* the outer product kernel gives every half-warp its own tables
	   and needs 64 threads to write out each 64-row group */

	if (d->launch[GPU_K_OUTER_PROD].threads_per_block % 16 != 0 ||
	    d->launch[GPU_K_OUTER_PROD].threads_per_block < 64) {
		printf("error: outer product kernel cannot run with "
			"%d threads per block\n",
			d->launch[GPU_K_OUTER_PROD].threads_per_block);
		exit(-1);
	}

	/* allocate scratch arrays */

	CUDA_TRY(cuMemAlloc(&d->gpu_scratch, VBITS * sizeof(v_t)))

	/* should we store only one copy of the matrix on the card */

	d->single_copy = 0;
	if (obj->nfs_args != NULL &&
	    strstr(obj->nfs_args, "single_copy=1") != NULL) {
		if (d->spmv_engine_run_trans == NULL) {
			printf("error: SpMV library does not support "
				"single_copy=1\n");
			exit(-1);
		}
		d->single_copy = 1;
		logprintf(obj, "storing a single copy of the matrix "
				"(no transpose) on the GPU\n");
	}

	/* Set preferred nonzeros per matrix block. The default sizes the
	   active input-vector window of each column-slice SpMV block to
	   about half the L2 cache, with a floor that keeps per-block
	   overhead amortized. Every block spans all the rows (all the
	   columns for the transpose): it carries a full row pointer
	   array and rewrites the whole output vector, so the floor
	   grows with the rows. A block must gather at least 12 times as
	   many 32-byte sectors as it rewrites (2*sizeof(v_t) bytes per
	   output row), and never fewer than 128M nonzeros. On a tall MPI
	   piece (A100, 61M rows) the old fixed 128M floor was 13.5%
	   slower than 1.75B; on the square matrices we tested the row
	   floor leaves every measured optimum alone. Within ~5% of the
	   measured optimum on RTX 5070 (48MB L2) and Tesla V100 (6MB L2)
	   at VBITS 64/128/256; on an RTX 3060 (3MB L2) at VBITS=256 a
	   single block was ~7% faster. See LANCZOS_OPTIMIZATION_NOTES.md.
	   Override with block_nnz=N

	   In single-copy mode the same window is also the output range
	   of the transpose scatter, whose atomics are only cheap while
	   it stays in L2, so the window is a third of L2 (the forward
	   gathers and the scatter compete for it) and the floor is much
	   lower. Measured optimum on RTX 5070 at VBITS=256 was 32M-48M
	   (TD=90 C170 matrix, 72.8 nonzeros/col). Every block carries a
	   full row pointer array, though, so on cards with a small L2
	   the floor keeps those arrays to at most half the size of the
	   column indices; otherwise they eat the memory the missing
	   transpose saves. When that floor binds, single copy was
	   slower than two copies everywhere we measured (RTX 3060 -19%,
	   A100 MPI piece -52%): it then only saves memory */

	p->block_nnz = 1750000000;
	if (p->unpacked_cols != NULL && p->ncols > 0) {
		int l2_bytes = 0;
		uint64 total_weight = 0;
		uint64 computed;
		double avg_col_weight;

		CUDA_TRY(cuDeviceGetAttribute(&l2_bytes,
				CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
				d->gpu_info->device_handle))

		for (i = 0; i < p->ncols; i++)
			total_weight += p->unpacked_cols[i].weight;
		avg_col_weight = (double)total_weight / p->ncols;

		computed = (uint64)((double)(l2_bytes /
					(d->single_copy ? 3 : 2)) /
					sizeof(v_t) * avg_col_weight);
		if (d->single_copy) {
			uint64 row_floor = MAX(16000000, 2 * (uint64)p->nrows);

			if (computed < row_floor)
				floor_binds = 1;
			computed = MAX(computed, row_floor);
		}
		else {
			uint64 row_floor = 12 * (uint64)MAX(p->nrows, p->ncols) *
					sizeof(v_t) / 32;

			computed = MAX(computed, MAX(128000000, row_floor));
		}
		computed = MIN(computed, (uint64)MAX_BLOCK_NNZ);
		p->block_nnz = (uint32)computed;
		logprintf(obj, "computed block_nnz %u (L2 cache %d bytes, "
				"average column weight %.1f)\n",
				p->block_nnz, l2_bytes, avg_col_weight);
	}
	if (obj->nfs_args != NULL) {
		const char *tmp;
		tmp = strstr(obj->nfs_args, "block_nnz=");
		if (tmp != NULL) {
			{
				/* atoi() would wrap anything past INT_MAX into a
				   block size nobody asked for, on a run that then
				   takes hours. strtoull() rather than strtoul()
				   because unsigned long is 32 bits on Windows,
				   where an oversized value saturates at UINT32_MAX
				   instead of being caught as out of range */

				char *endp;
				uint64 v;

				errno = 0;
				v = strtoull(tmp + 10, &endp, 10);

				if (endp == tmp + 10 || errno == ERANGE ||
				    v == 0 || v > UINT32_MAX) {
					logprintf(obj, "error: block_nnz out of range\n");
					exit(-1);
				}
				p->block_nnz = (uint32)v;
			}
			if (p->block_nnz < 100000) p->block_nnz = 100000;
			if (p->block_nnz > MAX_BLOCK_NNZ)
				p->block_nnz = MAX_BLOCK_NNZ;
			floor_binds = 0;
		}
	}
	if (floor_binds)
		logprintf(obj, "note: single_copy's L2-sized blocks are "
				"smaller than its row pointer floor; expect it "
				"to be slower than two copies, only saving GPU "
				"memory\n");
	logprintf(obj, "nonzeros per matrix block: %u\n", p->block_nnz);

	/* choose the kernel for the gather products (everything but
	   the single-copy transpose scatter). By default the engine
	   picks per block from its nonzeros per row: segscan for the
	   short row segments of small blocks, warpmerge for long ones.
	   Override with spmv_kernel=segscan|warpmerge */

	if (d->spmv_engine_set_kernel != NULL) {
		int kernel = SPMV_KERNEL_AUTO;
		const char *name = "auto (by nonzeros per row)";

		if (obj->nfs_args != NULL) {
			if (strstr(obj->nfs_args, "spmv_kernel=segscan")) {
				kernel = SPMV_KERNEL_SEGSCAN;
				name = "segscan";
			}
			if (strstr(obj->nfs_args, "spmv_kernel=warpmerge")) {
				kernel = SPMV_KERNEL_WARPMERGE;
				name = "warpmerge";
			}
		}
		d->spmv_engine_set_kernel(d->spmv_engine, kernel);
		logprintf(obj, "SpMV kernel: %s\n", name);
	}

	/* should we used CUDA managed memory to store the matrix */

	d->use_cudamanaged = 0;
	if (obj->nfs_args != NULL) {
		const char *tmp;
		tmp = strstr(obj->nfs_args, "use_managed=1");
		if (tmp != NULL) {
			if (d->gpu_info->concurrent_managed_access)
				d->use_cudamanaged = 2; /* can prefetch */
			else d->use_cudamanaged = 1;
			printf("Storing matrix in managed memory\n");
		}
	}
	
	/* Adjust L2 fetch granularity. Default is 128. Tried 32 for VBITS=256, but makes no difference */
	/* if (gpu_info->compute_version_major >= 8) CUDA_TRY(cuCtxSetLimit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY, 32)) */

	/* set up the matrix on the card, streaming what doesn't fit */

	plan_matrix(obj, p, &plan);
	gpu_matrix_init(p, &plan);
	setup_schedule(obj, p, &plan);
	plan_free(&plan);
}

/*-------------------------------------------------------------------*/
void matrix_extra_free(packed_matrix_t *p) {

	gpudata_t *d = (gpudata_t *)p->extra;

	gpu_matrix_free(p);

	CUDA_TRY(cuMemFree(d->gpu_scratch))

	free(d->launch);

	d->spmv_engine_free(d->spmv_engine);
	if (d->spmv_engine_handle != NULL)
		unload_dynamic_lib(d->spmv_engine_handle);

	CUDA_TRY(cuCtxDestroy(d->gpu_context))
	/* CUDA_TRY(cuDevicePrimaryCtxRelease(d->gpu_info->device_handle)) */

	free(d->gpu_info);
	free(d);
}

/*-------------------------------------------------------------------*/
static void run_spmv_block(gpudata_t *d, block_row_t *blk,
			CUdeviceptr vector_in, CUdeviceptr vector_out,
			spmv_engine_run_func run, uint32 pass,
			const char *label) {

	spmv_data_t spmv_data;
	stream_slot_t *slot = NULL;
	uint32 pos = 0;

	spmv_data.col_entries = blk->col_entries;
	spmv_data.row_entries = blk->row_entries;

	if (blk->streamed) {

		/* a streamed block: wait for its copy, which was usually
		   started while earlier blocks ran. pass is 1 for the
		   single-copy transpose product */

		pos = blk->sched_idx[pass];
		slot = stream_load(d, blk, pos);
		CUDA_TRY(cuStreamWaitEvent(NULL, slot->ready, 0))
		spmv_data.col_entries = slot->buf;
		spmv_data.row_entries = slot->buf +
				blk->row_offset * sizeof(uint32);
	}
	else if (d->use_cudamanaged == 2) {
		CUDA_TRY(my_cuMemPrefetchAsync(blk->col_entries,
			blk->num_col_entries * sizeof(uint32),
			d->gpu_info->device_handle, 0))
		CUDA_TRY(my_cuMemPrefetchAsync(blk->row_entries,
			(blk->num_rows + 1) * sizeof(uint32),
			d->gpu_info->device_handle, 0))
	}
	spmv_data.num_rows = blk->num_rows;
	spmv_data.num_col_entries = blk->num_col_entries;
	spmv_data.vector_in = vector_in;
	spmv_data.vector_out = vector_out;

	LANCZOS_NVTX_PUSH(label, LANCZOS_NVTX_COLOR_SPMV_RUN);
	run(d->spmv_engine, &spmv_data);
	LANCZOS_NVTX_POP();
	(void)label;

	if (slot != NULL) {
		uint32 i;

		/* the buffer can be reused once this SpMV is done; start
		   copying the next streamed blocks the schedule needs */

		CUDA_TRY(cuEventRecord(slot->freed, NULL))
		for (i = 1; i < d->num_slots; i++)
			stream_load(d, d->sched[(pos + i) % d->sched_len],
					pos);
	}
}

/*-------------------------------------------------------------------*/
static void mul_packed_gpu(packed_matrix_t *p, 
				gpuvec_t *x, gpuvec_t *b) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;

	LANCZOS_NVTX_PUSH("mul_packed.memset", LANCZOS_NVTX_COLOR_MUL);
	CUDA_TRY(cuMemsetD8(b->gpu_vec, 0,
			p->nrows * sizeof(v_t)));
	LANCZOS_NVTX_POP();

	/* sweep through the matrix a block col at a time */

	LANCZOS_NVTX_PUSH("mul_packed.spmv_blocks", LANCZOS_NVTX_COLOR_MUL);
	for (i = 0; i < d->num_block_rows; i++) {

		block_row_t *blk = d->block_rows + i;

		run_spmv_block(d, blk,
			(CUdeviceptr)((v_t *)x->gpu_vec + blk->start),
			b->gpu_vec, d->spmv_engine_run, 0,
			"spmv_engine_run.normal");
	}
	LANCZOS_NVTX_POP();

	/* handle dense rows */

	LANCZOS_NVTX_PUSH("mul_packed.dense_rows", LANCZOS_NVTX_COLOR_DENSE);
	for (i = 0; i < (p->num_dense_rows + VBITS - 1) / VBITS; i++) {
		if (d->use_cudamanaged == 2) {
			CUDA_TRY(my_cuMemPrefetchAsync(d->dense_blocks[i],
				p->ncols * sizeof(v_t),
				d->gpu_info->device_handle, 0))
		}
		mul_BxN_NxB_gpu(p,
			d->dense_blocks[i],
			x->gpu_vec,
			(CUdeviceptr)((v_t *)b->gpu_vec + VBITS * i),
			p->ncols);
	}
	LANCZOS_NVTX_POP();
}

/*-------------------------------------------------------------------*/
static void mul_packed_trans_gpu(packed_matrix_t *p, 
				gpuvec_t *x, gpuvec_t *b) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;

	LANCZOS_NVTX_PUSH("mul_packed_trans.memset", LANCZOS_NVTX_COLOR_MUL_TRANS);
	CUDA_TRY(cuMemsetD8(b->gpu_vec, 0,
			p->ncols * sizeof(v_t)));
	LANCZOS_NVTX_POP();

	LANCZOS_NVTX_PUSH("mul_packed_trans.spmv_blocks", LANCZOS_NVTX_COLOR_MUL_TRANS);
	if (d->single_copy) {

		/* no transpose on the card: sweep through the matrix
		   a block col at a time, scattering into that block's
		   columns of b. Going backwards starts with the blocks
		   the forward product used last, which matters when
		   they are streamed from the host */

		for (i = d->num_block_rows; i-- > 0; ) {

			block_row_t *blk = d->block_rows + i;

			run_spmv_block(d, blk, x->gpu_vec,
				(CUdeviceptr)((v_t *)b->gpu_vec + blk->start),
				d->spmv_engine_run_trans, 1,
				"spmv_engine_run.trans_scatter");
		}
	}
	else {
		/* sweep through the transpose a block row at a time */

		for (i = 0; i < d->num_trans_block_rows; i++) {

			block_row_t *blk = d->trans_block_rows + i;

			run_spmv_block(d, blk,
				(CUdeviceptr)((v_t *)x->gpu_vec + blk->start),
				b->gpu_vec, d->spmv_engine_run, 0,
				"spmv_engine_run.trans");
		}
	}
	LANCZOS_NVTX_POP();

	/* handle dense rows; every dense block contributes to all
	   of b, as in the CPU code */

	LANCZOS_NVTX_PUSH("mul_packed_trans.dense_rows", LANCZOS_NVTX_COLOR_DENSE);
	for (i = 0; i < (p->num_dense_rows + VBITS - 1) / VBITS; i++) {
		if (d->use_cudamanaged == 2) {
			CUDA_TRY(my_cuMemPrefetchAsync(d->dense_blocks[i],
				p->ncols * sizeof(v_t),
				d->gpu_info->device_handle, 0))
		}
		mul_NxB_BxB_acc_gpu(p,
			d->dense_blocks[i],
			(CUdeviceptr)((v_t *)x->gpu_vec + VBITS * i),
			b->gpu_vec,
			p->ncols);
	}
	LANCZOS_NVTX_POP();
}

/*-------------------------------------------------------------------*/
void mul_core(packed_matrix_t *A, void *x_in, void *b_in) {

	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	LANCZOS_NVTX_PUSH("mul_core", LANCZOS_NVTX_COLOR_MUL);
	mul_packed_gpu(A, x, b);
	LANCZOS_NVTX_POP();

#ifdef LANCZOS_GPU_DEBUG
	{
		uint32 i, j;
		v_t *tmp = (v_t *) xmalloc(A->ncols * 
						sizeof(v_t));

		CUDA_TRY(cuMemcpyDtoH(tmp, b->gpu_vec, 
					A->nrows * sizeof(v_t)))
		CUDA_TRY(cuMemcpyDtoH(x->host_vec, x->gpu_vec, 
					A->ncols * sizeof(v_t)))

		mul_unpacked(A, x->host_vec, b->host_vec);

		for (i = 0; i < MIN(A->ncols, A->nrows); i++) {
			for (j = 0; j < VWORDS; j++) {				
				if (tmp[i].w[j] != b->host_vec[i].w[j]) { 
					printf("m error %u %" PRIx64 " %" PRIx64 "\n", 
							i, b->host_vec[i].w[j], tmp[i].w[j]);
					exit(-1);
				}
			}
		}

		free(tmp);
	}
#endif
}

/*-------------------------------------------------------------------*/
void mul_trans_core(packed_matrix_t *A, void *x_in, void *b_in) {

	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	LANCZOS_NVTX_PUSH("mul_trans_core", LANCZOS_NVTX_COLOR_MUL_TRANS);
	mul_packed_trans_gpu(A, x, b);
	LANCZOS_NVTX_POP();

#ifdef LANCZOS_GPU_DEBUG
	{
		uint32 i, j;
		v_t *tmp = (v_t *)xmalloc(A->ncols * 
						sizeof(v_t));

		CUDA_TRY(cuMemcpyDtoH(tmp, b->gpu_vec, 
					A->ncols * sizeof(v_t)))
		CUDA_TRY(cuMemcpyDtoH(x->host_vec, x->gpu_vec, 
					A->nrows * sizeof(v_t)))

		mul_trans_unpacked(A, x->host_vec, b->host_vec);

		for (i = 0; i < A->ncols; i++) {
			for (j = 0; j < VWORDS; j++) {				
				if (tmp[i].w[j] != b->host_vec[i].w[j]) { 
					printf("tr error %u %" PRIx64 " %" PRIx64 "\n", 
							i, b->host_vec[i].w[j], tmp[i].w[j]);
					exit(-1);
				}
			}
		}

		free(tmp);
	}
#endif
}

/*-------------------------------------------------------------------*/
size_t packed_matrix_sizeof(packed_matrix_t *p) {

	uint32 i;
	size_t mem_use, tot_mem_use;
	gpudata_t *d = (gpudata_t*) p->extra;

	/* account for the vectors used in the lanczos iteration,
	   and for the vv kernel scratch array */

	mem_use = vector_mem_bytes(p);

	tot_mem_use = mem_use;
	printf("vector memory use: %.1f MB\n", (double)mem_use/1048576);

	/* and for the matrix */

	/* dense rows */
	mem_use = ((p->num_dense_rows + VBITS - 1) / VBITS) * p->ncols * sizeof(v_t);

	tot_mem_use += mem_use;
	printf("dense rows memory use: %.1f MB\n", (double)mem_use/1048576);

	mem_use = 0;

	/* matrix in CSR format, and its transpose, where on the card;
	   streamed blocks only take up the staging buffers */
	for (i = 0; i < d->num_block_rows; i++) {
		block_row_t *b = d->block_rows + i;
		if (!b->streamed)
			mem_use += (b->num_rows + 1 + b->num_col_entries) * sizeof(uint32);
	}

	for (i = 0; i < d->num_trans_block_rows; i++) {
		block_row_t *b = d->trans_block_rows + i;
		if (!b->streamed)
			mem_use += (b->num_rows + 1 + b->num_col_entries) * sizeof(uint32);
	}
	mem_use += d->staging_bytes;

	tot_mem_use += mem_use;
	printf("sparse matrix memory use: %.1f MB\n", (double)mem_use/1048576);
	if (d->num_slots > 0 && d->streamed_bytes == 0)
		printf("streamed from host memory: 0.0 MB (blocks held in "
			"their staging buffers)\n");
	else
		printf("streamed from host memory: %.1f MB\n",
				(double)d->streamed_bytes/1048576);

	return tot_mem_use;
}
