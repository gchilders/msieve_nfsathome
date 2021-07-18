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
#include "lanczos_gpu_core.h"

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

		CUDA_TRY(cuMemAlloc(&d->dense_blocks[i], 
					ncols * sizeof(v_t)))
		CUDA_TRY(cuMemcpyHtoD(d->dense_blocks[i], tmp,
					ncols * sizeof(v_t)))
	}

	free(tmp);
}

/*-------------------------------------------------------------------*/
static uint32 extract_block(la_col_t *cols,
			uint32 row_min, uint32 row_max,
			uint32 col_min, uint32 col_max,
			entry_idx_t **entries_in, 
			uint32 *max_entries_in)
{
	uint32 i, j;
	uint32 num_entries = 0;
	entry_idx_t *entries = *entries_in;
	uint32 max_entries = *max_entries_in;

	for (i = col_min; i < col_max; i++) {

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
static void pack_matrix_block(gpudata_t *d, block_row_t *b,
			entry_idx_t *entries, uint32 num_entries,
			uint32 row_min, uint32 row_max, 
			uint32 col_min, uint32 col_max,
			uint32 is_trans)
{

	uint32 i, j;
	uint32 num_rows = row_max - row_min;
	spmv_data_t spmv_data;

	/* convert a block of matrix rows from COO to CSR format */

	uint32 *col_entries = (uint32 *)xmalloc(num_entries * 
					sizeof(uint32));
	uint32 *row_entries = (uint32 *)xcalloc(num_rows + 1,
					sizeof(uint32));

	if (is_trans) {
		for (i = 0; i < num_entries; i++) {
			entry_idx_t *e = entries + i;
			j = e->row_off;
			e->row_off = e->col_off;
			e->col_off = j;
		}
	}
	else {
		qsort(entries, num_entries, sizeof(entry_idx_t), 
				compare_row_off);
	}

	for (i = j = 0; i < num_entries; i++, j++) {

		entry_idx_t *e = entries + i;

		col_entries[i] = e[0].col_off - col_min;

		if (i > 0 && e[0].row_off != e[-1].row_off) {
			row_entries[e[-1].row_off - row_min] = j;
			j = 0;
		}
	}
	row_entries[entries[i-1].row_off - row_min] = j;

	for (i = j = 0; i < num_rows; i++) {
		uint32 t = row_entries[i];
		row_entries[i] = j;
		j += t;
	}

	b->num_rows = num_rows;
	b->num_col_entries = num_entries;
	printf("%u %u\n", num_entries, num_rows);

	CUDA_TRY(cuMemAlloc(&b->col_entries,
				num_entries * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(b->col_entries,
				col_entries,
				num_entries * sizeof(uint32)))

	CUDA_TRY(cuMemAlloc(&b->row_entries,
				num_rows * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(b->row_entries,
				row_entries,
				num_rows * sizeof(uint32)))

	free(col_entries);
	free(row_entries);

	/* configure the engine for this block of rows */

	spmv_data.num_rows = b->num_rows;
	spmv_data.num_col_entries = b->num_col_entries;
	spmv_data.col_entries = b->col_entries;
	spmv_data.row_entries = b->row_entries;
	spmv_data.vector_in = (CUdeviceptr)0;
	spmv_data.vector_out = (CUdeviceptr)0;
	b->spmv_preprocess_handle = d->spmv_engine_preprocess(&spmv_data);
}

/*-------------------------------------------------------------------*/
static void gpu_matrix_init(packed_matrix_t *p) {

	uint32 start_row = 0;
	uint32 start_col = 0;
	gpudata_t *d = (gpudata_t *)p->extra;

	uint32 num_block_rows = 0;
	uint32 num_trans_block_rows = 0;
	uint32 num_block_rows_alloc = 100;
	uint32 num_trans_block_rows_alloc = 100;
	block_row_t *block_rows = (block_row_t *)xmalloc(
					num_block_rows_alloc *
					sizeof(block_row_t));
	block_row_t *trans_block_rows = (block_row_t *)xmalloc(
					num_trans_block_rows_alloc *
					sizeof(block_row_t));

	uint32 num_entries_alloc = 10000;
	entry_idx_t *entries = (entry_idx_t *)xmalloc(
					num_entries_alloc *
					sizeof(entry_idx_t));

	/* deal with the dense rows */

	copy_dense(p);

	/* deal with the sparse rows */

	printf("converting matrix to CSR and copying it onto the GPU\n");

#ifdef HAVE_MPI
	p->preferred_block = MAX(p->max_nrows, p->max_ncols) / MIN(p->mpi_nrows, p->mpi_ncols) / 4 + 1;
#else
	p->preferred_block = MAX(p->nrows, p->ncols) / 4 + 1;
#endif
	while (start_col < p->ncols) {

		block_row_t *b;
		uint32 block_size = MIN(p->preferred_block, 
					p->ncols - start_col);
		uint32 num_entries;

		num_entries = extract_block(p->unpacked_cols,
					0, p->nrows,
					start_col,
					start_col + block_size,
					&entries,
					&num_entries_alloc);

		if (num_entries > 2147483647) {
			printf("max column entries is 2147483647\n");
			printf("adjust preferred block to compensate\n");
			exit(42);
		}

		if (num_block_rows == num_block_rows_alloc) {
			num_block_rows_alloc *= 2;
			block_rows = (block_row_t *)xrealloc(
					block_rows,
					num_block_rows_alloc *
					sizeof(block_row_t));
		}

		b = block_rows + num_block_rows++;

		pack_matrix_block(d, b, entries, num_entries,
				0, p->nrows, 
				start_col,
				start_col + block_size,
				0);

		start_col += block_size;
	}

	d->num_block_rows = num_block_rows;
	d->block_rows = block_rows;

	/* handle the transpose of the matrix */

	while (start_row < p->nrows) {

		block_row_t *b;
		uint32 block_size = MIN(p->preferred_block, 
					p->nrows - start_row);
		uint32 num_entries;

		num_entries = extract_block(p->unpacked_cols,
					start_row,
					start_row + block_size,
					0, p->ncols,
					&entries,
					&num_entries_alloc);

		if (num_entries > 2147483647) {
			printf("max column entries is 2147483647\n");
			printf("adjust preferred block to compensate\n");
			exit(42);
		}

		if (num_trans_block_rows == num_trans_block_rows_alloc) {
			num_trans_block_rows_alloc *= 2;
			trans_block_rows = (block_row_t *)xrealloc(
					trans_block_rows,
					num_trans_block_rows_alloc *
					sizeof(block_row_t));
		}

		b = trans_block_rows + num_trans_block_rows++;

		pack_matrix_block(d, b, entries, num_entries,
				0, p->ncols, 
				start_row,
				start_row + block_size,
				1);

		start_row += block_size;
	}

	d->num_trans_block_rows = num_trans_block_rows;
	d->trans_block_rows = trans_block_rows;

	CUDA_TRY(cuMemAlloc(&d->matmul_scratch,
				MAX(p->ncols, p->nrows) * sizeof(v_t)))
	free(entries);
}

/*-------------------------------------------------------------------*/
static void gpu_matrix_free(packed_matrix_t *p) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;

	CUDA_TRY(cuMemFree(d->matmul_scratch))

	for (i = 0; i < d->num_block_rows; i++) {
		block_row_t *b = d->block_rows + i;

		CUDA_TRY(cuMemFree(b->row_entries))
		CUDA_TRY(cuMemFree(b->col_entries))
	}
	free(d->block_rows);

	for (i = 0; i < d->num_trans_block_rows; i++) {
		block_row_t *b = d->trans_block_rows + i;

		CUDA_TRY(cuMemFree(b->row_entries))
		CUDA_TRY(cuMemFree(b->col_entries))
	}
	free(d->trans_block_rows);

	for (i = 0; i < (p->num_dense_rows + VBITS - 1) / VBITS; i++)
		CUDA_TRY(cuMemFree(d->dense_blocks[i]))
	free(d->dense_blocks);
}

/*------------------------------------------------------------------------*/
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

	sprintf(libname, "mgpu/spmv_engine%s", suffix);

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
	d->spmv_engine_preprocess = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_preprocess");
	d->spmv_engine_run = get_lib_symbol(
					d->spmv_engine_handle,
					"spmv_engine_run");
	if (d->spmv_engine_init == NULL ||
	    d->spmv_engine_free == NULL ||
	    d->spmv_engine_preprocess == NULL ||
	    d->spmv_engine_run == NULL) {
		printf("error: cannot find GPU matrix multiply function\n");
		exit(-1);
	}
}

/*-------------------------------------------------------------------*/
void matrix_extra_init(msieve_obj *obj, packed_matrix_t *p,
			uint32 first_block_size) {

	uint32 i;
	gpudata_t *d;
	gpu_config_t gpu_config;
	gpu_info_t *gpu_info;

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

	load_spmv_engine(obj, d);
	d->spmv_engine_init(obj->which_gpu);

	/* initialize context */

	CUDA_TRY(cuDevicePrimaryCtxRetain(&d->gpu_context,
			d->gpu_info->device_handle))

	/* load kernels */

	CUDA_TRY(cuModuleLoad(&d->gpu_module, "lanczos_kernel.ptx"))

	d->launch = (gpu_launch_t *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(gpu_launch_t));

	for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
		gpu_launch_t *launch = d->launch + i;

		gpu_launch_init(d->gpu_module, gpu_kernel_names[i],
				launch);

		launch->threads_per_block = MIN(256, 
				launch->threads_per_block);
	}

	/* allocate scratch arrays */

	CUDA_TRY(cuMemAlloc(&d->gpu_scratch, VBITS * sizeof(v_t)))

	/* set up the matrix on the card */

	gpu_matrix_init(p);
}

/*-------------------------------------------------------------------*/
void matrix_extra_free(packed_matrix_t *p) {

	gpudata_t *d = (gpudata_t *)p->extra;

	gpu_matrix_free(p);

	CUDA_TRY(cuMemFree(d->gpu_scratch))

	free(d->launch);

	d->spmv_engine_free();
	unload_dynamic_lib(d->spmv_engine_handle);

	CUDA_TRY(cuDevicePrimaryCtxRelease(d->gpu_info->device_handle)) 

	free(d->gpu_info);
	free(d);
}

/*-------------------------------------------------------------------*/
static void mul_packed_gpu(packed_matrix_t *p, 
				gpuvec_t *x, gpuvec_t *b) {

	uint32 i;
	uint32 start_col = 0;
	gpudata_t *d = (gpudata_t *)p->extra;

	CUDA_TRY(cuMemsetD8(b->gpu_vec, 0, 
			p->nrows * sizeof(v_t)));

	/* sweep through the matrix a block col at a time */

	for (i = 0; i < d->num_block_rows; i++) {

		CUDA_TRY(cuMemsetD8(d->matmul_scratch, 0, 
				p->nrows * sizeof(v_t)))

		block_row_t *blk = d->block_rows + i;
		spmv_data_t spmv_data;

		spmv_data.num_rows = blk->num_rows;
		spmv_data.num_col_entries = blk->num_col_entries;
		spmv_data.col_entries = blk->col_entries;
		spmv_data.row_entries = blk->row_entries;
		spmv_data.vector_in = (CUdeviceptr)((v_t *)x->gpu_vec + start_col);
		spmv_data.vector_out = d->matmul_scratch;

		d->spmv_engine_run(blk->spmv_preprocess_handle, &spmv_data);

		{
			/* combine with previous output */

			gpu_launch_t *launch = d->launch + GPU_K_XOR;
			uint32 n = p->nrows;

			uint32 num_blocks = (n + launch->threads_per_block - 1) / 
					launch->threads_per_block;

			void *args[3] = {&b->gpu_vec, &d->matmul_scratch, &n};

			CUDA_TRY(cuLaunchKernel(launch->kernel_func, 
				MIN(1000, num_blocks), 1, 1, launch->threads_per_block, 1, 1,
				0, NULL, args, NULL))

		}

		start_col += p->preferred_block;
	}

	/* handle dense rows */

	for (i = 0; i < (p->num_dense_rows + VBITS - 1) / VBITS; i++) {
		mul_BxN_NxB_gpu(p, 
			d->dense_blocks[i], 
			x->gpu_vec, 
			(CUdeviceptr)((v_t *)b->gpu_vec + VBITS * i), 
			p->ncols);
	}
}

/*-------------------------------------------------------------------*/
static void mul_packed_trans_gpu(packed_matrix_t *p, 
				gpuvec_t *x, gpuvec_t *b) {

	uint32 i;
	uint32 start_row = 0;
	gpudata_t *d = (gpudata_t *)p->extra;

	CUDA_TRY(cuMemsetD8(b->gpu_vec, 0, 
			p->ncols * sizeof(v_t)));

	/* sweep through the matrix a block row at a time */

	for (i = 0; i < d->num_trans_block_rows; i++) {

		CUDA_TRY(cuMemsetD8(d->matmul_scratch, 0, 
				p->ncols * sizeof(v_t)))

		block_row_t *blk = d->trans_block_rows + i;
		spmv_data_t spmv_data;

		spmv_data.num_rows = blk->num_rows;
		spmv_data.num_col_entries = blk->num_col_entries;
		spmv_data.col_entries = blk->col_entries;
		spmv_data.row_entries = blk->row_entries;
		spmv_data.vector_in = (CUdeviceptr)((v_t *)x->gpu_vec + start_row);
		spmv_data.vector_out = d->matmul_scratch;

		d->spmv_engine_run(blk->spmv_preprocess_handle, &spmv_data);

		{
			/* combine with previous output */

			gpu_launch_t *launch = d->launch + GPU_K_XOR;
			uint32 n = p->ncols;

			uint32 num_blocks = (n + launch->threads_per_block - 1) / 
					launch->threads_per_block;

			void *args[3] = {&b->gpu_vec, &d->matmul_scratch, &n};

			CUDA_TRY(cuLaunchKernel(launch->kernel_func, 
				MIN(1000, num_blocks), 1, 1, launch->threads_per_block, 1, 1,
				0, NULL, args, NULL))

		}

		start_row += p->preferred_block;
	}

	/* handle dense rows */

	for (i = 0; i < (p->num_dense_rows + VBITS - 1) / VBITS; i++) {
		mul_NxB_BxB_acc_gpu(p, 
			d->dense_blocks[i], 
			(CUdeviceptr)((v_t *)x->gpu_vec + VBITS * i),
			(CUdeviceptr)((v_t *)b->gpu_vec + VBITS * i), 
			p->ncols);
	}
}

/*-------------------------------------------------------------------*/
void mul_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	mul_packed_gpu(A, x, b);

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

	mul_packed_trans_gpu(A, x, b);

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

	uint32 i, max_block_rows;
	size_t mem_use, tot_mem_use;
	gpudata_t *d = (gpudata_t*) p->extra;

	/* account for the vectors used in the lanczos iteration */

#ifdef HAVE_MPI
	mem_use = (6 * p->nsubcols + 2 * 
			MAX(p->nrows, p->ncols)) * sizeof(v_t);
#else
	mem_use = 7 * p->max_ncols * sizeof(v_t);
#endif

	/* and for the vv kernel scratch array */

	mem_use += VBITS * sizeof(v_t);

	tot_mem_use = mem_use;
	printf("vector memory use: %.1f MB\n", (double)mem_use/1048576);

	/* and for the matrix */

	/* dense rows */
	mem_use = ((p->num_dense_rows + VBITS - 1) / VBITS) * p->ncols * sizeof(v_t);

	tot_mem_use += mem_use;
	printf("dense rows memory use: %.1f MB\n", (double)mem_use/1048576);

	/* spmv scratch array */
	mem_use = MAX(p->ncols, p->nrows) * sizeof(v_t);

	/* matrix in CSR format */
	max_block_rows = 0;
	for (i = 0; i < d->num_block_rows; i++) {
		block_row_t *b = d->block_rows + i;
		mem_use += (b->num_rows + b->num_col_entries) * sizeof(uint32);
		if (b->num_rows > max_block_rows) max_block_rows = b->num_rows;
	}

	/* transpose matrix in CSR format */
	for (i = 0; i < d->num_trans_block_rows; i++) {
		block_row_t *b = d->trans_block_rows + i;
		mem_use += (b->num_rows + b->num_col_entries) * sizeof(uint32);
		if (b->num_rows > max_block_rows) max_block_rows = b->num_rows;
	}

	/* second copy of block row array for stripping empty rows in MGPU spmv library */
	mem_use += (max_block_rows + 1) * sizeof(uint32);

	tot_mem_use += mem_use;
	printf("sparse matrix memory use: %.1f MB\n", (double)mem_use/1048576);
	printf("  empty rows memory use: %.1f MB\n", (double)(max_block_rows + 1) * sizeof(uint32) / 1048576);

	return tot_mem_use;
}
