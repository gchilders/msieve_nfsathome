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
	"lanczos_kernel_gather",
	"lanczos_kernel_fixup",
};

static const gpu_arg_type_list_t gpu_kernel_args[] = 
{
	/* lanczos_kernel_mask */
	{ 3,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_UINT64,
		  GPU_ARG_UINT32,
		}
	},
	/* lanczos_kernel_xor */
	{ 3,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		}
	},
	/* lanczos_kernel_inner_prod */
	{ 4,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		}
	},
	/* lanczos_kernel_outer_prod */
	{ 4,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		}
	},
	/* lanczos_kernel_gather */
	{ 3,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		}
	},
	/* lanczos_kernel_fixup */
	{ 4,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		}
	},
};

typedef struct {
	uint32 row_off;
	uint32 col_off;
} entry_idx_t;

/*-------------------------------------------------------------------*/
static void copy_dense(packed_matrix_t *p) 
{
	/* copy the dense arrays to device memory */

	uint32 i, j;
	uint32 ncols = p->ncols;
	gpudata_t *d = (gpudata_t *)p->extra;
	uint32 num_dense_blocks = (p->num_dense_rows + 63) / 64;
	uint64 *tmp = (uint64 *)xmalloc(ncols * sizeof(uint64));

	d->dense_blocks = (CUdeviceptr *)xmalloc(num_dense_blocks *
						sizeof(CUdeviceptr));

	for (i = 0; i < num_dense_blocks; i++) {

		for (j = 0; j < ncols; j++) {
			la_col_t *col = p->unpacked_cols + j;
			uint32 *src = col->data + col->weight;

			tmp[j] = (uint64)src[2 * i + 1] << 32 |
				 (uint64)src[2 * i];
		}

		CUDA_TRY(cuMemAlloc(&d->dense_blocks[i], 
					ncols * sizeof(uint64)))
		CUDA_TRY(cuMemcpyHtoD(d->dense_blocks[i], tmp,
					ncols * sizeof(uint64)))
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
static void pack_matrix_block(block_row_t *b,
			entry_idx_t *entries, uint32 num_entries,
			uint32 row_min, uint32 row_max)
{

	uint32 i, j;
	uint32 num_rows = row_max - row_min;
	uint32 curr_row;

	/* convert a block of matrix rows from COO to CSR format */

	uint32 *vector_off = (uint32 *)xmalloc(num_entries * 
					sizeof(uint32));
	uint32 *vector_off_start = (uint32 *)xmalloc((num_rows + 1) * 
					sizeof(uint32));

	qsort(entries, num_entries, sizeof(entry_idx_t), compare_row_off);

	for (i = j = 0; i < num_entries; i++) {

		uint32 row_off = entries[i].row_off - row_min;
		uint32 col_off = entries[i].col_off;

		vector_off[i] = col_off;
		while (j <= row_off)
			vector_off_start[j++] = i;
	}
	while (j <= num_rows)
		vector_off_start[j++] = i;

	b->num_rows = num_rows;
	b->num_entries = num_entries;
	printf("%u\n", num_entries);

	CUDA_TRY(cuMemAlloc(&b->vector_off,
				num_entries * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(b->vector_off,
				vector_off,
				num_entries * sizeof(uint32)))

	CUDA_TRY(cuMemAlloc(&b->vector_off_start,
				(num_rows + 1) * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(b->vector_off_start,
				vector_off_start,
				(num_rows + 1)  * sizeof(uint32)))

	free(vector_off);
	free(vector_off_start);
}

/*-------------------------------------------------------------------*/
static void gpu_matrix_init(packed_matrix_t *p) {

	uint32 start_row = 0;
	uint32 block_size = 1000;
	gpudata_t *d = (gpudata_t *)p->extra;

	uint32 num_block_rows = 0;
	uint32 num_block_rows_alloc = 100;
	block_row_t *block_rows = (block_row_t *)xmalloc(
					num_block_rows_alloc *
					sizeof(block_row_t));

	uint32 num_entries_alloc = 10000;
	uint32 max_entries = 0;
	entry_idx_t *entries = (entry_idx_t *)xmalloc(
					num_entries_alloc *
					sizeof(entry_idx_t));

	uint32 preferred_entries_per_block = 20000000;

	/* deal with the dense rows */

	copy_dense(p);

	/* deal with the sparse rows */

	while (start_row < p->nrows) {

		block_row_t *b;
		uint32 num_entries;

		num_entries = extract_block(p->unpacked_cols,
					start_row, 
					start_row + block_size,
					0, p->ncols,
					&entries,
					&num_entries_alloc);

		if (num_block_rows == num_block_rows_alloc) {
			num_block_rows_alloc *= 2;
			block_rows = (block_row_t *)xrealloc(
					block_rows,
					num_block_rows_alloc *
					sizeof(block_row_t));
		}

		b = block_rows + num_block_rows++;

		pack_matrix_block(b, entries, num_entries,
				start_row, start_row + block_size);

		max_entries = MAX(max_entries, num_entries);

		/* choose the next block size so that more of the
		   matrix is grabbed at a time as it becomes more
		   sparse */

		start_row += block_size;
		if (num_entries < preferred_entries_per_block) {
			block_size = MIN(p->nrows - start_row,
                                             2 * block_size);
                }
	}

	/* need working space large enough to fit the biggest
	   block previously allocated */

	CUDA_TRY(cuMemAlloc(&d->matmul_scratch,
				(max_entries + 1) * 
                                sizeof(uint64)))

	d->num_block_rows = num_block_rows;
	d->block_rows = block_rows;
	free(entries);
}

/*-------------------------------------------------------------------*/
static void gpu_matrix_free(packed_matrix_t *p) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;

	for (i = 0; i < d->num_block_rows; i++) {
		block_row_t *b = d->block_rows + i;

		CUDA_TRY(cuMemFree(b->vector_off))
		CUDA_TRY(cuMemFree(b->vector_off_start))
	}
	free(d->block_rows);

	CUDA_TRY(cuMemFree(d->matmul_scratch))

	for (i = 0; i < (p->num_dense_rows + 63) / 64; i++)
		CUDA_TRY(cuMemFree(d->dense_blocks[i]))
	free(d->dense_blocks);
}

/*------------------------------------------------------------------------*/
static void
load_scan_engine(msieve_obj *obj, gpudata_t *d)
{
	char libname[256];
	const char *arch;
	#if defined(WIN32) || defined(_WIN64)
	const char *suffix = ".dll";
	#else
	const char *suffix = ".so";
	#endif

	if (d->gpu_info->compute_version_major < 2) {
		printf("error: GPU compute capability >= 2.0 required\n");
		exit(-1);
	}

	if (d->gpu_info->compute_version_major == 2)
		arch = "sm20";
	else if (d->gpu_info->compute_version_minor >= 5)
		arch = "sm35";
	else
		arch = "sm30";

	sprintf(libname, "mgpu/scan_engine_%s%s", arch, suffix);

	/* override from input args */

	if (obj->nfs_args != NULL) {
		char *tmp = strstr(obj->nfs_args, "scanlib=");

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

	d->scan_engine_handle = load_dynamic_lib(libname);
	if (d->scan_engine_handle == NULL) {
		printf("error: failed to load GPU scanning engine\n");
		exit(-1);
	}

	/* the scan engine uses the same CUDA context */

	d->scan_engine_init = get_lib_symbol(
					d->scan_engine_handle,
					"scan_engine_init");
	d->scan_engine_free = get_lib_symbol(
					d->scan_engine_handle,
					"scan_engine_free");
	d->scan_engine_run = get_lib_symbol(
					d->scan_engine_handle,
					"scan_engine_run");
	if (d->scan_engine_init == NULL ||
	    d->scan_engine_free == NULL ||
	    d->scan_engine_run == NULL) {
		printf("error: cannot find GPU scanning function\n");
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

	load_scan_engine(obj, d);
	d->scan_engine = d->scan_engine_init(obj->which_gpu);

	/* initialize context */

	CUDA_TRY(cuCtxCreate(&d->gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			d->gpu_info->device_handle))

	/* load kernels */

	CUDA_TRY(cuModuleLoad(&d->gpu_module, "lanczos_kernel_sm20.ptx"))

	d->launch = (gpu_launch_t *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(gpu_launch_t));

	for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
		gpu_launch_t *launch = d->launch + i;

		gpu_launch_init(d->gpu_module, gpu_kernel_names[i],
				gpu_kernel_args + i, launch);

		launch->threads_per_block = MIN(256, 
				launch->threads_per_block);

		CUDA_TRY(cuFuncSetBlockShape(launch->kernel_func,
					launch->threads_per_block, 1, 1))
	}

	/* allocate scratch arrays */

	CUDA_TRY(cuMemAlloc(&d->inner_scratch, 256 * 8 * sizeof(uint64)))
	CUDA_TRY(cuMemAlloc(&d->outer_scratch, 64 * sizeof(uint64)))

	/* set the texture reference used by the matrix multiply */
	
	CUDA_TRY(cuModuleGetTexRef(&d->matmul_texref, 
				d->gpu_module, "matmul_tex"))

	CUDA_TRY(cuTexRefSetFlags(d->matmul_texref, 
				CU_TRSF_READ_AS_INTEGER))

	CUDA_TRY(cuTexRefSetFormat(d->matmul_texref,
				CU_AD_FORMAT_UNSIGNED_INT32, 2))

	/* set up the matrix on the card */

	gpu_matrix_init(p);
}

/*-------------------------------------------------------------------*/
void matrix_extra_free(packed_matrix_t *p) {

	gpudata_t *d = (gpudata_t *)p->extra;

	gpu_matrix_free(p);

	CUDA_TRY(cuMemFree(d->inner_scratch))
	CUDA_TRY(cuMemFree(d->outer_scratch))

	free(d->launch);

	d->scan_engine_free(d->scan_engine);
	unload_dynamic_lib(d->scan_engine_handle);

	CUDA_TRY(cuCtxDestroy(d->gpu_context)) 

	free(d->gpu_info);
	free(d);
}

/*-------------------------------------------------------------------*/
static void mul_packed_gpu(packed_matrix_t *p, gpuvec_t *x, gpuvec_t *b) {

	uint32 i;
	uint32 num_blocks;
	gpu_launch_t *launch;
	gpudata_t *d = (gpudata_t *)p->extra;
	uint32 start_row = 0;

	CUDA_TRY(cuTexRefSetAddress(NULL, d->matmul_texref, 
				x->gpu_vec, p->ncols * sizeof(uint64)))

	/* sweep through the matrix a block row at a time */

	for (i = 0; i < d->num_block_rows; i++) {

		block_row_t *blk = d->block_rows + i;
		scan_data_t scan_data;
		gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];

		/* gather entries of x into linear order */

		launch = d->launch + GPU_K_GATHER;
		num_blocks = (blk->num_entries + 
				launch->threads_per_block - 1) / 
				launch->threads_per_block;
		num_blocks = MIN(num_blocks, 
				(uint32)(25 * d->gpu_info->num_compute_units));

		gpu_args[0].ptr_arg = (void *)(size_t)blk->vector_off;
		gpu_args[1].ptr_arg = (void *)(size_t)d->matmul_scratch;
		gpu_args[2].uint32_arg = blk->num_entries;

		gpu_launch_set(launch, gpu_args);
		CUDA_TRY(cuLaunchGrid(launch->kernel_func, num_blocks, 1))

		/* compute a prefix scan of the whole list */

		scan_data.data_in = d->matmul_scratch;
		scan_data.num_elements = blk->num_entries;

		d->scan_engine_run(d->scan_engine, &scan_data);

		/* postprocess and store results to b */

		launch = d->launch + GPU_K_FIXUP;
		num_blocks = (blk->num_rows + 
				launch->threads_per_block - 1) / 
				launch->threads_per_block;
		num_blocks = MIN(num_blocks, 
				(uint32)(25 * d->gpu_info->num_compute_units));

		gpu_args[0].ptr_arg = (void *)(size_t)d->matmul_scratch;
		gpu_args[1].ptr_arg = (void *)(size_t)((uint64 *)b->gpu_vec +
							start_row);
		gpu_args[2].ptr_arg = (void *)(size_t)blk->vector_off_start;
		gpu_args[3].uint32_arg = blk->num_rows;

		gpu_launch_set(launch, gpu_args);
		CUDA_TRY(cuLaunchGrid(launch->kernel_func, num_blocks, 1))

		start_row += blk->num_rows;
	}

	/* handle dense rows */

	for (i = 0; i < (p->num_dense_rows + 63) / 64; i++) {
		v_mul_64xN_Nx64_gpu(p, 
			d->dense_blocks[i], 
			x->gpu_vec, 
			(CUdeviceptr)((uint64 *)b->gpu_vec + 64 * i), 
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
		uint32 i;
		uint64 *tmp = (uint64 *)xmalloc(A->ncols * 
						sizeof(uint64));

		CUDA_TRY(cuMemcpyDtoH(tmp, b->gpu_vec, 
					A->ncols * sizeof(uint64)))

		mul_unpacked(A, x->host_vec, b->host_vec);
#if 1
		for (i = 0; i < A->nrows; i++) {
			if (tmp[i] != b->host_vec[i]) {
				printf("error %u %" PRIx64 " %" PRIx64 "\n", 
						i, b->host_vec[i], tmp[i]);
				exit(-1);
			}
		}
#endif
		free(tmp);

		CUDA_TRY(cuMemcpyHtoD(b->gpu_vec, b->host_vec, 
		 		A->ncols * sizeof(uint64)))
	}
#else
	CUDA_TRY(cuMemcpyDtoH(b->host_vec, b->gpu_vec, 
		 		A->ncols * sizeof(uint64)))
#endif
}

/*-------------------------------------------------------------------*/
void mul_trans_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	mul_trans_unpacked(A, x->host_vec, b->host_vec);

	CUDA_TRY(cuMemcpyHtoD(b->gpu_vec, b->host_vec, 
			       A->ncols * sizeof(uint64)))
}

/*-------------------------------------------------------------------*/
size_t packed_matrix_sizeof(packed_matrix_t *p) {

	/* FIXME */
	return 0;
}

