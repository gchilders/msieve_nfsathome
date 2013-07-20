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

static const char * gpu_kernel_names[] = 
{
	"lanczos_kernel_mask",
	"lanczos_kernel_xor",
	"lanczos_kernel_inner_prod",
	"lanczos_kernel_outer_prod",
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
};

/*-------------------------------------------------------------------*/
void matrix_extra_init(msieve_obj *obj, packed_matrix_t *p) {

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

	d = (gpudata_t *)xcalloc(1, sizeof(gpudata_t));

	d->gpu_info = gpu_info = (gpu_info_t *)xmalloc(sizeof(gpu_info_t));
	memcpy(gpu_info, gpu_config.info + obj->which_gpu,
			sizeof(gpu_info_t)); 

	logprintf(obj, "using GPU %u (%s)\n", obj->which_gpu, gpu_info->name);
	logprintf(obj, "selected card has CUDA arch %d.%d\n",
			gpu_info->compute_version_major,
			gpu_info->compute_version_minor);

	/* initialize context */

	CUDA_TRY(cuCtxCreate(&d->gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			d->gpu_info->device_handle))

//	CUDA_TRY(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1))

	/* load kernels */

	if (d->gpu_info->compute_version_major >= 2)
		CUDA_TRY(cuModuleLoad(&d->gpu_module, 
					"lanczos_kernel_sm20.ptx"))
	else
		CUDA_TRY(cuModuleLoad(&d->gpu_module, 
					"lanczos_kernel_sm11.ptx"))

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

	/* set the texture reference used by the inner product */
	
	CUDA_TRY(cuModuleGetTexRef(&d->inner_texref, 
				d->gpu_module, "inner_tex"))

	CUDA_TRY(cuTexRefSetAddress(NULL, d->inner_texref, 
				d->inner_scratch, 256 * 8 * sizeof(uint64)))

	CUDA_TRY(cuTexRefSetFlags(d->inner_texref, 
				CU_TRSF_READ_AS_INTEGER))

	CUDA_TRY(cuTexRefSetFormat(d->inner_texref,
				CU_AD_FORMAT_UNSIGNED_INT32, 2))
	p->extra = d;
}

/*-------------------------------------------------------------------*/
void matrix_extra_free(packed_matrix_t *p) {

	gpudata_t *d = (gpudata_t *)p->extra;

	CUDA_TRY(cuMemFree(d->inner_scratch))
	CUDA_TRY(cuMemFree(d->outer_scratch))

	free(d->launch);

	CUDA_TRY(cuCtxDestroy(d->gpu_context)) 

	free(d->gpu_info);
	free(d);
}

/*-------------------------------------------------------------------*/
static void mul_one_block(packed_block_t *curr_block,
			uint64 *curr_col, uint64 *curr_b) {

	uint32 i;
	uint32 num_entries = curr_block->num_entries;
	entry_idx_t *entries = curr_block->d.entries;

	#define _txor(x) curr_b[entries[i+x].row_off] ^= \
				 curr_col[entries[i+x].col_off]

	for (i = 0; i < (num_entries & (uint32)(~15)); i += 16) {
		_txor( 0); _txor( 1); _txor( 2); _txor( 3);
		_txor( 4); _txor( 5); _txor( 6); _txor( 7);
		_txor( 8); _txor( 9); _txor(10); _txor(11);
		_txor(12); _txor(13); _txor(14); _txor(15);
	}

	#undef _txor

	for (; i < num_entries; i++) {
		curr_b[entries[i].row_off] ^= curr_col[entries[i].col_off];
	}
}

/*-------------------------------------------------------------------*/
static void mul_packed_cpu(packed_matrix_t *p, uint64 *x, uint64 *b) {

	uint32 i, j;

	memset(b, 0, p->nrows * sizeof(uint64));

	for (i = 0; i < p->num_block_rows; i++) {

		packed_block_t *curr_block = p->blocks + 
					i * p->num_block_cols;
		uint64 *curr_x = x;
		uint32 b_off = 0;

		if (i > 0)
			b_off = (i - 1) * p->block_size + p->first_block_size;

		for (j = 0; j < p->num_block_cols; j++) {
			mul_one_block(curr_block, curr_x, b + b_off);
			curr_block++;
			curr_x += p->block_size;
		}
	}

	for (i = 0; i < (p->num_dense_rows + 63) / 64; i++)
		v_mul_64xN_Nx64_cpu(p->dense_blocks[i],
				x, b + 64 * i, p->ncols);
}

/*-------------------------------------------------------------------*/
void mul_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	if (A->unpacked_cols)
		mul_unpacked(A, x->host_vec, b->host_vec);
	else
		mul_packed_cpu(A, x->host_vec, b->host_vec);

	CUDA_TRY(cuMemcpyHtoD(b->gpu_vec, b->host_vec, 
			       A->ncols * sizeof(uint64)))
}

/*-------------------------------------------------------------------*/
static void mul_trans_one_block(packed_block_t *curr_block,
				uint64 *curr_row, uint64 *curr_b) {

	uint32 i;
	uint32 num_entries = curr_block->num_entries;
	entry_idx_t *entries = curr_block->d.entries;

	#define _txor(x) curr_b[entries[i+x].col_off] ^= \
				 curr_row[entries[i+x].row_off]	

	for (i = 0; i < (num_entries & (uint32)(~15)); i += 16) {
		_txor( 0); _txor( 1); _txor( 2); _txor( 3);
		_txor( 4); _txor( 5); _txor( 6); _txor( 7);
		_txor( 8); _txor( 9); _txor(10); _txor(11);
		_txor(12); _txor(13); _txor(14); _txor(15);
	}

	#undef _txor

	for (; i < num_entries; i++) {
		curr_b[entries[i].col_off] ^= curr_row[entries[i].row_off];
	}
}

/*-------------------------------------------------------------------*/
static void mul_trans_packed_cpu(packed_matrix_t *p, 
			uint64 *x, uint64 *b) {
	uint32 i, j;

	memset(b, 0, p->ncols * sizeof(uint64));

	for (i = 0; i < p->num_block_cols; i++) {

		packed_block_t *curr_block = p->blocks + i;
		uint32 b_off = i * p->block_size;
		uint64 *curr_x = x;

		for (j = 0; j < p->num_block_rows; j++) {
			mul_trans_one_block(curr_block, curr_x, b + b_off);
			curr_block += p->num_block_cols;
			curr_x += (j == 0) ? p->first_block_size : 
					p->block_size;
		}
	}

	for (i = 0; i < (p->num_dense_rows + 63) / 64; i++)
		v_mul_Nx64_64x64_acc_cpu(p->dense_blocks[i],
					x + 64 * i, b, p->ncols);
}

/*-------------------------------------------------------------------*/
void mul_trans_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	if (A->unpacked_cols)
		mul_trans_unpacked(A, x->host_vec, b->host_vec);
	else
		mul_trans_packed_cpu(A, x->host_vec, b->host_vec);

	CUDA_TRY(cuMemcpyHtoD(b->gpu_vec, b->host_vec, 
			       A->ncols * sizeof(uint64)))
}
