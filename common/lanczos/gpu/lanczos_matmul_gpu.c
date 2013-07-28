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
static void gpu_matrix_init(packed_matrix_t *p) {

	uint32 i;
	uint32 num_blocks = p->num_block_rows * p->num_block_cols;
	uint32 num_dense_blocks = (p->num_dense_rows + 63) / 64;
	uint32 num_nonzero;
	uint32 *tmp_counts;
	uint32 *tmp_starts;
	gpudata_t *d = (gpudata_t *)p->extra;

	/* copy the dense arrays */

	d->dense_blocks = (CUdeviceptr *)xmalloc(num_dense_blocks *
						sizeof(CUdeviceptr));
	for (i = 0; i < num_dense_blocks; i++) {
		CUDA_TRY(cuMemAlloc(&d->dense_blocks[i], 
					p->ncols * sizeof(uint64)))
		CUDA_TRY(cuMemcpyHtoD(d->dense_blocks[i],
					p->dense_blocks[i],
					p->ncols * sizeof(uint64)))
	}

	/* copy the sparse blocks (assume we can fit at most
	   4G matrix entries on the device) */

	tmp_counts = (uint32 *)xmalloc(num_blocks *
					sizeof(uint32));
	tmp_starts = (uint32 *)xmalloc(num_blocks *
					sizeof(uint32));

	for (i = num_nonzero = 0; i < num_blocks; i++) {
		tmp_counts[i] = p->blocks[i].num_entries;
		tmp_starts[i] = num_nonzero;
		num_nonzero += tmp_counts[i] + MATMUL_THREADS;
	}

	CUDA_TRY(cuMemAlloc(&d->block_num_entries,
				num_blocks * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(d->block_num_entries,
				tmp_counts,
				num_blocks * sizeof(uint32)))

	CUDA_TRY(cuMemAlloc(&d->block_entries_start,
				num_blocks * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(d->block_entries_start,
				tmp_starts,
				num_blocks * sizeof(uint32)))

	CUDA_TRY(cuMemAlloc(&d->block_entries,
				num_nonzero * sizeof(gpu_entry_idx_t)))

	for (i = 0; i < num_blocks; i++) {

		uint32 j;
		uint32 r, c_rem, curr_r, curr_c;
		gpu_entry_idx_t *gpu_marshall;
		gpu_entry_idx_t *g;
		packed_block_t *b = p->blocks + i;

		/* sort in row-major order */

		qsort(b->d.entries, b->num_entries, 
				sizeof(entry_idx_t), compare_row_off);

		gpu_marshall = (gpu_entry_idx_t *)xcalloc((b->num_entries + 
						MATMUL_THREADS),
						sizeof(gpu_entry_idx_t));

		r = (b->num_entries + MATMUL_THREADS - 1) / MATMUL_THREADS;
		c_rem = b->num_entries % MATMUL_THREADS;
		if (c_rem == 0)
			c_rem = MATMUL_THREADS;

		g = gpu_marshall + MATMUL_THREADS;

		for (j = curr_r = curr_c = 0; j < b->num_entries; j++) {

			entry_idx_t *e = b->d.entries + j;

			g->row_off = e->row_off;
			g->col_off = e->col_off;

			if (j == 0 || e[0].row_off != e[-1].row_off) {
				g->row_off_head = 1;
				gpu_marshall[curr_c].row_off = e->row_off;
				gpu_marshall[curr_c].row_off_head = 1;
			}

			g += MATMUL_THREADS;
			if (++curr_r == ((curr_c < c_rem) ? r : r - 1)) {
				curr_r = 0;
				curr_c++;
				g = gpu_marshall + MATMUL_THREADS + curr_c;
			}
		}

		/* the following assumes we can offset from a 
		   CUdeviceptr as if it was an ordinary host ptr */

		CUDA_TRY(cuMemcpyHtoD( (CUdeviceptr)(
					(entry_idx_t *)d->block_entries +
					tmp_starts[i]),
				gpu_marshall,
				(tmp_counts[i] + MATMUL_THREADS) * 
						sizeof(gpu_entry_idx_t)))

		free(gpu_marshall);
	}

	free(tmp_counts);
	free(tmp_starts);
#if 0
	{
		uint32 j, k;
		uint8 *row_bits;
		uint8 *col_bits;

		row_bits = (uint8 *)xmalloc((4*p->block_size + 7) / 8);
		col_bits = (uint8 *)xmalloc((4*p->block_size + 7) / 8);

		for (i = 0; i < p->num_block_rows; i++) {
			for (j = 0; j < p->num_block_cols; j++) {

				packed_block_t *b = p->blocks + 
						i * p->num_block_cols + j;
				uint32 num[4] = {0};

				memset(row_bits, 0, (4*p->block_size + 7) / 8);
				memset(col_bits, 0, (4*p->block_size + 7) / 8);

				for (k = 0; k < b->num_entries; k++) {

					entry_idx_t *e = b->d.entries + k;
					uint32 r = e->row_off;
					uint32 c = e->col_off;
					uint32 mask_r = (row_bits[r/2] >> 
							(4 * (r%2))) & 0xf;
					uint32 mask_c = (col_bits[c/2] >> 
							(4 * (c%2))) & 0xf;

					if ((mask_r & 0xc) == 0)
						mask_r |= 0x4;
					else if ((mask_r & 0xc) == 0x4)
						mask_r |= 0x8;

					row_bits[r/2] &= ~(0xf << (4 * (r%2)));
					row_bits[r/2] |= mask_r << (4 * (r%2));

					if ((mask_c & 0x3) == 0)
						mask_c |= 0x1;
					else if ((mask_c & 0x3) == 0x1)
						mask_c |= 0x2;

					col_bits[c/2] &= ~(0xf << (4 * (c%2)));
					col_bits[c/2] |= mask_c << (4 * (c%2));
				}

				for (k = 0; k < b->num_entries; k++) {

					entry_idx_t *e = b->d.entries + k;
					uint32 r = e->row_off;
					uint32 c = e->col_off;
					uint32 mask_r = (row_bits[r/2] >> 
							(4 * (r%2))) & 0xf;
					uint32 mask_c = (col_bits[c/2] >> 
							(4 * (c%2))) & 0xf;

					switch (mask_r | mask_c) {
						case 0x5:
							num[0]++; break;
						case 0x7:
							num[1]++; break;
						case 0xd:
							num[2]++; break;
						case 0xf:
							num[3]++; break;
					}
				}

				printf("%4u %4u: %u %u %u %u %u\n", 
						i, j, b->num_entries,
						num[0], num[1], num[2], num[3]);
			}
		}

		free(row_bits);
		free(col_bits);
	}
#endif
}

/*-------------------------------------------------------------------*/
static void gpu_matrix_free(packed_matrix_t *p) {

	uint32 i;
	gpudata_t *d = (gpudata_t *)p->extra;

	CUDA_TRY(cuMemFree(d->block_num_entries))
	CUDA_TRY(cuMemFree(d->block_entries_start))
	CUDA_TRY(cuMemFree(d->block_entries))

	for (i = 0; i < (p->num_dense_rows + 63) / 64; i++)
		CUDA_TRY(cuMemFree(d->dense_blocks[i]))
	free(d->dense_blocks);
}

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

	p->extra = d = (gpudata_t *)xcalloc(1, sizeof(gpudata_t));

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

	/* set the texture reference used by the matrix multiply */
	
	CUDA_TRY(cuModuleGetTexRef(&d->matmul_texref, 
				d->gpu_module, "matmul_tex"))

	CUDA_TRY(cuTexRefSetFlags(d->matmul_texref, 
				CU_TRSF_READ_AS_INTEGER))

	CUDA_TRY(cuTexRefSetFormat(d->matmul_texref,
				CU_AD_FORMAT_UNSIGNED_INT32, 2))

#if 0
	CUDA_TRY(cuTexRefSetAddress(NULL, d->matmul_texref, 
				d->inner_scratch, 256 * 8 * sizeof(uint64)))
#endif
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
