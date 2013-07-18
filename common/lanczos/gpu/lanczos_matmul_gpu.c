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
void mul_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	mul_unpacked(A, x->host_vec, b->host_vec);

	CUDA_TRY(cuMemcpyHtoD(b->gpu_vec, b->host_vec, 
			       A->ncols * sizeof(uint64)))
}

/*-------------------------------------------------------------------*/
void mul_trans_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *b = (gpuvec_t *)b_in;

	mul_trans_unpacked(A, x->host_vec, b->host_vec);

	CUDA_TRY(cuMemcpyHtoD(b->gpu_vec, b->host_vec, 
			       A->ncols * sizeof(uint64)))
}
