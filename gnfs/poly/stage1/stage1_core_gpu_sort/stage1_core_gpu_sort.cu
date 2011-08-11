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

#include "../cuda_intrinsics.h"
#include "stage1_core_gpu_sort.h"

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_trans(uint32 *p_array, uint32 num_p, uint64 *start_roots,
			uint64 *roots, uint32 num_roots,
			uint32 q, uint64 qroot)
{
	uint32 offset, p, p2_w, end;
	uint64 p2, p2_r, q2, tmp, inv;

	offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset >= num_p)
		return;

	p = p_array[offset];
	p2 = wide_sqr32(p);
	p2_w = montmul24_w((uint32)p2);
	p2_r = montmul48_r(p2, p2_w);
	q2 = wide_sqr32(q) % p2;
	end = num_p * num_roots;

	if (gcd32(p, q) != 1) {
		for (; offset < end; offset += num_p)
			roots[offset] = (uint64)(-1);

		return;
	}

	tmp = modinv32(q % p, p);
	tmp = wide_sqr32(tmp);
	tmp = montmul48(tmp, p2_r, p2, p2_w);
	inv = montmul48(q2, tmp, p2, p2_w);
	inv = modsub64((uint64)2, inv, p2);
	inv = montmul48(inv, tmp, p2, p2_w);
	inv = montmul48(inv, p2_r, p2, p2_w);

	for (; offset < end; offset += num_p) {
		uint64 proot = start_roots[offset];

		proot = modsub64(proot, qroot % p2, p2);
		proot = montmul48(proot, inv, p2, p2_w);

		roots[offset] = proot;
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_step(uint32 *p_array, uint32 num_p, uint64 *roots,
			uint32 num_roots, uint32 *p_out, uint64 *roots_out,
			uint64 sieve_end)
{
	uint32 offset, p, end;
	uint64 p2;

	offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset >= num_p)
		return;

	p = p_array[offset];
	p2 = wide_sqr32(p);

	end = num_p * num_roots;
	for (; offset < end; offset += num_p) {
		uint32 tmp_p;
		uint64 tmp_root;

		tmp_root = roots[offset];
		if (tmp_root < sieve_end) {
			tmp_p = p;
			roots[offset] += MIN(p2, (uint64)(-1) - tmp_root);
		}
		else {
			tmp_p = (uint32)(-1);
		}
		p_out[offset] = tmp_p;
		roots_out[offset] = tmp_root;
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_sort(uint32 *p_array, uint64 *roots)
{
	uint32 my_threadid, offset, j, k, u, dir;
	extern __shared__ char shared_cache[];
	uint32 *p_cache;
	uint64 *root_cache;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	offset = my_threadid + blockIdx.x * blockDim.x;

	p_cache = (uint32 *)shared_cache;
	root_cache = (uint64 *)(p_cache + blockDim.x * 2);

	p_cache[threadIdx.x] = p_array[offset];
	p_cache[threadIdx.x + blockDim.x] = p_array[offset + blockDim.x];

	root_cache[threadIdx.x] = roots[offset];
	root_cache[threadIdx.x + blockDim.x] = roots[offset + blockDim.x];

	__syncthreads();

	for (j = 1; j <= blockDim.x; j *= 2) {

		dir = !!(my_threadid & j);
		for (k = j; k; k /= 2) {

			u = (threadIdx.x & ~(k - 1)) * 2 +
						(threadIdx.x & (k - 1));

			if (dir != (root_cache[u] > root_cache[u + k])) {

				uint64 tmp = p_cache[u];
				p_cache[u] = p_cache[u + k];
				p_cache[u + k] = (uint32)tmp;

				tmp = root_cache[u];
				root_cache[u] = root_cache[u + k];
				root_cache[u + k] = tmp;
			}

			__syncthreads();
		}
	}

	p_array[offset] = p_cache[threadIdx.x];
	p_array[offset + blockDim.x] = p_cache[threadIdx.x + blockDim.x];

	roots[offset] = root_cache[threadIdx.x];
	roots[offset + blockDim.x] = root_cache[threadIdx.x + blockDim.x];
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_merge(uint32 *p_array, uint64 *roots, uint32 j)
{
	uint32 my_threadid, offset, k, u, dir;
	extern __shared__ char shared_cache[];
	uint32 *p_cache;
	uint64 *root_cache;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	offset = my_threadid + blockIdx.x * blockDim.x;

	p_cache = (uint32 *)shared_cache;
	root_cache = (uint64 *)(p_cache + blockDim.x * 2);

	p_cache[threadIdx.x] = p_array[offset];
	p_cache[threadIdx.x + blockDim.x] = p_array[offset + blockDim.x];

	root_cache[threadIdx.x] = roots[offset];
	root_cache[threadIdx.x + blockDim.x] = roots[offset + blockDim.x];

	__syncthreads();

	dir = !!(my_threadid & j);
	for (k = blockDim.x; k; k /= 2) {

		u = (threadIdx.x & ~(k - 1)) * 2 +
					(threadIdx.x & (k - 1));

		if (dir != (root_cache[u] > root_cache[u + k])) {

			uint64 tmp = p_cache[u];
			p_cache[u] = p_cache[u + k];
			p_cache[u + k] = (uint32)tmp;

			tmp = root_cache[u];
			root_cache[u] = root_cache[u + k];
			root_cache[u + k] = tmp;
		}

		__syncthreads();
	}

	p_array[offset] = p_cache[threadIdx.x];
	p_array[offset + blockDim.x] = p_cache[threadIdx.x + blockDim.x];

	roots[offset] = root_cache[threadIdx.x];
	roots[offset + blockDim.x] = root_cache[threadIdx.x + blockDim.x];
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_merge1(uint32 *p_array, uint64 *roots, uint32 j, uint32 k)
{
	uint32 my_threadid, offset, p_1, p_2;
	uint64 root_1, root_2;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	offset = (my_threadid & ~(k - 1)) * 2 + (my_threadid & (k - 1));

	p_1 = p_array[offset];
	p_2 = p_array[offset + k];
	root_1 = roots[offset];
	root_2 = roots[offset + k];

	if ((!!(my_threadid & j)) != (root_1 > root_2)) {

		p_array[offset] = p_2;
		p_array[offset + k] = p_1;

		roots[offset] = root_2;
		roots[offset + k] = root_1;
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_final(uint32 *p_array, uint64 *roots, uint32 num_entries,
			found_t *found_array)
{
	uint32 i, my_threadid, num_threads, p_1, p_2;
	uint64 root_1, root_2;

	i = my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;

	while (i < num_entries - 1) {
		p_1 = p_array[i];
		p_2 = p_array[i + 1];
		root_1 = roots[i];
		root_2 = roots[i + 1];

		if (p_1 < (uint32)(-1) && p_2 < (uint32)(-1) &&
		    root_1 == root_2) {

			if (gcd32(p_1, p_2) == 1) {

				found_array[my_threadid].p1 = p_1;
				found_array[my_threadid].p2 = p_2;
				found_array[my_threadid].root = root_1;
			}
		}

		i += num_threads;
	}
}

#ifdef __cplusplus
}
#endif
