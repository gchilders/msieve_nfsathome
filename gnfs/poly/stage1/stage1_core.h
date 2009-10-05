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

#ifndef _STAGE1_CORE_H_
#define _STAGE1_CORE_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
typedef unsigned int uint32;
typedef unsigned long long uint64;
#endif

/* structure indicating a collision */

typedef struct {
	uint32 p;
	uint32 q;
	uint64 offset;
	uint64 proot;
} found_t;

#define MAX_ROOTS 36

/* the outer loop needs parallel access to different p,
   so we store in SOA format. All the entries in the structure
   have the same number of roots */

#define P_SOA_BATCH_SIZE 8192

typedef struct {
	uint32 p[P_SOA_BATCH_SIZE];
	uint64 roots[MAX_ROOTS][P_SOA_BATCH_SIZE];
} p_soa_t;


/* the batch of p values used in the inner loop does not have
   a predefined size; instead, p values with different numbers
   of roots are compacted together contiguously, with unused
   entries in the roots[] array removed */

#define P_PACKED_HEADER_WORDS 2

typedef struct {
	uint32 p;
	uint32 lattice_size;
	uint32 num_roots;
	uint32 pad;
	uint64 roots[MAX_ROOTS];
} p_packed_t;

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_CORE_H_ */
