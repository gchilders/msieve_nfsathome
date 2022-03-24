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

#ifndef _COMMON_LANCZOS_CSR_LANCZOS_CSR_H_
#define _COMMON_LANCZOS_CSR_LANCZOS_CSR_H_

#include <omp.h>
#include "../lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	uint32 num_rows;
	uint32 num_cols;
	uint32 num_col_entries;         /* but int32 in cub */
	uint32 blocksize;
	uint32 *col_entries;        /* uint32 */
	uint32 *row_entries;        /* uint32 */
} block_row_t;

/* implementation-specific structure */

typedef struct {

	/* matrix data */

	v_t **dense_blocks;

	uint32 num_block_rows;
	block_row_t *block_rows;

	uint32 num_trans_block_rows;
	block_row_t *trans_block_rows;

} cpudata_t;

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_LANCZOS_CSR_LANCZOS_CSR_H_ */
