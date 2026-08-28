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

/* implementation of number field sieve filtering */

#ifndef _GNFS_FILTER_FILTER_H_
#define _GNFS_FILTER_FILTER_H_

#include <common/filter/filter.h>
#include "gnfs.h"

/* Keep a small but explicit namespace reserve below UINT32_MAX. The
   common-filter arithmetic that can grow counts is widened or checked; the
   remaining reserve protects sentinel use and leaves room for small
   bookkeeping additions without pretending UINT32_MAX itself is usable. */
#define NFS_COMMON_FILTER_SAFE_MAX ((uint64)4000000000ULL)

#ifdef __cplusplus
extern "C" {
#endif

/* create '<savefile_name>.d', a binary file containing
   the line numbers of duplicated or corrupted relations.
   Duplicate removal only applies to the first max_relations
   relations found (or all relations if zero). The return
   value is the large prime bound to use for the singleton removal */

uint32 nfs_purge_duplicates(msieve_obj *obj, factor_base_t *fb,
				uint64 max_relations,
				uint64 *num_relations_out);

/* read '<savefile_name>.d' and create '<savefile_name>.lp', a
   binary file containing the relations surviving the singleton
   removal pass. If pass = 0, the .d file is assumed to contain
   relation numbers to skip; otherwise it contains relation numbers
   to keep */

void nfs_write_lp_file(msieve_obj *obj, factor_base_t *fb,
			filter_t *filter, uint64 max_relations,
			uint32 pass);

/* Convert the NFS-specific 64-bit LP intermediate file to the
   traditional 32-bit common-filter format. Disk singleton removal is
   performed as needed until both surviving relation and ideal IDs fit
   in 32 bits. A <savefile>.rmap file records dense relation ID ->
   original 64-bit savefile relation number. */

void nfs_compact_lp_file(msieve_obj *obj, filter_t *filter,
			uint64 ram_size);

#ifdef __cplusplus
}
#endif

#endif /* _GNFS_FILTER_FILTER_H_ */
