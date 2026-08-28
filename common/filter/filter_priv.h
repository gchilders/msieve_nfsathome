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

/* implementation of relation filtering, using an intermediate
   representation for relations that is amenable to either QS or NFS.
   These routines work on an input filter_t structure and produce
   an output merge_t structure containing the completed relation sets */

#ifndef _COMMON_FILTER_FILTER_PRIV_H_
#define _COMMON_FILTER_FILTER_PRIV_H_

#include "filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* structure for the mapping between large ideals
   and relations (used during clique removal) */

typedef struct {
	uint64 relation_array_word;  /* 64-bit word offset into relation
					array where the relation starts */
	uint64 next;		     /* next relation containing this ideal */
} ideal_relation_t;

/* Reverse ideal->relation lists can contain more than 2^32 entries even
   when relation and ideal IDs themselves remain 32-bit (each weight-2 ideal
   contributes two entries). Allocate them in fixed segments so growth does
   not require enormous reallocations and use uint64 offsets throughout. */
#define IDEAL_RELATION_SEGMENT_BITS 20
#define IDEAL_RELATION_SEGMENT_SIZE ((uint64)1 << IDEAL_RELATION_SEGMENT_BITS)
#define IDEAL_RELATION_SEGMENT_MASK (IDEAL_RELATION_SEGMENT_SIZE - 1)

typedef struct {
	ideal_relation_t **segments;
	uint32 num_segments;
	uint32 segments_alloc;
	uint64 num_used;
} ideal_relation_list_t;

static INLINE void ideal_relation_list_init(ideal_relation_list_t *list) {
	memset(list, 0, sizeof(*list));
	list->num_used = 1; /* offset zero is the end-of-list sentinel */
}

static INLINE ideal_relation_t *ideal_relation_list_at(
				ideal_relation_list_t *list, uint64 offset) {
	uint64 segment = offset >> IDEAL_RELATION_SEGMENT_BITS;
	if (offset == 0 || segment >= list->num_segments ||
	    list->segments == NULL || list->segments[segment] == NULL) {
		printf("error: invalid reverse ideal-list offset\n");
		exit(-1);
	}
	return list->segments[segment] +
		(offset & IDEAL_RELATION_SEGMENT_MASK);
}

static INLINE uint64 ideal_relation_list_append(ideal_relation_list_t *list,
				uint64 relation_array_word, uint64 next) {
	uint64 offset = list->num_used;
	uint64 segment64 = offset >> IDEAL_RELATION_SEGMENT_BITS;
	uint32 segment;
	ideal_relation_t *entry;

	if (segment64 > UINT32_MAX) {
		printf("error: reverse ideal list exceeds addressable segment count\n");
		exit(-1);
	}
	segment = (uint32)segment64;
	if (segment >= list->num_segments) {
		if (segment >= list->segments_alloc) {
			uint32 new_alloc = list->segments_alloc ?
				list->segments_alloc * 2 : 16;
			if (new_alloc <= segment)
				new_alloc = segment + 1;
			list->segments = (ideal_relation_t **)xrealloc(
				list->segments, (size_t)new_alloc * sizeof(*list->segments));
			list->segments_alloc = new_alloc;
		}
		while (list->num_segments <= segment) {
			list->segments[list->num_segments++] =
				(ideal_relation_t *)xcalloc(
					(size_t)IDEAL_RELATION_SEGMENT_SIZE,
					 sizeof(ideal_relation_t));
		}
	}
	entry = ideal_relation_list_at(list, offset);
	entry->relation_array_word = relation_array_word;
	entry->next = next;
	list->num_used++;
	return offset;
}

static INLINE void ideal_relation_list_free(ideal_relation_list_t *list) {
	uint32 i;
	for (i = 0; i < list->num_segments; i++)
		free(list->segments[i]);
	free(list->segments);
	memset(list, 0, sizeof(*list));
}

/* structure used to map between a large ideal and a
   linked list of relations that use that ideal */

typedef struct {
	uint64 payload;	/* count, then offset in ideal_relation_t list;
				   can exceed 2^32 when billions of weight-2
				   ideals survive the prefilter */
	uint8 clique;      /* nonzero if this ideal can participate in
				   a clique */
	uint8 connected;   /* nonzero if this ideal has already been
				   added to a clique under construction */
} ideal_map_t;

/* a relation_set_t simulates matrix rows; the following simulates
   the matrix columns, mapping ideals to the relation sets
   containing those ideals */

typedef struct ideal_set_t {
	uint32 num_relsets;     /* the number of relation sets
				   containing this ideal */
	uint32 num_relsets_alloc; /* maximum number of relset numbers the
				     'relsets' array can hold */
	uint16 active;          /* 1 if ideal is active, 0 if inactive */
	uint32 min_relset_size; /* the number of ideals in the
				   lightest relation set that
				   contains this ideal */
	uint32 *relsets;        /* list of members in an array of
				   relation sets that contain this
				   ideal (no ordering assumed) */
	struct ideal_set_t *next;
	struct ideal_set_t *prev;  /* used to build circular linked lists */
} ideal_set_t;

/* relation sets with more than this many relations are deleted */

#define MAX_RELSET_SIZE 28

/* check relations array for errors */

void check_relations_array(filter_t *filter, uint32 location);

/* perform clique removal on the current set of relations */

void filter_purge_cliques(msieve_obj *obj, filter_t *filter);

/* initialize the merge process */

void filter_merge_init(msieve_obj *obj, filter_t *filter);

/* perform all 2-way merges, converting the results into
   relation-sets that the main merge routine operates on */

void filter_merge_2way(msieve_obj *obj, filter_t *filter, merge_t *merge);

/* do the rest of the merging. min_cycles is the minimum number
   of cycles that the input collection of relation-sets must
   produce, corresponding to the smallest matrix that can be
   built (the actual matrix is expected to be much larger than
   this). */

int32 filter_merge_full(msieve_obj *obj, merge_t *merge, uint32 min_cycles);

#ifdef __cplusplus
}
#endif

#endif /* _COMMON_FILTER_FILTER_PRIV_H_ */
