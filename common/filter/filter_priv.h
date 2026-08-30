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

/* The reverse-list hot path is bandwidth-bound. Both values are bounded by
   the 4-billion-relation common-filter ceiling and the maximum 100 ideals in
   a packed relation, so 48 bits is ample for either a packed-array word
   offset or a reverse-list link (the worst case is below 2^39). Packing the
   pair into 12 bytes saves 25% versus two uint64 values without giving up the
   large-dataset headroom. */
#define IDEAL_RELATION_VALUE_BITS 48
#define IDEAL_RELATION_VALUE_MASK ((((uint64)1) << IDEAL_RELATION_VALUE_BITS) - 1)

typedef struct {
	uint32 relation_lo;
	uint32 next_lo;
	uint32 hi;       /* relation high 16 bits, then next high 16 bits */
} ideal_relation_t;

static INLINE void ideal_relation_set(ideal_relation_t *entry,
				uint64 relation_array_word, uint64 next) {
	if (relation_array_word > IDEAL_RELATION_VALUE_MASK ||
	    next > IDEAL_RELATION_VALUE_MASK) {
		printf("error: reverse ideal-list value exceeds 48 bits\n");
		exit(-1);
	}
	entry->relation_lo = (uint32)relation_array_word;
	entry->next_lo = (uint32)next;
	entry->hi = (uint32)((relation_array_word >> 32) & 0xffff) |
		((uint32)((next >> 32) & 0xffff) << 16);
}

static INLINE uint64 ideal_relation_word(const ideal_relation_t *entry) {
	return (uint64)entry->relation_lo |
		((uint64)(entry->hi & 0xffff) << 32);
}

static INLINE uint64 ideal_relation_next(const ideal_relation_t *entry) {
	return (uint64)entry->next_lo |
		((uint64)(entry->hi >> 16) << 32);
}

/* Reverse ideal->relation lists can contain more than 2^32 entries even
   when relation and ideal IDs themselves remain 32-bit. Fixed segments avoid
   huge reallocations. */
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

/* All offsets traversed after construction were produced by append(), so the
   hot BFS path can skip redundant bounds checks. */
static INLINE ideal_relation_t *ideal_relation_list_at_fast(
				ideal_relation_list_t *list, uint64 offset) {
	return list->segments[offset >> IDEAL_RELATION_SEGMENT_BITS] +
		(offset & IDEAL_RELATION_SEGMENT_MASK);
}

static INLINE uint64 ideal_relation_list_append(ideal_relation_list_t *list,
				uint64 relation_array_word, uint64 next) {
	uint64 offset = list->num_used;
	uint64 segment64 = offset >> IDEAL_RELATION_SEGMENT_BITS;
	uint32 segment;
	ideal_relation_t *entry;

	if (offset > IDEAL_RELATION_VALUE_MASK ||
	    relation_array_word > IDEAL_RELATION_VALUE_MASK ||
	    next > IDEAL_RELATION_VALUE_MASK) {
		printf("error: reverse ideal list exceeds 48-bit packed range\n");
		exit(-1);
	}
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
	entry = list->segments[segment] +
		(offset & IDEAL_RELATION_SEGMENT_MASK);
	ideal_relation_set(entry, relation_array_word, next);
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

/* Keep the ideal map at eight bytes, matching the original cache footprint.
   The payload needs far fewer than 62 bits; the top two bits carry the clique
   traversal flags. */
#define IDEAL_MAP_CONNECTED (((uint64)1) << 62)
#define IDEAL_MAP_CLIQUE    (((uint64)1) << 63)
#define IDEAL_MAP_PAYLOAD_MASK (IDEAL_MAP_CONNECTED - 1)

typedef struct {
	uint64 data;
} ideal_map_t;

static INLINE uint64 ideal_map_payload(const ideal_map_t *map) {
	return map->data & IDEAL_MAP_PAYLOAD_MASK;
}

static INLINE void ideal_map_set_payload(ideal_map_t *map, uint64 payload) {
	if (payload > IDEAL_MAP_PAYLOAD_MASK) {
		printf("error: ideal-map payload exceeds packed range\n");
		exit(-1);
	}
	map->data = (map->data & ~IDEAL_MAP_PAYLOAD_MASK) | payload;
}

static INLINE uint32 ideal_map_is_clique(const ideal_map_t *map) {
	return (map->data & IDEAL_MAP_CLIQUE) != 0;
}

static INLINE uint32 ideal_map_is_connected(const ideal_map_t *map) {
	return (map->data & IDEAL_MAP_CONNECTED) != 0;
}

static INLINE void ideal_map_set_clique(ideal_map_t *map) {
	map->data |= IDEAL_MAP_CLIQUE;
}

static INLINE void ideal_map_set_connected(ideal_map_t *map) {
	map->data |= IDEAL_MAP_CONNECTED;
}

/* a relation_set_t simulates matrix rows; the following simulates
   the matrix columns, mapping ideals to the relation sets
   containing those ideals */

typedef struct ideal_set_t {
	uint32 num_relsets;     /* the number of relation sets
				   containing this ideal */
	uint32 num_relsets_alloc; /* maximum number of relset numbers the
				     'relsets' array can hold */
	uint32 min_relset_size; /* the number of ideals in the
				   lightest relation set that
				   contains this ideal */
	uint32 active;          /* 1 if ideal is active, 0 if inactive */
	uint32 *relsets;        /* list of members in an array of
				   relation sets that contain this
				   ideal (no ordering assumed) */
	uint32 next;            /* compact heap-list node reference */
	uint32 prev;            /* compact heap-list node reference */
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
