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

#include "filter_priv.h"
#include "merge_util.h"

/*--------------------------------------------------------------------*/
#define MERGE_POOL_MIN_WORDS 2
#define MERGE_POOL_MAX_WORDS 2048
#define MERGE_POOL_NUM_CLASSES 11
#define MERGE_POOL_SLAB_BYTES (16 * 1024 * 1024)

typedef struct merge_pool_slab_t {
	struct merge_pool_slab_t *next;
	void *mem;
} merge_pool_slab_t;

struct merge_mem_pool_t {
	void *free_list[MERGE_POOL_NUM_CLASSES];
	merge_pool_slab_t *slabs;
};

static uint32 merge_pool_class(uint32 words, uint32 *class_words) {
	uint32 idx = 0;
	uint32 w = MERGE_POOL_MIN_WORDS;
	while (w < words && w < MERGE_POOL_MAX_WORDS) {
		w <<= 1;
		idx++;
	}
	*class_words = w;
	return idx;
}

merge_mem_pool_t *merge_mem_pool_create(void) {
	return (merge_mem_pool_t *)xcalloc((size_t)1, sizeof(merge_mem_pool_t));
}

void merge_mem_pool_destroy(merge_mem_pool_t *pool) {
	merge_pool_slab_t *s;
	if (pool == NULL)
		return;
	s = pool->slabs;
	while (s != NULL) {
		merge_pool_slab_t *next = s->next;
		free(s->mem);
		free(s);
		s = next;
	}
	free(pool);
}

uint32 *merge_mem_alloc(merge_mem_pool_t *pool, uint32 words) {
	uint32 idx, class_words;
	void *p;
	if (words == 0)
		return NULL;
	if (pool == NULL || words > MERGE_POOL_MAX_WORDS)
		return (uint32 *)xmalloc((size_t)words * sizeof(uint32));
	idx = merge_pool_class(words, &class_words);
	if (pool->free_list[idx] == NULL) {
		size_t block_bytes = (size_t)class_words * sizeof(uint32);
		size_t slab_bytes = MERGE_POOL_SLAB_BYTES;
		size_t nblocks, i;
		uint8 *mem;
		merge_pool_slab_t *slab;
		if (slab_bytes < block_bytes)
			slab_bytes = block_bytes;
		nblocks = slab_bytes / block_bytes;
		slab_bytes = nblocks * block_bytes;
		slab = (merge_pool_slab_t *)xmalloc(sizeof(*slab));
		mem = (uint8 *)xmalloc(slab_bytes);
		slab->mem = mem;
		slab->next = pool->slabs;
		pool->slabs = slab;
		for (i = 0; i < nblocks; i++) {
			void *block = mem + i * block_bytes;
			*(void **)block = pool->free_list[idx];
			pool->free_list[idx] = block;
		}
	}
	p = pool->free_list[idx];
	pool->free_list[idx] = *(void **)p;
	return (uint32 *)p;
}

void merge_mem_free(merge_mem_pool_t *pool, uint32 *ptr, uint32 words) {
	uint32 idx, class_words;
	if (ptr == NULL)
		return;
	if (pool == NULL || words > MERGE_POOL_MAX_WORDS) {
		free(ptr);
		return;
	}
	idx = merge_pool_class(words, &class_words);
	(void)class_words;
	*(void **)ptr = pool->free_list[idx];
	pool->free_list[idx] = ptr;
}

uint32 *merge_mem_realloc(merge_mem_pool_t *pool, uint32 *ptr,
			uint32 old_words, uint32 new_words) {
	uint32 old_class_words, new_class_words;
	uint32 old_idx, new_idx;
	uint32 *out;
	if (ptr == NULL)
		return merge_mem_alloc(pool, new_words);
	if (new_words == 0) {
		merge_mem_free(pool, ptr, old_words);
		return NULL;
	}
	if (pool == NULL || old_words > MERGE_POOL_MAX_WORDS ||
	    new_words > MERGE_POOL_MAX_WORDS) {
		if (pool == NULL || (old_words > MERGE_POOL_MAX_WORDS &&
				new_words > MERGE_POOL_MAX_WORDS))
			return (uint32 *)xrealloc(ptr, (size_t)new_words * sizeof(uint32));
		out = merge_mem_alloc(pool, new_words);
		memcpy(out, ptr, (size_t)MIN(old_words, new_words) * sizeof(uint32));
		merge_mem_free(pool, ptr, old_words);
		return out;
	}
	old_idx = merge_pool_class(old_words, &old_class_words);
	new_idx = merge_pool_class(new_words, &new_class_words);
	if (old_idx == new_idx)
		return ptr;
	out = merge_mem_alloc(pool, new_words);
	memcpy(out, ptr, (size_t)MIN(old_words, new_words) * sizeof(uint32));
	merge_mem_free(pool, ptr, old_words);
	return out;
}

void merge_aux_init(merge_aux_t *aux, merge_mem_pool_t *data_pool) {
	memset(aux, 0, sizeof(*aux));
	aux->data_pool = data_pool;
}

void merge_aux_free(merge_aux_t *aux) {
	free(aux->tmp_relsets);
	free(aux->tmp_relset_idx);
	aux->tmp_relsets = NULL;
	aux->tmp_relset_idx = NULL;
	aux->num_relsets = 0;
	aux->relsets_alloc = 0;
}

static void merge_aux_reserve(merge_aux_t *aux, uint32 needed) {
	uint64 grown;
	uint32 new_alloc;

	if (needed <= aux->relsets_alloc)
		return;
	grown = aux->relsets_alloc ?
		(uint64)aux->relsets_alloc + MAX((uint64)64,
				(uint64)aux->relsets_alloc / 2) : 512;
	if (grown < needed)
		grown = needed;
	if (grown > UINT32_MAX)
		grown = UINT32_MAX;
	if (grown < needed ||
	    grown > SIZE_MAX / sizeof(relation_set_t) ||
	    grown > SIZE_MAX / sizeof(uint32)) {
		printf("error: merge group exceeds addressable memory\n");
		exit(-1);
	}
	new_alloc = (uint32)grown;
	aux->tmp_relsets = (relation_set_t *)xrealloc(aux->tmp_relsets,
			(size_t)new_alloc * sizeof(relation_set_t));
	aux->tmp_relset_idx = (uint32 *)xrealloc(aux->tmp_relset_idx,
			(size_t)new_alloc * sizeof(uint32));
	aux->relsets_alloc = new_alloc;
}

/*--------------------------------------------------------------------*/
void ideal_list_init(ideal_list_t *ideal_list,
		uint32 num_ideals, uint32 is_active,
		merge_mem_pool_t *data_pool) {

	uint32 i;

	ideal_list->num_ideals = num_ideals;
	ideal_list->data_pool = data_pool;
	ideal_list->list = (ideal_set_t *)xcalloc((size_t)num_ideals,
						sizeof(ideal_set_t));

	for (i = 0; i < num_ideals; i++) {
		ideal_set_t *entry = ideal_list->list + i;
		entry->active = is_active;
		entry->min_relset_size = UINT32_MAX;
		entry->next = entry;
		entry->prev = entry;
	}
}

/*--------------------------------------------------------------------*/
void ideal_list_free(ideal_list_t *ideal_list) {

	uint32 i;

	for (i = 0; i < ideal_list->num_ideals; i++) {
		ideal_set_t *entry = ideal_list->list + i;
		merge_mem_free(ideal_list->data_pool, entry->relsets,
				entry->num_relsets_alloc);
	}
	free(ideal_list->list);
}

/*--------------------------------------------------------------------*/
size_t get_merge_memuse(relation_set_t *relsets, uint32 num_relsets,
			ideal_list_t *ideal_list) {

	uint32 i;
	size_t s = num_relsets * sizeof(relation_set_t) +
		   ideal_list->num_ideals * sizeof(ideal_set_t);

	for (i = 0; i < num_relsets; i++) {
		relation_set_t *r = relsets + i;
		s += sizeof(uint32) * (r->num_large_ideals +
					r->num_relations);
	}
	for (i = 0; i < ideal_list->num_ideals; i++) {
		s += sizeof(uint32) *
			ideal_list->list[i].num_relsets_alloc;
	}
	return s;
}

/*--------------------------------------------------------------------*/
void heap_init(heap_t *heap) {

	uint32 i;

	heap->num_bins = 256;
	heap->num_ideals = 0;
	heap->next_bin = heap->num_bins;
	heap->worst_bin = (uint32)(-1);
	heap->hashtable = (ideal_set_t *)xcalloc((size_t)heap->num_bins,
						sizeof(ideal_set_t));

	for (i = 0; i < heap->num_bins; i++) {
		ideal_set_t *entry = heap->hashtable + i;
		entry->next = entry;
		entry->prev = entry;
	}
}

/*--------------------------------------------------------------------*/
void heap_free(heap_t *heap) {

	free(heap->hashtable);
}

/*--------------------------------------------------------------------*/
#define HEAP_MAX_KEY 1048575U

static uint32 heap_compute_key(ideal_set_t *ideal) {
	uint64 key;

	/* the heap is used to pick the next ideal to be merged,
	   and the key is the maximum amount of fill-in that can
	   occur when the merge takes place. The key computation
	   implements the Markowitz criterion: merging the least
	   dense matrix row (of weight c) with r-1 other rows, in
	   order to eliminate one ideal they all have in common,
	   will cause at most (r-1)*(c-1) extra nonzero entries to
	   appear in the matrix. This is a quantitative way of
	   choosing the ideal to merge that will perturb the current
	   matrix the least, and is a general case of the minimum
	   degree algorithm.

	   Using a heap to track the Markowitz value of every ideal
	   means that the next ideal to eliminate is chosen in
	   constant time, with no approximations. */

	if (ideal->num_relsets == 0)
		return (uint32)(-1);

	key = ((uint64)ideal->num_relsets - 1) *
	      ((uint64)ideal->min_relset_size - 1);
	return (uint32)MIN(key, (uint64)HEAP_MAX_KEY);
}

/*--------------------------------------------------------------------*/
void heap_add_ideal(heap_t *heap,
			ideal_list_t *ideal_list,
			uint32 ideal) {

	ideal_set_t *head;
	ideal_set_t *new_node = ideal_list->list + ideal;
	uint32 key = heap_compute_key(new_node);

	if (key == (uint32)(-1)) {
		printf("error: attempted to heapify an empty ideal\n");
		exit(-1);
	}

	/* possibly increase the number of hash buckets in the
	   heap to hold the new key value */

	if (key >= heap->num_bins) {
		uint32 i;
		ideal_set_t *ideal_array = ideal_list->list;
		uint32 new_size;
		uint64 doubled = (uint64)2 * heap->num_bins;
		uint64 requested = (uint64)key + 100;
		uint64 wanted = MAX(requested, doubled);
		if (wanted > (uint64)HEAP_MAX_KEY + 1)
			wanted = (uint64)HEAP_MAX_KEY + 1;
		new_size = (uint32)wanted;
		ideal_set_t *new_hashtable = (ideal_set_t *)xmalloc(
						new_size * sizeof(ideal_set_t));

		/* transfer chains of ideals with the same
		   key value over to the new hashtable */
		for (i = 0; i < heap->num_bins; i++) {
			ideal_set_t *old_entry = heap->hashtable + i;
			ideal_set_t *new_entry = new_hashtable + i;
			new_entry->next = new_entry;
			new_entry->prev = new_entry;
			if (old_entry->next != old_entry) {
				uint32 j;

				j = old_entry->next - ideal_array;
				new_entry->next = old_entry->next;
				ideal_array[j].prev = new_entry;

				j = old_entry->prev - ideal_array;
				new_entry->prev = old_entry->prev;
				ideal_array[j].next = new_entry;
			}
		}

		/* initialize the rest of the entries */

		for (; i < new_size; i++) {
			ideal_set_t *entry = new_hashtable + i;
			entry->next = entry;
			entry->prev = entry;
		}
		free(heap->hashtable);
		heap->hashtable = new_hashtable;
		heap->num_bins = new_size;
	}

	/* add the new ideal */

	head = heap->hashtable + key;
	new_node->prev = head;
	new_node->next = head->next;
	head->next->prev = new_node;
	head->next = new_node;

	/* adjust the best and worst heap bucket pointers */

	heap->num_ideals++;
	heap->next_bin = MIN(heap->next_bin, key);
	if (heap->worst_bin == (uint32)(-1))
		heap->worst_bin = key;
	else
		heap->worst_bin = MAX(heap->worst_bin, key);
}


/*--------------------------------------------------------------------*/
void heap_remove_ideal(heap_t *heap,
			ideal_list_t *ideal_list,
			uint32 ideal) {

	ideal_set_t *node = ideal_list->list + ideal;
	uint32 key = heap_compute_key(node);
	ideal_set_t *head;

	/* do nothing if the ideal is not connected */

	if (node->next == node)
		return;

	/* remove the ideal from the chain of ideals with
	   this key value */

	heap->num_ideals--;
	node->next->prev = node->prev;
	node->prev->next = node->next;
	node->prev = node;
	node->next = node;

	/* adjust the best and worst heap bucket pointers */

	head = heap->hashtable + heap->next_bin;
	if (head->next == head) {
		uint32 i = key;
		while (i < heap->num_bins && head->next == head) {
			head++;
			i++;
		}
		heap->next_bin = i;
	}

	head = heap->hashtable + heap->worst_bin;
	if (head->next == head) {
		uint32 i = heap->worst_bin;
		while (i > 0 && head->next == head) {
			i--;
			head--;
		}
		if (head->next == head)
			heap->worst_bin = (uint32)(-1);
		else
			heap->worst_bin = i;
	}
}

/*--------------------------------------------------------------------*/
uint32 heap_add_relset(heap_t *active_heap,
			heap_t *inactive_heap,
			ideal_list_t *ideal_list,
			relation_set_t *r,
			uint32 relset_num,
			uint32 min_ideal_weight) {

	uint32 i;
	uint32 weight = r->num_small_ideals + r->num_large_ideals;

	r->num_active_ideals = 0;

	/* add the relation set by adding each of its
	   large ideals to the heap */

	for (i = 0; i < r->num_large_ideals; i++) {
		uint32 ideal = r->data[r->num_relations + i];
		ideal_set_t *ideal_set = ideal_list->list + ideal;
		heap_t *heap = ideal_set->active ? active_heap : inactive_heap;
		uint32 prev_weight = ideal_set->min_relset_size;

		/* pull ideal i off the heap temporarily */

		heap_remove_ideal(heap, ideal_list, ideal);

		/* add the relation set to the list of relation
		   sets containing ideal i */

		if (ideal_set->num_relsets == ideal_set->num_relsets_alloc) {
			uint32 old_alloc = ideal_set->num_relsets_alloc;
			uint64 grown = old_alloc ? (uint64)old_alloc + MAX((uint64)3,
					(uint64)old_alloc / 2) : 4;
			if (grown > UINT32_MAX) {
				printf("error: ideal adjacency exceeds 32-bit capacity\n");
				exit(-1);
			}
			ideal_set->num_relsets_alloc = (uint32)grown;
			ideal_set->relsets = merge_mem_realloc(ideal_list->data_pool,
						ideal_set->relsets, old_alloc,
						ideal_set->num_relsets_alloc);
		}

		ideal_set->relsets[ideal_set->num_relsets++] = relset_num;
		ideal_set->min_relset_size = MIN(prev_weight, weight);

		/* if there are enough relation sets still
		   containing ideal i, put it back into the heap */

		if (ideal_set->num_relsets > min_ideal_weight)
			heap_add_ideal(heap, ideal_list, ideal);

		if (heap == active_heap)
			r->num_active_ideals++;
	}

	/* return the number of ideals that still
	   need merging before r can become a cycle */

	return r->num_active_ideals;
}

/*--------------------------------------------------------------------*/
void heap_remove_relset(heap_t *active_heap,
			heap_t *inactive_heap,
			ideal_list_t *ideal_list,
			relation_set_t *r,
			relation_set_t *relset_array,
			uint32 min_ideal_weight) {

	uint32 i, j;
	uint32 weight = r->num_small_ideals + r->num_large_ideals;
	uint32 relset_num = r - relset_array;

	/* remove the relation set by removing each of its
	   large ideals from the heap */

	for (i = 0; i < r->num_large_ideals; i++) {
		uint32 ideal = r->data[r->num_relations + i];
		ideal_set_t *ideal_set = ideal_list->list + ideal;
		heap_t *heap = ideal_set->active ? active_heap : inactive_heap;
		uint32 prev_weight = ideal_set->min_relset_size;

		/* pull ideal i off the heap temporarily */

		heap_remove_ideal(heap, ideal_list, ideal);

		/* remove the relation set from the list of
		   relation sets containing ideal i */

		for (j = 0; j < ideal_set->num_relsets; j++) {
			if (ideal_set->relsets[j] == relset_num)
				break;
		}
		if (j == ideal_set->num_relsets) {
			printf("error: relation set missing from ideal adjacency list\n");
			exit(-1);
		}
		ideal_set->relsets[j] = ideal_set->relsets[
					--ideal_set->num_relsets];

		if (ideal_set->num_relsets == 0) {

			/* no more relations with this ideal */

			ideal_set->min_relset_size = UINT32_MAX;
			merge_mem_free(ideal_list->data_pool, ideal_set->relsets,
					ideal_set->num_relsets_alloc);
			ideal_set->num_relsets_alloc = 0;
			ideal_set->relsets = NULL;
		}
		else {
			/* if r was the lightest relation set, we need
			   to update the minimum relation set weight for
			   ideal i */

			if (weight == prev_weight) {
				uint32 curr_weight = UINT32_MAX;
				for (j = 0; j < ideal_set->num_relsets; j++) {
					relation_set_t *r2 = relset_array +
							ideal_set->relsets[j];
					curr_weight = MIN(curr_weight,
							r2->num_small_ideals +
							r2->num_large_ideals);
				}
				ideal_set->min_relset_size = curr_weight;
			}

			/* if there are enough relation sets still
			   containing ideal i, put it back into the heap */

			if (ideal_set->num_relsets > min_ideal_weight)
				heap_add_ideal(heap, ideal_list, ideal);
		}
	}
}

/*--------------------------------------------------------------------*/
uint32 heap_remove_best(heap_t *heap, ideal_list_t *ideal_list) {

	/* remove the heap element with the smallest
	   Markowitz value */

	uint32 ideal;
	uint32 key = heap->next_bin;

	if (key == heap->num_bins)
		return (uint32)(-1);

	ideal = heap->hashtable[key].next - ideal_list->list;
	heap_remove_ideal(heap, ideal_list, ideal);
	return ideal;
}

/*--------------------------------------------------------------------*/
uint32 heap_remove_worst(heap_t *heap, ideal_list_t *ideal_list) {

	/* remove the heap element with the largest
	   Markowitz value */

	uint32 ideal;
	uint32 key = heap->worst_bin;

	if ((int32)key < 0)
		return (uint32)(-1);

	ideal = heap->hashtable[key].next - ideal_list->list;
	heap_remove_ideal(heap, ideal_list, ideal);
	return ideal;
}

/*--------------------------------------------------------------------*/
void load_next_relset_group(merge_aux_t *aux,
			heap_t *active_heap, heap_t *inactive_heap,
			ideal_list_t *ideal_list,
			relation_set_t *relset_array,
			uint32 ideal,
			uint32 min_ideal_weight) {

	uint32 i;
	ideal_set_t *ideal_set = ideal_list->list + ideal;

	/* make aux ready to receive a batch of relation sets */

	aux->num_relsets = ideal_set->num_relsets;
	merge_aux_reserve(aux, aux->num_relsets);

	/* copy each relation set containing 'ideal', and
	   remember the index into the full array of
	   relation sets where these occur. We'll need this
	   information in order to put everything back when
	   processing of aux has finished */

	for (i = 0; i < ideal_set->num_relsets; i++) {
		uint32 relset_num = ideal_set->relsets[i];
		aux->tmp_relset_idx[i] = relset_num;
		aux->tmp_relsets[i] = relset_array[relset_num];
	}

	/* now that all the relation sets are safely copied,
	   go back and remove the originals from the heap,
	   wiping out the old relation set structures in
	   the process */

	for (i = 0; i < aux->num_relsets; i++) {
		uint32 relset_num = aux->tmp_relset_idx[i];
		relation_set_t *r = relset_array + relset_num;
		heap_remove_relset(active_heap, inactive_heap,
				ideal_list, r, relset_array,
				min_ideal_weight);
		memset(r, 0, sizeof(relation_set_t));
	}
}

/*--------------------------------------------------------------------*/
void bury_inactive_ideal(relation_set_t *relset_array,
			ideal_list_t *ideal_list, uint32 ideal) {

	uint32 i, j;
	ideal_set_t *ideal_set = ideal_list->list + ideal;

	for (i = 0; i < ideal_set->num_relsets; i++) {
		relation_set_t *r = relset_array + ideal_set->relsets[i];
		uint32 num_ideals = r->num_large_ideals;
		uint32 old_words = r->num_relations + num_ideals;

		if (r->num_small_ideals == UINT16_MAX) {
			printf("error: relation-set small-ideal count exceeds 16 bits\n");
			exit(-1);
		}
		r->num_small_ideals++;
		--(r->num_large_ideals);
		{
			uint32 *r_array = r->data + r->num_relations;

			/* squeeze the ideal out of r. Because the total
			   weight of the relation set is unchanged, we
			   don't have to re-heapify any of the other
			   ideals in r */

			for (j = 0; j < num_ideals; j++) {
				if (r_array[j] == ideal)
					break;
			}
			if (j == num_ideals) {
				printf("error: inactive ideal missing from relation set\n");
				exit(-1);
			}
			for (j++; j < num_ideals; j++)
				r_array[j-1] = r_array[j];
		}

		/* Keep the pool size class synchronized with the logical
		   payload size. Otherwise a later free after crossing a size-class
		   boundary would put this block on the wrong free list. */
		r->data = merge_mem_realloc(ideal_list->data_pool, r->data,
				old_words, old_words - 1);
	}

	/* reset the ideal structure */

	merge_mem_free(ideal_list->data_pool, ideal_set->relsets,
			ideal_set->num_relsets_alloc);
	memset(ideal_set, 0, sizeof(ideal_set_t));
}

/*--------------------------------------------------------------------*/
void merge_two_relsets(relation_set_t *r1, relation_set_t *r2,
			relation_set_t *r_out, merge_aux_t *aux) {

	uint32 i;
	uint32 max_relations = r1->num_relations + r2->num_relations;
	uint32 max_ideals = r1->num_large_ideals + r2->num_large_ideals - 2;

	memset(r_out, 0, sizeof(relation_set_t));

	{
		uint32 small = (uint32)r1->num_small_ideals + r2->num_small_ideals;
		if (small > UINT16_MAX) {
			printf("error: merged small-ideal count exceeds 16 bits\n");
			exit(-1);
		}
		r_out->num_small_ideals = (uint16)small;
	}

	/* combine the list of relations in r1 and r2
	   using a merge operation */

	if (max_relations >= MERGE_MAX_OBJECTS) {
		printf("error: relation list too large\n");
		exit(-1);
	}

	i = merge_relations(aux->tmp_relations,
				r1->data, r1->num_relations,
				r2->data, r2->num_relations);
	if (i == 0) {
		memset(r_out, 0, sizeof(relation_set_t));
		return;
	}
	r_out->num_relations = i;

	/* merge the ideal lists of r1 and r2 */

	if (max_ideals >= MERGE_MAX_OBJECTS) {
		printf("error: list of merged ideals too large\n");
		exit(-1);
	}

	i = merge_relations(aux->tmp_ideals,
				r1->data + r1->num_relations,
				r1->num_large_ideals,
				r2->data + r2->num_relations,
				r2->num_large_ideals);

	r_out->num_large_ideals = i;

	/* save the merged lists */

	r_out->data = merge_mem_alloc(aux->data_pool,
				r_out->num_relations + r_out->num_large_ideals);
	memcpy(r_out->data,
	       aux->tmp_relations,
	       r_out->num_relations * sizeof(uint32));
	memcpy(r_out->data + r_out->num_relations,
	       aux->tmp_ideals,
	       r_out->num_large_ideals * sizeof(uint32));
}

/*--------------------------------------------------------------------*/
uint32 estimate_new_weight(relation_set_t *r1,
			relation_set_t *r2) {

	/* guess the total number of ideals derived from
	   merging r1 and r2. This is much faster than
	   actually merging r1 and r2, but must occur much
	   more often */

	uint32 i, j, k;
	uint32 num_small;
	uint32 num_ideals1 = r1->num_large_ideals;
	uint32 num_ideals2 = r2->num_large_ideals;
	uint32 *ilist1 = r1->data + r1->num_relations;
	uint32 *ilist2 = r2->data + r2->num_relations;

	/* count the number of large ideals that would
	   be left if r1 and r2 were merged */

	i = j = k = 0;
	while (i < num_ideals1 && j < num_ideals2) {
		uint32 ideal1 = ilist1[i];
		uint32 ideal2 = ilist2[j];
		if (ideal1 < ideal2) {
			i++; k++;
		}
		else if (ideal1 > ideal2) {
			j++; k++;
		}
		else {
			i++; j++;
		}
	}

	num_small = r1->num_small_ideals + r2->num_small_ideals;
	return k + num_small + (num_ideals1 - i) + (num_ideals2 - j);
}

