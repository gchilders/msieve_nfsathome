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
static int compare_uint32(const void *x, const void *y) {
	uint32 *xx = (uint32 *)x;
	uint32 *yy = (uint32 *)y;
	if (*xx > *yy)
		return 1;
	if (*xx < *yy)
		return -1;
	return 0;
}

/*--------------------------------------------------------------------*/
#define NUM_IDEAL_BINS 6

#define MERGE_RENUMBER_BLOCK_SIZE ((uint64)1 << 20)

static uint32 renumber_ideal_map(ideal_map_t *ideal_map, uint32 num_ideals) {
	uint32 num_blocks;
	uint64 *block_offsets;
	uint64 total = 0;
	int b;

	if (num_ideals == 0)
		return 0;

	num_blocks = (uint32)(((uint64)num_ideals +
			MERGE_RENUMBER_BLOCK_SIZE - 1) / MERGE_RENUMBER_BLOCK_SIZE);
	block_offsets = (uint64 *)xmalloc((size_t)num_blocks * sizeof(uint64));

#pragma omp parallel for schedule(static)
	for (b = 0; b < (int)num_blocks; b++) {
		uint64 k;
		uint64 start = (uint64)b * MERGE_RENUMBER_BLOCK_SIZE;
		uint64 end = MIN(start + MERGE_RENUMBER_BLOCK_SIZE,
				(uint64)num_ideals);
		uint64 local_count = 0;
		for (k = start; k < end; k++) {
			if (ideal_map_payload(ideal_map + (size_t)k))
				local_count++;
		}
		block_offsets[b] = local_count;
	}

	for (b = 0; b < (int)num_blocks; b++) {
		uint64 block_count = block_offsets[b];
		block_offsets[b] = total;
		total += block_count;
	}
	if (total > UINT32_MAX) {
		printf("error: merge ideal renumbering exceeds 32-bit capacity\n");
		exit(-1);
	}

#pragma omp parallel for schedule(static)
	for (b = 0; b < (int)num_blocks; b++) {
		uint64 k;
		uint64 start = (uint64)b * MERGE_RENUMBER_BLOCK_SIZE;
		uint64 end = MIN(start + MERGE_RENUMBER_BLOCK_SIZE,
				(uint64)num_ideals);
		uint32 next = (uint32)block_offsets[b];
		for (k = start; k < end; k++) {
			ideal_map_t *entry = ideal_map + (size_t)k;
			if (ideal_map_payload(entry))
				ideal_map_set_payload(entry, next++);
		}
	}

	free(block_offsets);
	return (uint32)total;
}

void filter_merge_init(msieve_obj *obj, filter_t *filter) {

	/* start the merge process. Right now this only prints
	   histograms of the number of ideals per relation, and
	   sorts the ideals of each relation into ascending order */

	uint32 i;
	uint32 num_relations = filter->num_relations;
	uint32 bin0 = 0, bin1 = 0, bin2 = 0, bin3 = 0;
	uint32 bin4 = 0, bin5 = 0, bin6 = 0, bin7plus = 0;
	uint32 ideal_count_bins[NUM_IDEAL_BINS+2];

	/* relation_ptr[] makes every packed relation independently addressable.
	   Sorting relation-local ideal lists is therefore embarrassingly parallel;
	   scalar reductions keep the tiny histogram portable to OpenMP 2.x. */
#pragma omp parallel for schedule(static) \
		reduction(+:bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7plus)
	for (i = 0; i < num_relations; i++) {
		relation_ideal_t *curr_relation = filter->relation_ptr[i];
		uint32 num_ideals = curr_relation->ideal_count;

		switch (num_ideals) {
		case 0: bin0++; break;
		case 1: bin1++; break;
		case 2: bin2++; break;
		case 3: bin3++; break;
		case 4: bin4++; break;
		case 5: bin5++; break;
		case 6: bin6++; break;
		default: bin7plus++; break;
		}

		if (num_ideals > 1) {
			qsort(curr_relation->ideal_list, (size_t)num_ideals,
					sizeof(uint32), compare_uint32);
		}
	}

	ideal_count_bins[0] = bin0;
	ideal_count_bins[1] = bin1;
	ideal_count_bins[2] = bin2;
	ideal_count_bins[3] = bin3;
	ideal_count_bins[4] = bin4;
	ideal_count_bins[5] = bin5;
	ideal_count_bins[6] = bin6;
	ideal_count_bins[7] = bin7plus;

	for (i = 0; i < NUM_IDEAL_BINS+1; i++) {
		logprintf(obj, "relations with %u large ideals: %u\n",
					i, ideal_count_bins[i]);
	}
	logprintf(obj, "relations with %u+ large ideals: %u\n",
				i, ideal_count_bins[i]);
}

/*--------------------------------------------------------------------*/
#define MAX_2WAY_RELATIONS 200
#define MAX_2WAY_IDEALS 1000

void filter_merge_2way(msieve_obj *obj, filter_t *filter,
			merge_t *merge) {

	/* performs 2-way merges and initializes the structures
	   needed by the rest of the merge code. 2-way merges are
	   handled separately because there are a lot of them to do
	   (typically 30% of the ideals participate in 2-way merges)
	   and they can be performed very efficiently.

	   We can perform a 2-way merge by locating all the 2-way
	   cliques in the input dataset, then collapsing each clique
	   into a relation set. A relation belongs to at most one
	   clique, and no optimization is possible in the combining
	   process */

	uint32 i, j;
	ideal_map_t *ideal_map;
	relation_ideal_t *relation_array;
	relation_ideal_t **relation_ptr;
	relation_ideal_t *curr_relation;
	uint32 num_relations;
	uint32 num_ideals;
	uint32 num_deleted;

	ideal_relation_list_t reverse_list;

	relation_set_t *relset_array;
	uint32 num_relset;
	size_t num_relset_alloc;

	logprintf(obj, "commencing 2-way merge\n");
	if (merge->data_pool == NULL)
		merge->data_pool = merge_mem_pool_create();

	/* set up the hashtable for ideal counts */

	relation_array = filter->relation_array;
	relation_ptr = filter->relation_ptr;
	num_relations = filter->num_relations;
	num_ideals = filter->num_ideals;
	ideal_map = (ideal_map_t *)xcalloc((size_t)num_ideals,
					sizeof(ideal_map_t));

	/* set up structure for linked lists of clique relations */

	ideal_relation_list_init(&reverse_list);

	/* count the number of times each ideal occurs in relations */

#pragma omp parallel for private(j, curr_relation)
	for (i = 0; i < num_relations; i++) {
		curr_relation = relation_ptr[i];
		curr_relation->connected = 0;
		for (j = 0; j < curr_relation->ideal_count; j++) {
			uint32 ideal = curr_relation->ideal_list[j];
#pragma omp atomic update
			ideal_map[ideal].data++;
		}
	}

	/* mark all the ideals with weight 2 as belonging
	   to a clique, and set the head of their linked
	   list of relations to empty */

#pragma omp parallel for
	for (i = 0; i < num_ideals; i++) {
		if (ideal_map_payload(ideal_map + i) == 2) {
			ideal_map[i].data = IDEAL_MAP_CLIQUE;
		}
	}

	/* for each relation */

// #pragma omp parallel for private(j, curr_relation)
	for (i = 0; i < num_relations; i++) {
		curr_relation = relation_ptr[i];
		/* for each ideal in the relation */

		for (j = 0; j < curr_relation->ideal_count; j++) {
			uint32 ideal = curr_relation->ideal_list[j];

			if (!ideal_map_is_clique(ideal_map + ideal))
				continue;

			/* relation belongs in a clique because of this
			   ideal; add it to the ideal's linked list */
// #pragma omp critical (add_relation)
			{
				uint64 relation_array_word = (uint64)(
					(uint32 *)curr_relation - (uint32 *)relation_array);
				ideal_map_set_payload(ideal_map + ideal,
				ideal_relation_list_append(&reverse_list, relation_array_word,
					ideal_map_payload(ideal_map + ideal)));
			}
		}
	}

	num_relset = 0;
	num_relset_alloc = 10000;
	relset_array = (relation_set_t *)xmalloc(num_relset_alloc *
					sizeof(relation_set_t));

	/* find all the cliques and convert to relation sets.
	   We perform breadth first search by iterating through
	   all of the relations.

	   Note that the clique removal step iterated through
	   ideals looking for cliques. It could do that because
	   the only objective was to find cliques. Here the objective
	   is different: find all relations, performing extra work
	   on cliques. If we discovered relations through the ideals
	   they contained, we'd have to do a lot more traversing
	   of linked lists */

	curr_relation = relation_array;
	num_deleted = 0;
	for (i = 0; i < num_relations;
		i++, curr_relation = next_relation_ptr(curr_relation)) {

		uint32 tmp_relations[MAX_2WAY_RELATIONS];
		uint32 tmp_ideals[MAX_2WAY_IDEALS];
		uint32 accum_ideals[MAX_2WAY_IDEALS];
		uint32 num_tmp_relation = 0;
		uint32 num_tmp_ideal = 0;
		uint32 num_small_ideal;
		relation_set_t *curr_relset;

		if (curr_relation->connected)
			continue;

		/* relation hasn't been seen before. Start counting
		   the ideals in it, and list the large ideals that
		   the relation contains. The clique is complete when
		   all the ideals in the list have been processed */

		tmp_relations[num_tmp_relation++] = curr_relation->rel_index;
		num_small_ideal = curr_relation->gf2_factors;
		num_tmp_ideal = curr_relation->ideal_count;
		for (j = 0; j < num_tmp_ideal; j++)
			tmp_ideals[j] = curr_relation->ideal_list[j];
		curr_relation->connected = 1;

		/* for each ideal in the combined ideal list */

		for (j = 0; j < num_tmp_ideal; j++) {
			uint32 ideal = tmp_ideals[j];
			uint64 offset;

			/* check if the ideal is not part of a clique */

			if (!ideal_map_is_clique(ideal_map + ideal))
				continue;

			/* we've found a clique, and have to find all the
			   relations in it */

			offset = ideal_map_payload(ideal_map + ideal);
			while (offset) {

				ideal_relation_t *rev = ideal_relation_list_at_fast(&reverse_list, offset);
				relation_ideal_t *r = (relation_ideal_t *)
						((uint32 *)relation_array +
						ideal_relation_word(rev));
				if (r->connected) {
					offset = ideal_relation_next(rev);
					continue;
				}

				/* relation seen for the first time;
				   add its count of ideals to the totals for
				   the current clique, and merge its ideal
				   list with that of the current clique. The
				   relation is now considered processed */

				/* check for overflow */

				if (num_tmp_ideal + r->ideal_count >=
							MAX_2WAY_IDEALS) {
					printf("error: clique merge requires "
						"too many ideals\n");
					exit(-1);
				}

				if (num_tmp_relation == MAX_2WAY_RELATIONS) {
					printf("error: clique merge requires "
						"too many relations\n");
					exit(-1);
				}

				/* perform the merge */

				tmp_relations[num_tmp_relation++] =
							r->rel_index;
				num_small_ideal += r->gf2_factors;
				num_tmp_ideal = merge_relations(accum_ideals,
						tmp_ideals, num_tmp_ideal,
						r->ideal_list, r->ideal_count);
				memcpy(tmp_ideals, accum_ideals,
						num_tmp_ideal * sizeof(uint32));
				r->connected = 1;
				offset = ideal_relation_next(rev);
			}

			/* the ideals were merged into the existing list,
			   so that new ideals to examine can be anywhere in
			   tmp_ideals. Thus we have to start the loop for
			   examining ideals all over again. We have to
			   restart the loop once for every relation in the
			   clique, but this isn't a big deal because most
			   cliques have only two relations */

			j = (uint32)(-1);
		}

		/* clique is complete; throw it away if it
		   contains too many relations */

		if (num_tmp_relation >= MAX_RELSET_SIZE) {
			num_deleted++;
			continue;
		}

		/* allocate a relation set for the clique */

		if ((size_t)num_relset == num_relset_alloc) {
			size_t max_alloc = (size_t)UINT32_MAX;
			size_t increment = MAX((size_t)1, num_relset_alloc / 4);
			size_t new_alloc;
			if (num_relset_alloc >= max_alloc) {
				logprintf(obj, "error: relation-set count exceeds 32-bit capacity\n");
				exit(-1);
			}
			new_alloc = num_relset_alloc > max_alloc - increment ?
				max_alloc : num_relset_alloc + increment;
			if (new_alloc > (size_t)-1 / sizeof(relation_set_t)) {
				logprintf(obj, "error: relation-set allocation exceeds address space\n");
				exit(-1);
			}
			num_relset_alloc = new_alloc;
			relset_array = (relation_set_t *)xrealloc(relset_array,
						num_relset_alloc * sizeof(relation_set_t));
		}
		curr_relset = relset_array + num_relset;
		curr_relset->num_relations = num_tmp_relation;
		curr_relset->num_small_ideals = num_small_ideal;
		curr_relset->num_large_ideals = num_tmp_ideal;

		/* sort the relation numbers in ascending order
		   (the ideal list is already ordered), then store */

		curr_relset->data = merge_relset_alloc(merge->data_pool,
					curr_relset, num_tmp_relation + num_tmp_ideal);
		if (num_tmp_relation > 1) {
			qsort(tmp_relations, (size_t)num_tmp_relation,
					sizeof(uint32), compare_uint32);
		}
		memcpy(curr_relset->data, tmp_relations,
				num_tmp_relation * sizeof(uint32));
		memcpy(curr_relset->data + num_tmp_relation, tmp_ideals,
				num_tmp_ideal * sizeof(uint32));
		num_relset++;
	}

	/* free unneeded objects */

	ideal_relation_list_free(&reverse_list);
	free(filter->relation_array);
	filter->relation_array = NULL;
	free(filter->relation_ptr);
	filter->relation_ptr = NULL;
	relset_array = (relation_set_t *)xrealloc(relset_array,
				num_relset * sizeof(relation_set_t));

	/* renumber the ideals to skip ideals that have been
	   completely merged */

	memset(ideal_map, 0, num_ideals * sizeof(ideal_map_t));
#pragma omp parallel for private(j)
	for (i = 0; i < num_relset; i++) {
		relation_set_t *r = relset_array + i;
		uint32 *ideal_list = r->data + r->num_relations;
		for (j = 0; j < r->num_large_ideals; j++) {
#pragma omp atomic update
			ideal_map[ideal_list[j]].data++;
		}
	}
	num_ideals = renumber_ideal_map(ideal_map, num_ideals);
#pragma omp parallel for private(j)
	for (i = 0; i < num_relset; i++) {
		relation_set_t *r = relset_array + i;
		uint32 *ideal_list = r->data + r->num_relations;
		for (j = 0; j < r->num_large_ideals; j++) {
			ideal_list[j] = (uint32)ideal_map_payload(
					ideal_map + ideal_list[j]);
		}
	}

	logprintf(obj, "reduce to %u relation sets and %u "
			"unique ideals\n", num_relset, num_ideals);
	if (num_deleted) {
		logprintf(obj, "ignored %u oversize "
				"relation sets\n", num_deleted);
	}
	merge->relset_array = relset_array;
	merge->num_relsets = num_relset;
	merge->num_ideals = num_ideals;
	free(ideal_map);
}
