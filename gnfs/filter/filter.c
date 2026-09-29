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

#include "filter.h"
#include "rmap.h"

/*--------------------------------------------------------------------*/
static void find_fb_size(factor_base_t *fb,
			uint32 limit_r, uint32 limit_a,
			uint32 *entries_r_out, uint32 *entries_a_out) {

	prime_sieve_t prime_sieve;
	uint32 entries_r = 0;
	uint32 entries_a = 0;

	/* If the filtering bounds are extremely large, just
	   estimate the target as the number of primes less than
	   the filtering bounds */

	if (limit_r > 20000000 && limit_a > 20000000) {
		*entries_r_out = 1.02 * limit_r / (log((double)limit_r) - 1);
		*entries_a_out = 1.02 * limit_a / (log((double)limit_a) - 1);
		return;
	}

	init_prime_sieve(&prime_sieve, 0,
			MAX(limit_r, limit_a) + 1000);

	while (1) {
		uint32 p = get_next_prime(&prime_sieve);
		uint32 num_roots;
		uint32 high_coeff;
		uint32 roots[MAX_POLY_DEGREE + 1];

		if (p >= limit_r && p >= limit_a)
			break;

		if (p < limit_r) {
			num_roots = poly_get_zeros(roots, &fb->rfb.poly, p,
							&high_coeff, 1);
			if (high_coeff == 0)
				num_roots++;
			entries_r += num_roots;
		}

		if (p < limit_a) {
			num_roots = poly_get_zeros(roots, &fb->afb.poly, p,
							&high_coeff, 1);
			if (high_coeff == 0)
				num_roots++;
			entries_a += num_roots;
		}
	}

	free_prime_sieve(&prime_sieve);
	*entries_r_out = entries_r;
	*entries_a_out = entries_a;
}

/*--------------------------------------------------------------------*/
static uint32 check_excess(filter_t *filter) {

	/* give up if there is not enough excess
	   to form a matrix */

	uint32 relations_needed = 0;

	if (filter->num_relations < filter->num_ideals ||
	    filter->num_relations - filter->num_ideals <
	    				filter->target_excess) {
		uint64 needed64 = 1000000;
		if (filter->num_relations > filter->num_ideals) {
			uint64 excess = (uint64)filter->num_relations - filter->num_ideals;
			uint64 deficit = (uint64)filter->target_excess - excess;
			needed64 = 3 * deficit;
			if (needed64 < 1000000)
				needed64 = 1000000;
		}
		relations_needed = needed64 > UINT32_MAX ? UINT32_MAX : (uint32)needed64;
		free(filter->relation_array);
		free(filter->relation_ptr);
		filter->relation_array = NULL;
	}
	return relations_needed;
}

/*--------------------------------------------------------------------*/
static void dump_relation_numbers(msieve_obj *obj, filter_t *filter) {

	uint32 i;
	char buf[256];
	FILE *relation_fp;
	nfs_rmap_reader_t map;
	uint32 have_map = 0;
	relation_ideal_t *r = filter->relation_array;

	if (filter->lp_needs_rmap) {
		if (nfs_rmap_reader_open(obj, &map, 1, UINT64_MAX) != 0) {
			logprintf(obj, "error: this filtered dataset requires a valid committed relation map\n");
			exit(-1);
		}
		have_map = 1;
	}

	get_filter_tmp_name(obj, buf, sizeof(buf), ".d");
	relation_fp = fopen(buf, "wb");
	if (relation_fp == NULL) {
		logprintf(obj, "error: reldump can't open out file\n");
		exit(-1);
	}

	for (i = 0; i < filter->num_relations; i++) {
		uint64 source_id = have_map ? nfs_rmap_get(obj, &map, r->rel_index) :
			(uint64)r->rel_index;
		if (fwrite(&source_id, sizeof(uint64), 1, relation_fp) != 1) {
			logprintf(obj, "error: write failed creating relation keep-list\n");
			exit(-1);
		}
		r = next_relation_ptr(r);
	}
	if (fflush(relation_fp) != 0 || ferror(relation_fp) ||
	    fclose(relation_fp) != 0) {
		logprintf(obj, "error: write failed finalizing relation keep-list\n");
		exit(-1);
	}
	if (have_map)
		nfs_rmap_reader_close(&map);
}

/*--------------------------------------------------------------------*/
static void set_filtering_bounds(msieve_obj *obj, factor_base_t *fb,
			uint32 filtmin_r, uint32 filtmin_a,
			uint32 *entries_r_out, uint32 *entries_a_out,
			uint64 num_relations, uint32 force_small,
			filter_t *filter) {

	uint32 entries_r, entries_a;

	if (force_small) {
		if (num_relations < 2000000)
			filtmin_r = filtmin_a = 30000;
		else if (num_relations < 10000000) {
			filtmin_r = MIN(filtmin_r / 2, 100000);
			filtmin_a = MIN(filtmin_a / 2, 100000);
		}
		else {
			filtmin_r = MIN(filtmin_r / 2, 720000);
			filtmin_a = MIN(filtmin_a / 2, 720000);
		}
	}

	logprintf(obj, "reading ideals above %u\n", filtmin_r);
	find_fb_size(fb, filtmin_r, filtmin_a, &entries_r, &entries_a);
	filter->filtmin_r = filtmin_r;
	filter->filtmin_a = filtmin_a;
	{
		uint64 target = (uint64)entries_r + entries_a;
		if (target > UINT32_MAX) {
			logprintf(obj, "error: filtering target excess exceeds 32-bit common-filter capacity\n");
			exit(-1);
		}
		filter->target_excess = (uint32)target;
	}

	*entries_r_out = entries_r;
	*entries_a_out = entries_a;
}

/*--------------------------------------------------------------------*/
/* the multiple of the amount of excess needed for
   merging to proceed */

#define FINAL_EXCESS_FRACTION 1.16

/* the default expected number of sparse nonzeros in the
   average matrix column (may be overriden if you know
   what you are doing) */

#define DEFAULT_TARGET_DENSITY 90.0

static uint32 do_merge(msieve_obj *obj, filter_t *filter,
			merge_t *merge, double target_density,
			const char *ckpt_path, uint32 commit_ckpt) {

	uint32 relations_needed;
	uint32 extra_needed = filter->target_excess;

	/* make the clique removal more conservative by leaving
	   some of the excess; this makes the merge phase easier.
	   Note that the singleton removal probably threw away
	   many large ideals that occur too often to be worth
	   tracking, which forces the target matrix size to
	   increase, so that target_excess is larger now */

	{
		double target = filter->target_excess * FINAL_EXCESS_FRACTION;
		if (target > UINT32_MAX) {
			logprintf(obj, "error: adjusted filtering target exceeds 32-bit capacity\n");
			return UINT32_MAX;
		}
		filter->target_excess = (uint32)target;
	}

	if ((relations_needed = check_excess(filter)) > 0)
		return relations_needed;

	/* build the matrix */

	merge->num_extra_relations = NUM_EXTRA_RELATIONS;

	merge->target_density = DEFAULT_TARGET_DENSITY;
	if (target_density != 0)
		merge->target_density = target_density;

	if (filter_make_relsets(obj, filter, merge, extra_needed,
				ckpt_path) != 0) {
		if (merge->relset_array != NULL || merge->data_pool != NULL)
			filter_free_relsets(merge);
		filter_merge_checkpoint_commit(obj, ckpt_path, 0);
		return 1000000;
	}

	/* a caller that may still reject this merge commits it itself */

	if (commit_ckpt)
		filter_merge_checkpoint_commit(obj, ckpt_path, 1);
	return 0;
}

/*--------------------------------------------------------------------*/
#define MAX_KEEP_WEIGHT 45

static uint32 do_partial_filtering(msieve_obj *obj, filter_t *filter,
				merge_t *merge, uint32 entries_r,
				uint32 entries_a, double target_density,
				uint32 max_weight, const char *ckpt_path) {

	uint32 relations_needed;
	uint32 num_relations = filter->num_relations;
	uint32 num_ideals = filter->num_ideals;

	if (filter->num_relations < 20000000 && max_weight < 25) {
		logprintf(obj, "raising initial max weight because there are few relations\n");
		max_weight = 25;
	}

	for (; max_weight < MAX_KEEP_WEIGHT; max_weight += 5) {

		filter->target_excess = entries_r + entries_a;
		filter->num_relations = num_relations;
		filter->num_ideals = num_ideals;

		filter_read_lp_file(obj, filter, max_weight);

		if ((relations_needed = do_merge(obj, filter,
						merge, target_density,
						ckpt_path, 0)) > 0) {
			/* a rejected earlier attempt may have left one */
			filter_merge_checkpoint_commit(obj, ckpt_path, 0);
			return relations_needed;
		}

		/* accept the collection of generated cycles
		   if the matrix they form is dense enough or
		   max_weight has been incremented enough */

		if (merge->avg_cycle_weight > 63.0 ||
		    max_weight >= MAX_KEEP_WEIGHT - 5) {
			filter_merge_checkpoint_commit(obj, ckpt_path, 1);
			break;
		}

		logprintf(obj, "matrix not dense enough, retrying\n");
		filter_free_relsets(merge);
	}

	return 0;
}

/*--------------------------------------------------------------------*/
uint32 nfs_filter_relations(msieve_obj *obj, mpz_t n) {

	filter_t filter;
	merge_t merge;
	uint32 filtmin_r, filtmin_a;
	uint32 entries_r, entries_a;
	uint64 num_relations;
	uint32 relations_needed = 0;
	factor_base_t fb;
	time_t wall_time = time(NULL);
	uint64 savefile_size = get_file_size(obj->savefile.name);
	uint64 ram_size = 0;
	char ckpt_buf[256];
	const char *ckpt_path = NULL;
	uint64 max_relations = 0;
	uint32 filter_bound = 0;
	double target_density = 0;
	double target_densities[16];
	uint32 num_densities = 0;
	uint32 max_weight = 20;
	char lp_filename[256];

	logprintf(obj, "\n");
	logprintf(obj, "commencing relation filtering\n");
	savefile_check_scratch(obj);

	/* parse arguments */

	if (obj->nfs_args != NULL) {

		const char *tmp;

		/* merge_ckpt=<path> records the relation sets just before the
		   full merge, or restarts from that file when it exists, so the
		   merge can be iterated on without repeating the hours of
		   deterministic work in front of it */

		tmp = strstr(obj->nfs_args, "merge_ckpt=");
		if (tmp != NULL) {
			size_t k = 0;

			tmp += 11;
			while (*tmp && *tmp != ',' && !isspace((int)(unsigned char)*tmp) &&
					k < sizeof(ckpt_buf) - 1)
				ckpt_buf[k++] = *tmp++;
			ckpt_buf[k] = 0;
			if (k > 0)
				ckpt_path = ckpt_buf;
		}

		tmp = strstr(obj->nfs_args, "filter_mem_mb=");
		if (tmp != NULL) {
			ram_size = strtoull(tmp + 14, NULL, 10) << 20;
			logprintf(obj, "setting memory use to %.1f MB\n",
					(double)ram_size / 1048576);
		}

		tmp = strstr(obj->nfs_args, "filter_maxrels=");
		if (tmp != NULL) {
			max_relations = strtoull(tmp + 15, NULL, 10);
			logprintf(obj, "setting max relations to %" PRIu64 "\n",
					max_relations);
		}

		tmp = strstr(obj->nfs_args, "filter_lpbound=");
		if (tmp != NULL) {
			filter_bound = strtoul(tmp + 15, NULL, 10);
			logprintf(obj, "setting large prime bound to %u\n",
					filter_bound);
		}

		tmp = strstr(obj->nfs_args, "target_density=");
		if (tmp != NULL) {
			const char *p = tmp + 15;
			char *endptr;
			uint32 di, dj;
			while (num_densities < 16) {
				double d = strtod(p, &endptr);
				if (endptr == p) break;
				target_densities[num_densities++] = d;
				if (*endptr != ',') break;
				p = endptr + 1;
			}
			/* sort numerically (insertion sort) */
			for (di = 1; di < num_densities; di++) {
				double key = target_densities[di];
				dj = di;
				while (dj > 0 && target_densities[dj-1] > key) {
					target_densities[dj] = target_densities[dj-1];
					dj--;
				}
				target_densities[dj] = key;
			}
			if (num_densities == 1) {
				target_density = target_densities[0];
				logprintf(obj, "setting target matrix density to %.1f\n",
						target_density);
			} else {
				logprintf(obj, "setting %u target densities:", num_densities);
				for (di = 0; di < num_densities; di++)
					logprintf(obj, " %.0f", target_densities[di]);
				logprintf(obj, "\n");
			}
		}

		/* the checkpoint holds the relation sets of one merge; the
		   multi-density paths run their own merges per density and
		   write .cyc.NNN files, which it can neither record nor
		   restart. Refuse the pair rather than write a plain .cyc
		   that all_matbuild would never read */

		if (ckpt_path != NULL && num_densities > 1) {
			logprintf(obj, "error: merge_ckpt works with a single "
					"target_density, not a list\n");
			exit(-1);
		}

		tmp = strstr(obj->nfs_args, "max_weight=");
		if (tmp != NULL) {
			max_weight = strtoul(tmp + 11, NULL, 10);
			if (max_weight < MAX_KEEP_WEIGHT) {
				logprintf(obj, "setting initial max weight to %u\n",
						max_weight);
			}
			else {
#define str(s) #s
#define xstr(s) str(s)
				logprintf(obj, "initial max weight must be <= " xstr(MAX_KEEP_WEIGHT) "\n");
#undef xstr
#undef str
				exit(-1);
			}
		}

		/* old-style 'X,Y' format */

		tmp = strchr(obj->nfs_args, ',');
		if (tmp != NULL) {
			const char *tmp0 = tmp - 1;
			while (tmp0 > obj->nfs_args && isdigit(tmp0[-1]))
				tmp0--;
			/* skip if this comma is inside a target_density= value */
			if (tmp0 == obj->nfs_args || tmp0[-1] != '=') {
				max_relations = strtoull(tmp + 1, NULL, 10);
				filter_bound = strtoul(tmp0, NULL, 10);

				logprintf(obj, "setting max relations to %" PRIu64 "\n",
						max_relations);
				logprintf(obj, "setting large prime bound to %u\n",
						filter_bound);
			}
		}
	}

	if (ram_size == 0)
		ram_size = get_ram_size();

	memset(&filter, 0, sizeof(filter));
	memset(&merge, 0, sizeof(merge));
	memset(&fb, 0, sizeof(fb));
	mpz_poly_init(&fb.rfb.poly);
	mpz_poly_init(&fb.afb.poly);
	if (read_poly(obj, n, &fb.rfb.poly, &fb.afb.poly, NULL)) {
		printf("filtering failed to read polynomials\n");
		exit(-1);
	}
	logprintf(obj, "estimated available RAM is %.1lf MB\n",
				(double)ram_size / 1048576);

	/* a checkpoint makes everything below redundant: it already holds
	   the relation sets the full merge starts from */

	if (ckpt_path != NULL) {
		uint32 ckpt_min_cycles = 0;

		if (filter_merge_checkpoint_load(obj, &merge,
				&ckpt_min_cycles, ckpt_path) == 0) {
			if (target_density != 0)
				merge.target_density = target_density;
			if (filter_merge_full(obj, &merge,
					ckpt_min_cycles) != 0) {
				char failed[300];

				/* reloading it would fail the same way on
				   every later run, even after more relations
				   are sieved; move it aside so the next run
				   filters from the relations again */

				filter_free_relsets(&merge);
				if (snprintf(failed, sizeof(failed), "%s.failed",
						ckpt_path) < (int)sizeof(failed) &&
				    rename(ckpt_path, failed) == 0)
					logprintf(obj, "merge from checkpoint "
						"failed; moved it to '%s'\n",
						failed);
				relations_needed = 1000000;
				goto finished;
			}
			goto merge_done;
		}
	}

	/* with a scratch directory configured, work from a local
	   decompressed copy of the savefile */

	savefile_stage(obj);

	/* delete duplicate relations */

	filtmin_r = filtmin_a = nfs_purge_duplicates(obj, &fb,
					max_relations, ram_size,
					&num_relations);
	if (filter_bound > 0)
		filtmin_r = filtmin_a = filter_bound;

	/* set up the first disk-based pass; if the dataset is
	   "small", this will be the only such pass */

	set_filtering_bounds(obj, &fb, filtmin_r, filtmin_a,
				&entries_r, &entries_a, num_relations,
				(uint32)(savefile_size < ram_size / 2),
				&filter);

	/* separate out the large ideals and delete singletons
	   once they are all in memory. If the dataset is large,
	   first delete most of the singletons from the disk file */

	nfs_write_lp_file(obj, &fb, &filter, max_relations, 0);
	nfs_compact_lp_file(obj, &filter, ram_size);
	/* save filter state before initial LP read for multi-density small path */
	{
	filter_read_lp_file(obj, &filter, 0);

	if (savefile_size < ram_size / 2) {

		/* dataset is "small"; build the matrix immediately.
		   Depending on how much memory the machine has, really
		   big datasets may get to do this */

		if (num_densities <= 1) {
			if ((relations_needed = do_merge(obj, &filter,
							&merge, target_density,
							ckpt_path, 1)) > 0)
				goto finished;
		}
		else {
			uint32 d;
			char dsuffix[32];
			uint8 *saved_rel_array;
			size_t saved_rel_bytes;
			uint32 saved_nr, saved_ni;
			uint32 extra_needed = filter.target_excess;

			/* run cliques once for all densities */
			filter.target_excess = (uint32)(filter.target_excess *
							FINAL_EXCESS_FRACTION);
			if ((relations_needed = check_excess(&filter)) > 0)
				goto finished;
			filter_purge_cliques(obj, &filter);

			/* save post-clique state */
			saved_nr = filter.num_relations;
			saved_ni = filter.num_ideals;
			{
				relation_ideal_t *rp = filter.relation_array;
				uint32 k;
				for (k = 0; k < saved_nr; k++)
					rp = next_relation_ptr(rp);
				saved_rel_bytes = (uint8 *)rp - (uint8 *)filter.relation_array;
			}
			saved_rel_array = (uint8 *)xmalloc(saved_rel_bytes);
			memcpy(saved_rel_array, filter.relation_array, saved_rel_bytes);
			for (d = 0; d < num_densities; d++) {
				if (d > 0) {
					free(filter.relation_array);
					free(filter.relation_ptr);
					filter.relation_array = (relation_ideal_t *)xmalloc(saved_rel_bytes);
					memcpy(filter.relation_array, saved_rel_array, saved_rel_bytes);
					filter.relation_ptr = (relation_ideal_t **)xmalloc(
								saved_nr * sizeof(relation_ideal_t *));
					{
						relation_ideal_t *rp = filter.relation_array;
						uint32 k;
						for (k = 0; k < saved_nr; k++) {
							filter.relation_ptr[k] = rp;
							rp = next_relation_ptr(rp);
						}
					}
					filter.num_relations = saved_nr;
					filter.num_ideals    = saved_ni;
				}
				logprintf(obj, "trying target density %.0f\n",
						target_densities[d]);
				memset(&merge, 0, sizeof(merge));
				merge.num_extra_relations = NUM_EXTRA_RELATIONS;
				merge.target_density = target_densities[d];
				filter_merge_init(obj, &filter);
				filter_merge_2way(obj, &filter, &merge);
				if (filter_merge_full(obj, &merge, extra_needed) != 0) {

					/* do_merge() frees the half-built relsets before it
					   returns; these loops have to do it themselves,
					   and the finished: label does not */

					if (merge.relset_array != NULL || merge.data_pool != NULL)
						filter_free_relsets(&merge);
					if (d == 0) {
						relations_needed = 1000000;
						free(saved_rel_array);
						goto finished;
					}
					break;
				}
				filter_postproc_relsets(obj, &merge);
				sprintf(dsuffix, ".%d", (int)(target_densities[d] + 0.5));
				filter_dump_relsets(obj, &merge, dsuffix);
				filter_free_relsets(&merge);
				relations_needed = 0;
			}
			free(saved_rel_array);
			get_filter_tmp_name(obj, lp_filename,
					sizeof(lp_filename), ".lp");
			remove(lp_filename);
			wall_time = time(NULL) - wall_time;
			logprintf(obj, "RelProcTime: %u\n", (uint32)wall_time);
			goto finished;
		}
	}
	else {
		/* dataset is "large", perform multiple singleton passes.

		   The first pass used a large bound; the second
		   filtering bound is much smaller. To allow reuse of
		   previous results, the second bound is used
		   during the rest of the filtering */

		dump_relation_numbers(obj, &filter);
		set_filtering_bounds(obj, &fb, filtmin_r, filtmin_a,
					&entries_r, &entries_a,
					filter.num_relations, 1, &filter);

		free(filter.relation_array);
		free(filter.relation_ptr);
		filter.relation_array = NULL;

		nfs_write_lp_file(obj, &fb, &filter, max_relations, 1);
		nfs_compact_lp_file(obj, &filter, ram_size);

		if (filter.lp_file_size < ram_size / 2) {

			/* dataset is small enough for filtering to
			   complete in one pass */

			if (num_densities <= 1) {
				filter_read_lp_file(obj, &filter, 0);
				if ((relations_needed = do_merge(obj, &filter,
							&merge, target_density,
							ckpt_path, 1)) > 0) {
					goto finished;
				}
			}
			else {
				uint32 d;
				char dsuffix[32];
				uint8 *saved_rel_array;
				size_t saved_rel_bytes;
				uint32 saved_nr, saved_ni;
				uint32 extra_needed;
				filter_read_lp_file(obj, &filter, 0);
				extra_needed = filter.target_excess;

				/* run cliques once for all densities */
				filter.target_excess = (uint32)(filter.target_excess *
								FINAL_EXCESS_FRACTION);
				if ((relations_needed = check_excess(&filter)) > 0)
					goto finished;
				filter_purge_cliques(obj, &filter);

				/* save post-clique state */
				saved_nr = filter.num_relations;
				saved_ni = filter.num_ideals;
				{
					relation_ideal_t *rp = filter.relation_array;
					uint32 k;
					for (k = 0; k < saved_nr; k++)
						rp = next_relation_ptr(rp);
					saved_rel_bytes = (uint8 *)rp - (uint8 *)filter.relation_array;
				}
				saved_rel_array = (uint8 *)xmalloc(saved_rel_bytes);
				memcpy(saved_rel_array, filter.relation_array, saved_rel_bytes);
				for (d = 0; d < num_densities; d++) {
					if (d > 0) {
						free(filter.relation_array);
						free(filter.relation_ptr);
						filter.relation_array = (relation_ideal_t *)xmalloc(saved_rel_bytes);
						memcpy(filter.relation_array, saved_rel_array, saved_rel_bytes);
						filter.relation_ptr = (relation_ideal_t **)xmalloc(
									saved_nr * sizeof(relation_ideal_t *));
						{
							relation_ideal_t *rp = filter.relation_array;
							uint32 k;
							for (k = 0; k < saved_nr; k++) {
								filter.relation_ptr[k] = rp;
								rp = next_relation_ptr(rp);
							}
						}
						filter.num_relations = saved_nr;
						filter.num_ideals    = saved_ni;
					}
					logprintf(obj, "trying target density %.0f\n",
							target_densities[d]);
					memset(&merge, 0, sizeof(merge));
					merge.num_extra_relations = NUM_EXTRA_RELATIONS;
					merge.target_density = target_densities[d];
					filter_merge_init(obj, &filter);
					filter_merge_2way(obj, &filter, &merge);
					if (filter_merge_full(obj, &merge, extra_needed) != 0) {

						/* do_merge() frees the half-built relsets before it
						   returns; these loops have to do it themselves,
						   and the finished: label does not */

						if (merge.relset_array != NULL || merge.data_pool != NULL)
							filter_free_relsets(&merge);
						if (d == 0) {
							relations_needed = 1000000;
							free(saved_rel_array);
							goto finished;
						}
						break;
					}
					filter_postproc_relsets(obj, &merge);
					sprintf(dsuffix, ".%d", (int)(target_densities[d] + 0.5));
					filter_dump_relsets(obj, &merge, dsuffix);
					filter_free_relsets(&merge);
					relations_needed = 0;
				}
				free(saved_rel_array);
				get_filter_tmp_name(obj, lp_filename,
						sizeof(lp_filename), ".lp");
				remove(lp_filename);
				wall_time = time(NULL) - wall_time;
				logprintf(obj, "RelProcTime: %u\n", (uint32)wall_time);
				goto finished;
			}
		}
		else {
			/* dataset is so large that even the pruned version
			   cannot fit comfortably in memory. We have to put
			   up with reading only the ideals that occur in the
			   fewest relations, forming the matrix, and then
			   determining whether the matrix incorporates enough
			   of the dataset so that the matrix will work */

			if (num_densities <= 1) {
				if ((relations_needed = do_partial_filtering(obj,
							&filter, &merge, entries_r,
							entries_a, target_density,
							max_weight, ckpt_path)) > 0) {
					goto finished;
				}
			}
			else {
				uint32 d;
				uint32 sv_nr = filter.num_relations;
				uint32 sv_ni = filter.num_ideals;
				char dsuffix[32];
				for (d = 0; d < num_densities; d++) {
					if (d > 0) {
						free(filter.relation_array);
						free(filter.relation_ptr);
						filter.relation_array = NULL;
						filter.relation_ptr = NULL;
						filter.num_relations = sv_nr;
						filter.num_ideals    = sv_ni;
					}
					logprintf(obj, "trying target density %.0f\n",
							target_densities[d]);
					memset(&merge, 0, sizeof(merge));
					{
						uint32 dn = do_partial_filtering(obj, &filter, &merge,
								entries_r, entries_a, target_densities[d],
								max_weight, NULL);
						if (dn > 0) {

							/* as above: nothing downstream frees these */

							if (merge.relset_array != NULL || merge.data_pool != NULL)
								filter_free_relsets(&merge);
							if (d == 0) {
								relations_needed = dn;
								goto finished;
							}
							break;
						}
					}
					filter_postproc_relsets(obj, &merge);
					sprintf(dsuffix, ".%d", (int)(target_densities[d] + 0.5));
					filter_dump_relsets(obj, &merge, dsuffix);
					filter_free_relsets(&merge);
					relations_needed = 0;
				}
				get_filter_tmp_name(obj, lp_filename,
						sizeof(lp_filename), ".lp");
				remove(lp_filename);
				wall_time = time(NULL) - wall_time;
				logprintf(obj, "RelProcTime: %u\n", (uint32)wall_time);
				goto finished;
			}
		}
	}
	} /* end pre_read state block */

	/* single-density filtering succeeded; delete the LP file */

	get_filter_tmp_name(obj, lp_filename, sizeof(lp_filename), ".lp");
	remove(lp_filename);

	/* optimize and then save the collection of relation-sets */

merge_done:

	filter_postproc_relsets(obj, &merge);
	filter_dump_relsets(obj, &merge, "");
	filter_free_relsets(&merge);
	wall_time = time(NULL) - wall_time;
	logprintf(obj, "RelProcTime: %u\n", (uint32)wall_time);
finished:

	/* the matrix build reads the savefile once more. When it is going
	   to run in this same invocation, leave the staged copy in place
	   for it rather than decompressing the whole thing twice. */

	if (relations_needed == 0 && (obj->flags & MSIEVE_FLAG_NFS_LA))
		savefile_unstage_tmp(obj);
	else
		savefile_unstage(obj);

	mpz_poly_free(&fb.rfb.poly);
	mpz_poly_free(&fb.afb.poly);
	return relations_needed;
}
