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
void filter_free_relsets(merge_t *merge) {

	uint32 i;
	relation_set_t *relset_array = merge->relset_array;
	uint32 num_relsets = merge->num_relsets;

	for (i = 0; relset_array != NULL && i < num_relsets; i++) {
		relation_set_t *r = relset_array + i;
		merge_mem_free(merge->data_pool, r->data,
			r->num_relations + r->num_large_ideals);
	}
	merge_mem_pool_destroy(merge->data_pool);
	merge->data_pool = NULL;
	free(merge->relset_array);
	merge->relset_array = NULL;
	merge->num_relsets = 0;
	merge->num_ideals = 0;
}

static uint32 get_committed_rmap_generation(msieve_obj *obj, uint64 *generation) {
	char buf[256];
	FILE *map_fp, *commit_fp;
	uint64 magic, version, gen, count;
	uint64 cmagic, cgen, ccount;
	sprintf(buf, "%s.rmap", obj->savefile.name);
	map_fp = fopen(buf, "rb");
	if (map_fp == NULL)
		return 0;
	if (fread(&magic, sizeof(uint64), 1, map_fp) != 1 ||
	    fread(&version, sizeof(uint64), 1, map_fp) != 1 ||
	    fread(&gen, sizeof(uint64), 1, map_fp) != 1 ||
	    fread(&count, sizeof(uint64), 1, map_fp) != 1 ||
	    fclose(map_fp) != 0 || magic != NFS_RMAP_MAGIC ||
	    version != NFS_RMAP_VERSION) {
		logprintf(obj, "error: invalid relation map metadata\n");
		exit(-1);
	}
	sprintf(buf, "%s.rmap.commit", obj->savefile.name);
	commit_fp = fopen(buf, "rb");
	if (commit_fp == NULL ||
	    fread(&cmagic, sizeof(uint64), 1, commit_fp) != 1 ||
	    fread(&cgen, sizeof(uint64), 1, commit_fp) != 1 ||
	    fread(&ccount, sizeof(uint64), 1, commit_fp) != 1 ||
	    fclose(commit_fp) != 0 || cmagic != NFS_RMAP_COMMIT_MAGIC ||
	    cgen != gen || ccount != count) {
		logprintf(obj, "error: relation map is not transactionally committed\n");
		exit(-1);
	}
	*generation = gen;
	return 1;
}

/*--------------------------------------------------------------------*/
void filter_dump_relsets(msieve_obj *obj, merge_t *merge) {

	uint32 i;
	relation_set_t *relset_array = merge->relset_array;
	uint32 num_relsets = merge->num_relsets;
	char buf[256];
	FILE *cycle_fp;
	uint64 rmap_generation = 0;
	uint32 have_rmap;

	sprintf(buf, "%s.cyc", obj->savefile.name);
	have_rmap = get_committed_rmap_generation(obj, &rmap_generation);
	cycle_fp = fopen(buf, "wb");
	if (cycle_fp == NULL) {
		logprintf(obj, "error: can't open cycle file\n");
		exit(-1);
	}

	if (have_rmap) {
		uint32 magic = CYCLE_FILE_MAGIC;
		uint32 version = CYCLE_FILE_VERSION;
		uint32 flags = CYCLE_FLAG_RMAP_REQUIRED;
		if (fwrite(&magic, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&version, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&flags, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&num_relsets, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&rmap_generation, sizeof(uint64), 1, cycle_fp) != 1) {
			logprintf(obj, "error: can't write cycle header\n");
			exit(-1);
		}
	}
	else if (fwrite(&num_relsets, sizeof(uint32), 1, cycle_fp) != 1) {
		logprintf(obj, "error: can't write cycle header\n");
		exit(-1);
	}

	for (i = 0; i < num_relsets; i++) {
		relation_set_t *r = relset_array + i;
		uint32 num = r->num_relations;

		if (fwrite(&num, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(r->data, sizeof(uint32), (size_t)num, cycle_fp) != num) {
			logprintf(obj, "error: can't write cycle file\n");
			exit(-1);
		}
	}
	if (fflush(cycle_fp) != 0 || ferror(cycle_fp) || fclose(cycle_fp) != 0) {
		logprintf(obj, "error: can't finalize cycle file\n");
		exit(-1);
	}
}

/*--------------------------------------------------------------------*/
int32 filter_make_relsets(msieve_obj *obj, filter_t *filter,
				merge_t *merge, uint32 min_cycles) {

	filter_purge_cliques(obj, filter);
	filter_merge_init(obj, filter);
	filter_merge_2way(obj, filter, merge);
	return filter_merge_full(obj, merge, min_cycles);
}
