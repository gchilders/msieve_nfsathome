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
/* Checkpointing the merge input.

   Everything before the full merge -- duplicate removal, the singleton
   passes, clique removal, the 2-way merge -- is deterministic and takes
   the large majority of a filtering run, while the full merge is the
   part worth experimenting on. Writing the relation sets out once, just
   before the full merge starts, lets later runs restart from there and
   iterate on the merge in isolation.

   This is a developer tool, so the format is whatever is fastest to read
   back on the same machine: native byte order and native struct widths,
   with a magic number carrying a layout tag so a stale file is rejected
   rather than misread. */

/* The payload is a raw dump of relation_set_t fields and pool-allocated
   arrays, so a file written by a build with a different layout must be
   refused rather than reinterpreted. The magic alone cannot do that -- it
   is a constant nobody remembers to change -- so the header also carries
   the sizes and packing the reader depends on. */

#define MERGE_CKPT_MAGIC "MSVMRG"
#define MERGE_CKPT_VERSION 1
#define MERGE_CKPT_BUFSIZE (4 * 1024 * 1024)

typedef struct {
	FILE *fp;
	uint8 *buf;
	size_t used;
	size_t avail;
	uint32 failed;
} ckpt_io_t;

static void ckpt_flush(ckpt_io_t *io) {

	if (io->used && fwrite(io->buf, 1, io->used, io->fp) != io->used)
		io->failed = 1;
	io->used = 0;
}

static void ckpt_write(ckpt_io_t *io, const void *src, size_t len) {

	const uint8 *p = (const uint8 *)src;

	while (len) {
		size_t n = MERGE_CKPT_BUFSIZE - io->used;

		if (n == 0) {
			ckpt_flush(io);
			continue;
		}
		if (n > len)
			n = len;
		memcpy(io->buf + io->used, p, n);
		io->used += n;
		p += n;
		len -= n;
	}
}

static uint32 ckpt_read(ckpt_io_t *io, void *dst, size_t len) {

	uint8 *p = (uint8 *)dst;

	while (len) {
		size_t n = io->avail - io->used;

		if (n == 0) {
			io->avail = fread(io->buf, 1, MERGE_CKPT_BUFSIZE, io->fp);
			io->used = 0;
			if (io->avail == 0)
				return 0;
			continue;
		}
		if (n > len)
			n = len;
		memcpy(p, io->buf + io->used, n);
		io->used += n;
		p += n;
		len -= n;
	}
	return 1;
}

/*--------------------------------------------------------------------*/
int32 filter_merge_checkpoint_save(msieve_obj *obj, merge_t *merge,
				uint32 min_cycles, const char *path) {

	uint32 i;
	ckpt_io_t io;
	uint64 total_words = 0;
	time_t start = time(NULL);

	io.fp = fopen(path, "wb");
	if (io.fp == NULL) {
		logprintf(obj, "error: cannot create merge checkpoint '%s'\n", path);
		return -1;
	}
	io.buf = (uint8 *)xmalloc(MERGE_CKPT_BUFSIZE);
	io.used = 0;
	io.avail = 0;
	io.failed = 0;

	{
		uint32 layout[4];

		layout[0] = MERGE_CKPT_VERSION;
		layout[1] = (uint32)sizeof(relation_set_t);
		layout[2] = RELSET_ACTIVE_BITS;
		layout[3] = (uint32)sizeof(uint32);
		ckpt_write(&io, MERGE_CKPT_MAGIC, 6);
		ckpt_write(&io, layout, sizeof(layout));
	}
	ckpt_write(&io, &merge->num_relsets, sizeof(uint32));
	ckpt_write(&io, &merge->num_ideals, sizeof(uint32));
	ckpt_write(&io, &merge->num_extra_relations, sizeof(uint32));
	ckpt_write(&io, &min_cycles, sizeof(uint32));
	ckpt_write(&io, &merge->target_density, sizeof(double));

	for (i = 0; i < merge->num_relsets; i++) {
		relation_set_t *r = merge->relset_array + i;
		uint16 active = (uint16)relation_set_num_active(r);
		uint32 words = (uint32)r->num_relations + r->num_large_ideals;

		ckpt_write(&io, &r->num_relations, sizeof(uint16));
		ckpt_write(&io, &r->num_small_ideals, sizeof(uint16));
		ckpt_write(&io, &r->num_large_ideals, sizeof(uint16));
		ckpt_write(&io, &active, sizeof(uint16));
		if (words)
			ckpt_write(&io, r->data, words * sizeof(uint32));
		total_words += words;
	}
	ckpt_flush(&io);
	free(io.buf);

	if (io.failed || fclose(io.fp) != 0) {
		logprintf(obj, "error: write failed on merge checkpoint\n");
		remove(path);
		return -1;
	}

	logprintf(obj, "saved merge checkpoint: %u relation sets, "
			"%.1f MB in %u sec\n", merge->num_relsets,
			(double)(total_words * sizeof(uint32) +
			(uint64)merge->num_relsets * 8) / 1048576,
			(uint32)(time(NULL) - start));
	return 0;
}

/*--------------------------------------------------------------------*/
int32 filter_merge_checkpoint_load(msieve_obj *obj, merge_t *merge,
				uint32 *min_cycles, const char *path) {

	uint32 i;
	ckpt_io_t io;
	char magic[6];
	uint32 layout[4];
	uint32 num_relsets;
	time_t start = time(NULL);

	io.fp = fopen(path, "rb");
	if (io.fp == NULL)
		return -1;

	io.buf = (uint8 *)xmalloc(MERGE_CKPT_BUFSIZE);
	io.used = 0;
	io.avail = 0;
	io.failed = 0;

	if (!ckpt_read(&io, magic, 6) ||
	    memcmp(magic, MERGE_CKPT_MAGIC, 6) != 0 ||
	    !ckpt_read(&io, layout, sizeof(layout))) {
		logprintf(obj, "error: '%s' is not a merge checkpoint\n",
				path);
		free(io.buf);
		fclose(io.fp);
		return -1;
	}
	if (layout[0] != MERGE_CKPT_VERSION ||
	    layout[1] != (uint32)sizeof(relation_set_t) ||
	    layout[2] != RELSET_ACTIVE_BITS ||
	    layout[3] != (uint32)sizeof(uint32)) {
		logprintf(obj, "error: '%s' was written by a build with a "
				"different relation-set layout; delete it and "
				"let it be rebuilt\n", path);
		free(io.buf);
		fclose(io.fp);
		return -1;
	}

	memset(merge, 0, sizeof(*merge));
	if (!ckpt_read(&io, &num_relsets, sizeof(uint32)) ||
	    !ckpt_read(&io, &merge->num_ideals, sizeof(uint32)) ||
	    !ckpt_read(&io, &merge->num_extra_relations, sizeof(uint32)) ||
	    !ckpt_read(&io, min_cycles, sizeof(uint32)) ||
	    !ckpt_read(&io, &merge->target_density, sizeof(double))) {
		logprintf(obj, "error: truncated merge checkpoint\n");
		exit(-1);
	}

	merge->num_relsets = num_relsets;
	merge->data_pool = merge_mem_pool_create();
	merge->relset_array = (relation_set_t *)xcalloc((size_t)num_relsets,
					sizeof(relation_set_t));

	for (i = 0; i < num_relsets; i++) {
		relation_set_t *r = merge->relset_array + i;
		uint16 active;
		uint32 words;

		if (!ckpt_read(&io, &r->num_relations, sizeof(uint16)) ||
		    !ckpt_read(&io, &r->num_small_ideals, sizeof(uint16)) ||
		    !ckpt_read(&io, &r->num_large_ideals, sizeof(uint16)) ||
		    !ckpt_read(&io, &active, sizeof(uint16))) {
			logprintf(obj, "error: truncated merge checkpoint\n");
			exit(-1);
		}

		/* the payload is re-allocated here, so its pool class comes
		   from this allocation rather than from the file */

		words = (uint32)r->num_relations + r->num_large_ideals;
		r->num_active_ideals = 0;

		/* merge_relset_alloc() stamps the pool class into the relset
		   but hands the payload back rather than storing it */

		r->data = merge_relset_alloc(merge->data_pool, r, words);
		if (words) {
			if (!ckpt_read(&io, r->data, words * sizeof(uint32))) {
				logprintf(obj, "error: truncated merge "
						"checkpoint\n");
				exit(-1);
			}
		}
		relation_set_set_num_active(r, active);
	}

	free(io.buf);
	fclose(io.fp);
	logprintf(obj, "loaded merge checkpoint: %u relation sets, "
			"%u ideals in %u sec\n", merge->num_relsets,
			merge->num_ideals, (uint32)(time(NULL) - start));
	return 0;
}

/*--------------------------------------------------------------------*/
void filter_free_relsets(merge_t *merge) {

	uint32 i;
	relation_set_t *relset_array = merge->relset_array;
	uint32 num_relsets = merge->num_relsets;

	for (i = 0; relset_array != NULL && i < num_relsets; i++) {
		relation_set_t *r = relset_array + i;
		merge_relset_free(merge->data_pool, r);
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
				merge_t *merge, uint32 min_cycles,
				const char *ckpt_path) {

	filter_purge_cliques(obj, filter);
	filter_merge_init(obj, filter);
	filter_merge_2way(obj, filter, merge);

	/* This is the last point at which the merge input is still exactly
	   reproducible, so it is what a checkpoint records. do_partial_
	   filtering() calls this routine again for each retry at a higher
	   max_weight, and rewriting a multi-gigabyte dump every time would
	   cost minutes for nothing, so only write one if there is none. */

	if (ckpt_path != NULL) {
		FILE *probe = fopen(ckpt_path, "rb");

		if (probe != NULL)
			fclose(probe);
		else
			filter_merge_checkpoint_save(obj, merge, min_cycles,
					ckpt_path);
	}

	return filter_merge_full(obj, merge, min_cycles);
}
