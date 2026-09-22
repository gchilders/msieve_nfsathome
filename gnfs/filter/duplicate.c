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
#include "hash64.h"
#ifdef HAVE_OMP
#include <omp.h>
#endif

/* produce <savefile_name>.d, a binary file containing the
   line numbers of relations in the savefile that should *not*
   graduate to the singleton removal (or are just plain invalid).

   This code has to touch all of the relations, and when the
   dataset is large avoiding excessive memory use is tricky.
   Cavallar's paper suggests dropping relations into a hashtable
   and ignoring relations that map to the same hash bin. This keeps
   memory use down but causes good relations to be thrown away.
   Another option is to put the (a,b) values of relations into the
   hashtable, so that hash collisions can be resolved rigorously.
   At multi-billion-relation scale the second-pass table stores the
   complete 128-bit (a,b) key plus a 64-bit chain link in segmented
   storage. This costs about 24 bytes per collision candidate, but avoids
   both false duplicate matches and the old 32-bit table-size ceiling.

   The implementation here is a compromise: we do duplicate removal
   in two passes. The first pass maps relations into a hashtable
   of bits, and we save (on disk) the list of hash bins where two
   or more relations collide. The second pass refills the hashtable
   of bits with just these entries, then reads through the complete
   dataset again and saves the (a,b) values of any relation that
   maps to one of the filled-in hash bins. The memory use in the
   first pass is constant, and the memory use of the second pass is
   proportional only to relations landing in collision buckets. Assuming
   unique relations greatly outnumber duplicates, this solution finds all the duplicates
   with no false positives, and the memory use is low enough so
   that singleton filtering is a larger memory bottleneck

   One useful optimization for really big problems would turn the
   first-pass hashtable into a Bloom filter using several hash
   functions. This would make it much more effective at avoiding
   false positives as the hashtable gets more congested */

static const uint8 hashmask[] = {0x01, 0x02, 0x04, 0x08,
				 0x10, 0x20, 0x40, 0x80};

/* Pass 2 exists only because the set of colliding hash bins is not known
   until pass 1 has seen every relation. Its actual per-relation work is
   trivial -- pull (a,b) off the line, hash it, test one bit -- so on a
   machine whose relations live on a network share the pass is dominated by
   reading the savefile a second time.

   When there is memory to spare we therefore keep every relation's (a,b) in
   core during pass 1 and let pass 2 walk that instead of the file. The cache
   is segmented so that it never has to be reallocated or copied, and it is
   sized from the same relation-count estimate used for the pass 1 hashtable;
   if that estimate turns out to be too low the cache is dropped and pass 2
   falls back to re-reading, which costs time but never correctness.

   Storing the full 128-bit (a,b) rather than a digest is deliberate: it keeps
   the "no false duplicates" guarantee that the two-hashtable scheme exists to
   provide. */

/* the rigorous second-stage key is the full 128-bit (a,b) pair */
#define DUP_KEY_WORDS 4

#define DUP_AB_SEGMENT_LOG2 20
#define DUP_AB_SEGMENT_SIZE ((uint64)1 << DUP_AB_SEGMENT_LOG2)
#define DUP_AB_SEGMENT_MASK (DUP_AB_SEGMENT_SIZE - 1)

typedef struct {
	uint32 active;          /* zero once the cache has been abandoned */
	uint64 max_relations;   /* ordinals we are willing to store */
	uint64 num_segments;
	uint64 segment_alloc;
	abpair_t **segments;
} dup_ab_cache_t;

static void dup_ab_cache_init(dup_ab_cache_t *c, uint64 max_relations) {

	memset(c, 0, sizeof(*c));
	if (max_relations == 0)
		return;
	c->max_relations = max_relations;
	c->segment_alloc = (max_relations >> DUP_AB_SEGMENT_LOG2) + 2;
	c->segments = (abpair_t **)calloc((size_t)c->segment_alloc,
					sizeof(abpair_t *));
	if (c->segments == NULL)
		return;
	c->active = 1;
}

static void dup_ab_cache_free(dup_ab_cache_t *c) {

	uint64 i;

	if (c->segments != NULL) {
		for (i = 0; i < c->num_segments; i++)
			free(c->segments[i]);
		free(c->segments);
	}
	memset(c, 0, sizeof(*c));
}

/* record the coordinates of relation 'ordinal'; on any allocation failure or
   overflow of the estimate, abandon the cache rather than the run */

static void dup_ab_cache_store(dup_ab_cache_t *c, uint64 ordinal,
				int64 a, uint64 b) {

	uint64 seg = ordinal >> DUP_AB_SEGMENT_LOG2;
	abpair_t *p;

	if (!c->active)
		return;
	if (ordinal >= c->max_relations || seg >= c->segment_alloc) {
		dup_ab_cache_free(c);
		return;
	}

	while (c->num_segments <= seg) {
		p = (abpair_t *)malloc((size_t)DUP_AB_SEGMENT_SIZE *
					sizeof(abpair_t));
		if (p == NULL) {
			dup_ab_cache_free(c);
			return;
		}
		c->segments[c->num_segments++] = p;
	}

	p = c->segments[seg] + (size_t)(ordinal & DUP_AB_SEGMENT_MASK);
	p->a = a;
	p->b = b;
}

static const abpair_t *dup_ab_cache_get(const dup_ab_cache_t *c,
					uint64 ordinal) {

	return c->segments[ordinal >> DUP_AB_SEGMENT_LOG2] +
			(size_t)(ordinal & DUP_AB_SEGMENT_MASK);
}

static inline uint64 ror64(uint64 v, int r) {
    return (v >> r) | (v << (64 - r));
}

static uint64 rrxmrrxmsx_0(uint64 v) {
    v ^= ror64(v, 25) ^ ror64(v, 50);
    v *= 0xA24BAED4963EE407UL;
    v ^= ror64(v, 24) ^ ror64(v, 49);
    v *= 0x9FB21C651E98DF25UL;
    return v ^ v >> 28;
}

static void dup_write_u64(msieve_obj *obj, FILE *fp, uint64 value, const char *what) {
	if (fwrite(&value, sizeof(uint64), 1, fp) != 1) {
		logprintf(obj, "error: write failed for %s\n", what);
		exit(-1);
	}
}

static void dup_close_output(msieve_obj *obj, FILE *fp, const char *what) {
	if (fflush(fp) != 0 || ferror(fp) || fclose(fp) != 0) {
		logprintf(obj, "error: can't finalize %s\n", what);
		exit(-1);
	}
}

/* Return 1 for a complete record, 0 for clean EOF, and -1 for a
   truncated record or I/O error. */
static int dup_read_u64(FILE *fp, uint64 *value) {
	size_t n;
	if (feof(fp))
		return 0;
	n = fread(value, 1, sizeof(uint64), fp);
	if (n == sizeof(uint64))
		return 1;
	if (n == 0 && feof(fp) && !ferror(fp))
		return 0;
	return -1;
}

/* decide whether one relation is a duplicate, and record it if so. Shared
   by the cached and the re-reading forms of pass 2 so the two cannot drift */

static void dup2_classify(msieve_obj *obj, nfs_hashtable64_t *duplicates,
			const uint8 *bit_table, uint32 log2_hashtable1_size,
			int64 a, uint64 b, uint64 curr_relation, FILE *out_fp,
			uint64 *num_relations, uint64 *num_duplicates) {

	uint32 key[DUP_KEY_WORDS];
	uint64 hashval;

	key[0] = (uint32)(uint64)a;
	key[1] = (uint32)((uint64)a >> 32);
	key[2] = (uint32)b;
	key[3] = (uint32)(b >> 32);

	hashval = (rrxmrrxmsx_0((uint64)a) ^ rrxmrrxmsx_0((uint64)b)) >>
					(64 - log2_hashtable1_size);

	if (bit_table[hashval / 8] & hashmask[hashval % 8]) {

		/* relation collides in the first hashtable; use the second
		   hashtable to determine rigorously if it was seen before */

		uint32 is_new;
		nfs_hash64_find(obj, duplicates, key, &is_new);

		if (is_new) {
			(*num_relations)++;
		}
		else {
			dup_write_u64(obj, out_fp, curr_relation,
					"duplicate relation list");
			(*num_duplicates)++;
		}
	}
	else {
		/* no collision; relation is unique */

		(*num_relations)++;
	}
}

static uint64 purge_duplicates_pass2(msieve_obj *obj,
				uint32 log2_hashtable1_size,
				uint64 max_relations,
				const dup_ab_cache_t *ab_cache,
				uint8 *collision_bits,
				uint64 total_relations) {

	savefile_t *savefile = &obj->savefile;
	FILE *bad_relation_fp;
	FILE *collision_fp;
	FILE *out_fp;
	uint64 i;
	char buf[LINE_BUF_SIZE];
	uint64 num_duplicates;
	uint64 num_relations;
	uint64 next_bad_relation;
	uint64 curr_relation;
	uint8 *bit_table;
	nfs_hashtable64_t duplicates;
	uint32 use_cache = (ab_cache != NULL && ab_cache->active);

	logprintf(obj, "commencing duplicate removal, pass 2\n");

	/* fill in the list of hash collisions. Pass 1 hands these over
	   in memory whenever the relation cache survived; otherwise they
	   are read back off disk */

	if (use_cache) {
		bit_table = collision_bits;
		goto collisions_ready;
	}

	get_filter_tmp_name(obj, buf, sizeof(buf), ".hc");
	collision_fp = fopen(buf, "rb");
	if (collision_fp == NULL) {
		logprintf(obj, "error: dup2 can't open collision file\n");
		exit(-1);
	}
	bit_table = (uint8 *)xcalloc(
			(uint64)1 << (log2_hashtable1_size - 3),
			sizeof(uint8));

	while (1) {
		int rc = dup_read_u64(collision_fp, &i);
		if (rc == 0)
			break;
		if (rc < 0) {
			logprintf(obj, "error: truncated duplicate collision file\n");
			exit(-1);
		}
		if (i < ((uint64)1 << log2_hashtable1_size)) {
			bit_table[i / 8] |= 1 << (i % 8);
		}
	}
	fclose(collision_fp);

collisions_ready:

	/* set up for reading the list of relations */

	get_filter_tmp_name(obj, buf, sizeof(buf), ".br");
	bad_relation_fp = fopen(buf, "rb");
	if (bad_relation_fp == NULL) {
		logprintf(obj, "error: dup2 can't open rel file\n");
		exit(-1);
	}
	get_filter_tmp_name(obj, buf, sizeof(buf), ".d");
	out_fp = fopen(buf, "wb");
	if (out_fp == NULL) {
		logprintf(obj, "error: dup2 can't open output file\n");
		exit(-1);
	}
	nfs_hash64_init(obj, &duplicates, DUP_KEY_WORDS);

	num_duplicates = 0;
	num_relations = 0;
	curr_relation = UINT64_MAX;
	next_bad_relation = UINT64_MAX;
	{
		int rc = dup_read_u64(bad_relation_fp, &next_bad_relation);
		if (rc < 0) {
			logprintf(obj, "error: truncated bad-relation file\n");
			exit(-1);
		}
	}
	if (use_cache) {

		/* pass 1 kept every relation's coordinates, so the whole
		   savefile read can be skipped */

		for (curr_relation = 0; curr_relation < total_relations;
							curr_relation++) {

			const abpair_t *ab;

			if (max_relations && curr_relation >= max_relations)
				break;

			if (curr_relation == next_bad_relation) {
				dup_write_u64(obj, out_fp, curr_relation,
						"duplicate relation list");
				{
					int rc = dup_read_u64(bad_relation_fp,
							&next_bad_relation);
					if (rc < 0) {
						logprintf(obj, "error: truncated bad-relation file\n");
						exit(-1);
					}
					if (rc == 0)
						next_bad_relation = UINT64_MAX;
				}
				continue;
			}

			ab = dup_ab_cache_get(ab_cache, curr_relation);
			dup2_classify(obj, &duplicates, bit_table,
					log2_hashtable1_size, ab->a, ab->b,
					curr_relation, out_fp,
					&num_relations, &num_duplicates);
		}
		goto relations_done;
	}

	savefile_open(savefile, SAVEFILE_READ);
	savefile_read_line(buf, sizeof(buf), savefile);

	while (!savefile_eof(savefile)) {

		int64 a;
		uint64 b;
		char *next_field;

		if (buf[0] != '-' && !isdigit(buf[0])) {

			/* no relation on this line */

			savefile_read_line(buf, sizeof(buf), savefile);
			continue;
		}
		curr_relation++;
		if (max_relations && curr_relation >= max_relations)
			break;

		if (curr_relation == next_bad_relation) {

			/* this relation isn't valid; save it and
			   read in the next invalid relation line number */

			dup_write_u64(obj, out_fp, curr_relation, "duplicate relation list");
			{
				int rc = dup_read_u64(bad_relation_fp, &next_bad_relation);
				if (rc < 0) {
					logprintf(obj, "error: truncated bad-relation file\n");
					exit(-1);
				}
				if (rc == 0)
					next_bad_relation = UINT64_MAX;
			}
			savefile_read_line(buf, sizeof(buf), savefile);
			continue;
		}

		/* determine if the (a,b) coordinates of the
		   relation collide in the table of bits */

		a = strtoll(buf, &next_field, 10);
		b = strtoull(next_field + 1, NULL, 10);

		dup2_classify(obj, &duplicates, bit_table, log2_hashtable1_size,
				a, b, curr_relation, out_fp,
				&num_relations, &num_duplicates);

		savefile_read_line(buf, sizeof(buf), savefile);
	}

relations_done:
	logprintf(obj, "found %" PRIu64 " duplicates and %" PRIu64
			" unique relations\n", num_duplicates, num_relations);
	logprintf(obj, "memory use: %.1f MB\n",
			(double)(((uint64)1 << (log2_hashtable1_size-3)) +
			nfs_hash64_sizeof(&duplicates)) / 1048576);

	/* clean up and finish */

	if (!use_cache)
		savefile_close(savefile);
	fclose(bad_relation_fp);
	dup_close_output(obj, out_fp, "duplicate relation list");
	get_filter_tmp_name(obj, buf, sizeof(buf), ".hc");
	remove(buf);
	get_filter_tmp_name(obj, buf, sizeof(buf), ".br");
	remove(buf);

	if (!use_cache)
		free(bit_table);
	nfs_hash64_free(&duplicates);
	return num_relations;
}

/*--------------------------------------------------------------------*/
static double estimate_rel_size(savefile_t *savefile) {

	uint32 i;
	char buf[LINE_BUF_SIZE];
	uint32 num_relations = 0;
	uint32 totlen = 0;

	savefile_open(savefile, SAVEFILE_READ);
	savefile_read_line(buf, sizeof(buf), savefile);
	for (i = 0; i < 100 && !savefile_eof(savefile); i++) {

		if (buf[0] != '-' && !isdigit(buf[0])) {
			/* no relation on this line */
			savefile_read_line(buf, sizeof(buf), savefile);
			continue;
		}

		num_relations++;
		totlen += strlen(buf);
	}

	savefile_close(savefile);
	if (num_relations == 0)
		return 0;
	return (double)totlen / num_relations;
}

/*--------------------------------------------------------------------*/
#define LOG2_BIN_SIZE 17
#define BIN_SIZE (1 << (LOG2_BIN_SIZE))
#define TARGET_HITS_PER_PRIME 40.0

/* Reading the savefile is one gzgets per line and cannot be split across
   threads, but it does not have to sit between the parallel passes: it
   only has to stay ahead of them. Holding the sequential reader state
   here lets one thread fill the next batch's buffer while the rest of the
   team parses the batch already in hand. */

typedef struct {
	savefile_t *savefile;
	uint64 max_relations;
	uint32 batch;
	uint64 curr_relation;   /* ordinal of the last relation handed out */
	uint32 done;
} dup_reader_t;

static void dup_read_batch(dup_reader_t *r, char *buf, uint64 *ords,
			uint32 *count_out) {

	uint32 i;
	uint32 count = 0;

	if (r->done) {
		*count_out = 0;
		return;
	}

	for (i = 0; i < r->batch; i++) {
		char *buf_i = buf + i * LINE_BUF_SIZE;

		savefile_read_line(buf_i, LINE_BUF_SIZE * sizeof(char),
				r->savefile);
		if (savefile_eof(r->savefile)) {
			r->done = 1;
			break;
		}
		if (buf_i[0] != '-' && !isdigit(buf_i[0])) {

			/* no relation on this line */

			i--;
			continue;
		}
		ords[i] = r->curr_relation + i + 1;
		count++;
	}

	r->curr_relation += count;
	if (r->max_relations && r->curr_relation >= r->max_relations) {
		uint64 excess = r->curr_relation - r->max_relations + 1;

		r->curr_relation -= excess;
		count -= (uint32)excess;
		r->done = 1;
	}
	*count_out = count;
}

uint32 nfs_purge_duplicates(msieve_obj *obj, factor_base_t *fb,
				uint64 max_relations, uint64 ram_size,
				uint64 *num_relations_out) {

	uint32 i;
	savefile_t *savefile = &obj->savefile;
	FILE *bad_relation_fp;
	FILE *collision_fp;
	uint64 curr_relation;
	uint64 *my_curr_relation;
	char *buf;
	uint64 num_relations;
	uint64 num_collisions;
	uint64 num_composite;
	uint64 num_malformed;
	uint8 *hashtable;
	uint32 log2_hashtable1_size;
	double rel_size = estimate_rel_size(savefile);
	double est_num_rels = 0.0;
	dup_ab_cache_t ab_cache;
	uint8 *collision_bits = NULL;
	uint64 cache_bytes = 0;
	uint64 num_free_added = 0;
	mpz_t *scratch;

	uint8 *free_relation_bits;
	uint32 *free_relations;
	uint32 num_free_relations;
	uint32 num_free_relations_alloc;

	uint64 *prime_bins;
	uint64 *thread_bins;
	char *buf2;
	uint64 *ord2;
	char *buf_cur, *buf_nxt, *tmp_buf;
	uint64 *ord_cur, *ord_nxt, *tmp_ord;
	dup_reader_t rd;
	uint32 num_bins;
	uint32 nthreads;
	uint32 t;
	double bin_max;

	uint32 *array_size;
	relation_t *tmp_rel;

	uint32 batch = 1024 * obj->num_threads;
	uint32 num_relations_read;
	int32 *status;

	if (batch < 1) batch = 1;

	/* per thread variables */
	my_curr_relation = (uint64 *)xcalloc((size_t)batch, sizeof(uint64));
	buf = (char *)malloc(batch * LINE_BUF_SIZE * sizeof(char));

	/* the second pair is what the reader fills while the team works
	   on the first */

	buf2 = (char *)malloc(batch * LINE_BUF_SIZE * sizeof(char));
	ord2 = (uint64 *)malloc(batch * sizeof(uint64));
	buf_cur = buf;
	buf_nxt = buf2;
	ord_cur = my_curr_relation;
	ord_nxt = ord2;
	scratch = (mpz_t *)malloc(batch * sizeof(mpz_t));
	array_size = (uint32 *)malloc(batch * sizeof(uint32));
	tmp_rel = (relation_t *)malloc(batch * sizeof(relation_t));
	status = (int32 *)malloc(batch * sizeof(int32));

	for (i = 0; i < batch; i++) {
		tmp_rel[i].factors = (uint8 *)malloc(COMPRESSED_P_MAX_SIZE * sizeof(uint8));
		mpz_init(scratch[i]);
	}

	logprintf(obj, "commencing duplicate removal, pass 1\n");

	savefile_open(savefile, SAVEFILE_READ);
	get_filter_tmp_name(obj, buf, LINE_BUF_SIZE, ".br");
	bad_relation_fp = fopen(buf, "wb");
	if (bad_relation_fp == NULL) {
		logprintf(obj, "error: dup1 can't open relation file\n");
		exit(-1);
	}
	get_filter_tmp_name(obj, buf, LINE_BUF_SIZE, ".hc");
	collision_fp = fopen(buf, "wb");
	if (collision_fp == NULL) {
		logprintf(obj, "error: dup1 can't open collision file\n");
		exit(-1);
	}

	/* figure out how large the stage 1 hashtable should be.
	   We want there to be many more bins in the hashtable than
	   relations in the savefile, but it takes too long to
	   actually count the relations. So we estimate the average
	   relation size and then the number of relations */

	log2_hashtable1_size = 28;
	if (rel_size > 0.0) {
		est_num_rels = get_file_size(savefile->name) / rel_size;
		log2_hashtable1_size = log(est_num_rels * 10.0) / M_LN2 + 0.5;
	}
	if (log2_hashtable1_size < 25)
		log2_hashtable1_size = 25;
	if (log2_hashtable1_size > 63)
	 	log2_hashtable1_size = 63;
	/* printf("log2_hashtable1_size = %u\n", log2_hashtable1_size); */
	hashtable = (uint8 *)xcalloc((uint64)1 <<
				(log2_hashtable1_size - 3), sizeof(uint8));
	num_bins = (uint32)1 << (32 - LOG2_BIN_SIZE);
	prime_bins = (uint64 *)xcalloc((size_t)num_bins, sizeof(uint64));

	/* Walking each relation's factors to build the prime histogram is
	   the largest part of this pass's serial work -- about 1.9 billion
	   factors on a 138M relation dataset, and each one that qualifies
	   also posts a random write into the 16 MB free-relation bitmap.
	   None of it is order-dependent, so it runs in parallel with a set
	   of per-thread histograms. Those are allocated once for the whole
	   pass and merged at the end: doing it per batch would cost more in
	   reduction than the walk itself. */

#ifdef HAVE_OMP
	nthreads = (uint32)omp_get_max_threads();
#else
	nthreads = 1;
#endif
	if (nthreads < 1)
		nthreads = 1;
	thread_bins = (uint64 *)xcalloc((size_t)nthreads * num_bins,
					sizeof(uint64));

	/* If the coordinates of every relation fit comfortably in memory,
	   keep them, so that pass 2 can skip re-reading the savefile. That
	   read is the dominant cost of pass 2 whenever the relations live on
	   a network filesystem. The estimate is deliberately generous -- if
	   it is still too low the cache is dropped mid-pass and pass 2 falls
	   back to reading the file. Budget half of RAM, which leaves room for
	   pass 2's own bit table and collision hashtable. */

	memset(&ab_cache, 0, sizeof(ab_cache));
	if (est_num_rels > 0.0 && ram_size > 0) {
		/* tolerate a 50% underestimate of the relation count; the
		   budget below is checked against this cap, so the memory
		   promise holds even if the estimate was low */

		uint64 cache_limit = (uint64)(est_num_rels * 1.5) + 1024;
		uint64 bits_bytes = (uint64)1 << (log2_hashtable1_size - 3);

		cache_bytes = cache_limit * sizeof(abpair_t) + bits_bytes;
		if (cache_bytes <= ram_size / 2) {
			collision_bits = (uint8 *)calloc((size_t)bits_bytes, 1);
			if (collision_bits != NULL) {
				dup_ab_cache_init(&ab_cache, cache_limit);
				if (!ab_cache.active) {
					free(collision_bits);
					collision_bits = NULL;
				}
			}
		}
	}

	/* set up the structures for tracking free relations */

	free_relation_bits = (uint8 *)xcalloc(
				((size_t)(FREE_RELATION_LIMIT/2) + 7) / 8,
				(size_t)1);
	num_free_relations = 0;
	num_free_relations_alloc = 5000;
	free_relations = (uint32 *)xmalloc(num_free_relations_alloc *
						sizeof(uint32));

	curr_relation = UINT64_MAX;
	num_relations = 0;
	num_collisions = 0;
	num_composite = 0;
	num_malformed = 0;

	rd.savefile = savefile;
	rd.max_relations = max_relations;
	rd.batch = batch;
	rd.curr_relation = curr_relation;
	rd.done = 0;

	/* prime the pipeline with the first batch */

	dup_read_batch(&rd, buf_cur, ord_cur, &num_relations_read);

	while (num_relations_read > 0) {
		uint32 next_count = 0;

#pragma omp parallel
		{
			/* one thread runs ahead into the other buffer while the
			   rest parse and tally this batch; the barrier ending
			   each worksharing loop keeps them in step */

#pragma omp single nowait
				dup_read_batch(&rd, buf_nxt, ord_nxt, &next_count);

#pragma omp for schedule(dynamic, 64)
			for (i = 0; i < num_relations_read; i++) {
				char *buf_i = buf_cur + i * LINE_BUF_SIZE;
				status[i] = nfs_read_relation(buf_i, fb,
						&tmp_rel[i], &array_size[i], 1,
						scratch[i], 1);
			}

#pragma omp for schedule(dynamic, 64)
			for (i = 0; i < num_relations_read; i++) {
				uint32 num_r, num_a, k, asize = 0;
				uint64 *my_bins;

				if (status[i] != 0 || tmp_rel[i].b == 0)
					continue;

#ifdef HAVE_OMP
				my_bins = thread_bins +
					(size_t)omp_get_thread_num() * num_bins;
#else
				my_bins = thread_bins;
#endif
				num_r = tmp_rel[i].num_factors_r;
				num_a = tmp_rel[i].num_factors_a;

				for (k = 0; k < num_r + num_a; k++) {
					uint64 p = decompress_p(tmp_rel[i].factors,
								&asize);

					if (p >= ((uint64)1 << 32))
						continue;

					my_bins[p / BIN_SIZE]++;

					if (k >= num_r && p > MAX_PACKED_PRIME &&
							p < FREE_RELATION_LIMIT) {
						uint64 h = p / 2;
#pragma omp atomic update
						free_relation_bits[h / 8] |=
								hashmask[h % 8];
					}
				}
			}
		}

		for (i = 0; i < num_relations_read; i++) {
			uint64 hashval;
			uint64 blob[2];

			/* keep the coordinates for pass 2. Relations that
			   failed to parse get a placeholder: they are listed
			   in the .br file and pass 2 skips them without
			   looking at the cache */

			if (ab_cache.active) {
				dup_ab_cache_store(&ab_cache, ord_cur[i],
						status[i] ? 0 : tmp_rel[i].a,
						status[i] ? 0 : tmp_rel[i].b);
			}

			if (ord_cur[i] > 0 && (ord_cur[i] % 10000000 == 0)) {
				printf("read %" PRIu64 "M relations\n", curr_relation / 1000000);
			}
			if (status[i] != 0) {

				/* save the line number of bad relations (hopefully
			   		there are very few of them) */

				dup_write_u64(obj, bad_relation_fp, ord_cur[i],
					"bad relation list");
				if (status[i] == -98)
					num_composite++;
				else if (status[i] == -97)
					num_malformed++;
				else
				logprintf(obj, "error %d reading relation %" PRIu64 "\n",
						status[i], ord_cur[i]);
			} else {

				/* relation is good; find the value to which it
				hashes. Note that only the bottom 35 bits of 'a'
				and the bottom 29 bits of 'b' figure into the hash,
				so that spurious hash collisions are possible
				(though highly unlikely) */

				num_relations++;
				blob[0] = tmp_rel[i].a;
				blob[1] = (uint64)tmp_rel[i].b;

				hashval = (rrxmrrxmsx_0(blob[0]) ^ rrxmrrxmsx_0(blob[1])) >>
                                        (64 - log2_hashtable1_size);

				/* save the hash bucket if there's a collision. We
				don't need to save any more collisions to this bucket,
				but future duplicates could cause the same bucket to
				be saved more than once. We can cut the number of
				redundant bucket reports in half by resetting the
				bit to zero */

				if (hashtable[hashval / 8] & hashmask[hashval % 8]) {
					dup_write_u64(obj, collision_fp, hashval,
						"collision list");
					num_collisions++;
					hashtable[hashval / 8] &= ~hashmask[hashval % 8];

					/* the .hc file is still written, so that
					   pass 2 can fall back to it; this is the
					   same set of bins, kept in memory */

					if (collision_bits != NULL) {
						collision_bits[hashval / 8] |=
							hashmask[hashval % 8];
					}
				}
				else {
					hashtable[hashval / 8] |= hashmask[hashval % 8];
				}

				if (tmp_rel[i].b == 0) {
					/* remember any free relations that are found */

					if (num_free_relations == num_free_relations_alloc) {
						num_free_relations_alloc *= 2;
						free_relations = (uint32 *)xrealloc(
								free_relations,
								num_free_relations_alloc *
								sizeof(uint32));
					}
					free_relations[num_free_relations++] =
								(uint32)(tmp_rel[i].a);
				}
				/* the factors are tallied by the parallel pass below */
			}
		}

		/* the lookahead already filled the other buffer */

		tmp_buf = buf_cur; buf_cur = buf_nxt; buf_nxt = tmp_buf;
		tmp_ord = ord_cur; ord_cur = ord_nxt; ord_nxt = tmp_ord;
		num_relations_read = next_count;
	}
	curr_relation = rd.curr_relation;

	/* fold the per-thread histograms together, once */

	for (t = 0; t < nthreads; t++) {
		uint64 *src = thread_bins + (size_t)t * num_bins;
		uint32 bin;

		for (bin = 0; bin < num_bins; bin++)
			prime_bins[bin] += src[bin];
	}
	free(thread_bins);

	free(hashtable);
	savefile_close(savefile);
	dup_close_output(obj, bad_relation_fp, "bad relation list");
	dup_close_output(obj, collision_fp, "collision list");

	if (num_composite > 0)
		logprintf(obj, "skipped %" PRIu64 " relations with composite factors\n",
				num_composite);
	if (num_malformed > 0)
		logprintf(obj, "skipped %" PRIu64 " malformed relations\n",
				num_malformed);
	logprintf(obj, "found %" PRIu64 " hash collisions in %" PRIu64
			" relations\n", num_collisions, num_relations);
	if (ab_cache.active) {
		logprintf(obj, "cached %" PRIu64 " relation coordinates for pass 2 (%.1f MB)\n",
				curr_relation + 1,
				(double)((curr_relation + 1) * sizeof(abpair_t) +
				((uint64)1 << (log2_hashtable1_size - 3))) / 1048576);
	}

	if (max_relations == 0 || max_relations > curr_relation + 1) {

		/* cancel out any free relations that are
		   already present in the dataset, then add
		   free relations that remain */

		for (i = 0; i < num_free_relations; i++) {
			uint32 p = free_relations[i];

			if (p < FREE_RELATION_LIMIT) {
				p = p / 2;
				free_relation_bits[p / 8] &= ~hashmask[p % 8];
			}
		}
		{
			uint32 *free_primes = NULL;
			uint32 nfree;

			/* these are appended to the savefile, so pass 2 will
			   see them as relations following everything pass 1
			   read. The cache has to be extended to match, or the
			   cached pass 2 would silently ignore them */

			nfree = add_free_relations(obj, fb, free_relation_bits,
					(ab_cache.active ||
					 savefile->staged_name != NULL) ?
						&free_primes : NULL);
			num_relations += nfree;

			if (free_primes != NULL) {
				uint32 k;
				for (k = 0; k < nfree; k++) {
					dup_ab_cache_store(&ab_cache,
						curr_relation + 1 + k,
						(int64)free_primes[k], 0);
				}
			}

			/* add_free_relations() appended to the original
			   savefile. A staged copy is read by every later
			   pass, so it has to receive the same lines or
			   those passes would not see the free relations */

			if (nfree > 0 && savefile->staged_name != NULL) {
				FILE *fp = fopen(savefile->staged_name, "ab");
				uint32 k;

				if (fp == NULL) {
					logprintf(obj, "error: cannot mirror free "
						"relations into the staged savefile\n");
					exit(-1);
				}
				for (k = 0; k < nfree; k++)
					fprintf(fp, "%u,0:\n", free_primes[k]);
				if (fclose(fp) != 0) {
					logprintf(obj, "error: write failed mirroring "
						"free relations\n");
					exit(-1);
				}
			}
			free(free_primes);
			num_free_added = nfree;
		}
	}
	free(free_relations);
	free(free_relation_bits);

	/* free per thread variables */

	for (i = 0; i < batch; i++) {
		free(tmp_rel[i].factors);
		mpz_clear(scratch[i]);
	}

	free(my_curr_relation);
	free(ord2);
	free(buf2);
	free(scratch);
	free(array_size);
	free(tmp_rel);
	free(status);

	if (num_collisions == 0) {

		/* no duplicates; no second pass is necessary */

		char buf2[256];
		get_filter_tmp_name(obj, buf, LINE_BUF_SIZE, ".hc");
		remove(buf);
		get_filter_tmp_name(obj, buf, LINE_BUF_SIZE, ".br");
		get_filter_tmp_name(obj, buf2, sizeof(buf2), ".d");
		if (rename(buf, buf2) != 0) {
			logprintf(obj, "error: dup1 can't rename outfile\n");
			exit(-1);
		}
	}
	else {
		num_relations = purge_duplicates_pass2(obj,
					log2_hashtable1_size,
					max_relations, &ab_cache,
					collision_bits,
					curr_relation + 1 + num_free_added);
	}

	dup_ab_cache_free(&ab_cache);
	free(collision_bits);
	free(buf);

	/* the large prime cutoff for the rest of the filtering
	   process should be chosen here. We don't want the bound
	   to depend on an arbitrarily chosen factor base, since
	   that bound may be too large or much too small. The former
	   would make filtering take too long, and the latter
	   could make filtering impossible.

	   Conceptually, we want the bound to be the point below
	   which large primes appear too often in the dataset. */

	i = 1 << (32 - LOG2_BIN_SIZE);
	bin_max = (double)BIN_SIZE * i /
			log((double)BIN_SIZE * i);
	for (i--; i > 2; i--) {
		double bin_min = (double)BIN_SIZE * i /
				log((double)BIN_SIZE * i);
		double hits_per_prime = (double)prime_bins[i] /
						(bin_max - bin_min);
		if (hits_per_prime > TARGET_HITS_PER_PRIME)
			break;
		bin_max = bin_min;
	}

	free(prime_bins);
	*num_relations_out = num_relations;
	return BIN_SIZE * (i + 0.5);
}
