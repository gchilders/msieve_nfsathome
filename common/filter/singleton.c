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

/* target_excess is 32-bit, but removing ignored ideals can add billions. */
static void add_target_excess(msieve_obj *obj, filter_t *filter, uint32 delta) {
	uint64 target = (uint64)filter->target_excess + delta;
	if (target > UINT32_MAX) {
		logprintf(obj, "error: filtering target excess exceeds 32-bit capacity\n");
		exit(-1);
	}
	filter->target_excess = (uint32)target;
}


#define LP32_IO_BUFFER_BYTES (8 * 1024 * 1024)

typedef struct {
    FILE *fp;
    uint8 *buf;
    size_t pos;
    size_t valid;
} lp32_reader_t;

typedef struct {
    FILE *fp;
    uint8 *buf;
    size_t used;
    int failed;
} lp32_writer_t;

static void lp32_reader_init(lp32_reader_t *r, FILE *fp) {
    r->fp = fp;
    r->buf = (uint8 *)xmalloc(LP32_IO_BUFFER_BYTES);
    r->pos = r->valid = 0;
}

static void lp32_reader_free(lp32_reader_t *r) {
    free(r->buf);
    memset(r, 0, sizeof(*r));
}

static int lp32_reader_rewind(lp32_reader_t *r) {
    if (fseeko(r->fp, 0, SEEK_SET) != 0)
        return -1;
    clearerr(r->fp);
    r->pos = r->valid = 0;
    return 0;
}

/* Return 1 on success, 0 only for a clean EOF before reading any byte when
   allow_eof is true, and -1 for a partial record or I/O error. */
static int lp32_read_exact(lp32_reader_t *r, void *dst, size_t bytes,
                           int allow_eof) {
    uint8 *out = (uint8 *)dst;
    size_t done = 0;
    while (done < bytes) {
        size_t avail;
        if (r->pos == r->valid) {
            if (feof(r->fp))
                return (done == 0 && allow_eof) ? 0 : -1;
            r->valid = fread(r->buf, 1, LP32_IO_BUFFER_BYTES, r->fp);
            r->pos = 0;
            if (ferror(r->fp))
                return -1;
            if (r->valid == 0)
                return (done == 0 && allow_eof) ? 0 : -1;
        }
        avail = r->valid - r->pos;
        if (avail > bytes - done)
            avail = bytes - done;
        memcpy(out + done, r->buf + r->pos, avail);
        r->pos += avail;
        done += avail;
    }
    return 1;
}

static int lp32_read_record(lp32_reader_t *r, relation_ideal_t *rec,
                            uint32 num_ideals, int allow_eof) {
    size_t header_words = (sizeof(relation_ideal_t) -
            sizeof(rec->ideal_list)) / sizeof(uint32);
    size_t header_bytes = header_words * sizeof(uint32);
    int rc = lp32_read_exact(r, rec, header_bytes, allow_eof);
    uint32 j;
    if (rc != 1)
        return rc;
    if (rec->ideal_count > TEMP_FACTOR_LIST_SIZE)
        return -1;
    if (lp32_read_exact(r, rec->ideal_list,
            (size_t)rec->ideal_count * sizeof(uint32), 0) != 1)
        return -1;
    for (j = 0; j < rec->ideal_count; j++) {
        if (rec->ideal_list[j] >= num_ideals)
            return -1;
    }
    return 1;
}

static void lp32_writer_init(lp32_writer_t *w, FILE *fp) {
    w->fp = fp;
    w->buf = (uint8 *)xmalloc(LP32_IO_BUFFER_BYTES);
    w->used = 0;
    w->failed = 0;
}

static int lp32_writer_flush(lp32_writer_t *w) {
    if (w->failed)
        return -1;
    if (w->used && fwrite(w->buf, 1, w->used, w->fp) != w->used) {
        w->failed = 1;
        return -1;
    }
    w->used = 0;
    return 0;
}

static int lp32_write(lp32_writer_t *w, const void *src, size_t bytes) {
    const uint8 *in = (const uint8 *)src;
    while (bytes) {
        size_t room = LP32_IO_BUFFER_BYTES - w->used;
        size_t take = bytes < room ? bytes : room;
        if (take) {
            memcpy(w->buf + w->used, in, take);
            w->used += take;
            in += take;
            bytes -= take;
        }
        if (w->used == LP32_IO_BUFFER_BYTES && lp32_writer_flush(w) != 0)
            return -1;
    }
    return 0;
}

static int lp32_write_record(lp32_writer_t *w, relation_ideal_t *rec) {
    size_t header_words = (sizeof(relation_ideal_t) -
            sizeof(rec->ideal_list)) / sizeof(uint32);
    size_t bytes = (header_words + rec->ideal_count) * sizeof(uint32);
    return lp32_write(w, rec, bytes);
}

static int lp32_writer_finish(lp32_writer_t *w) {
    int rc = lp32_writer_flush(w);
    free(w->buf);
    w->buf = NULL;
    if (rc != 0 || fflush(w->fp) != 0 || ferror(w->fp))
        return -1;
    return 0;
}

#define LP32_ALIVE_TEST(a,i) ((a)[(size_t)(i) >> 3] & (uint8)(1U << ((i) & 7)))
#define LP32_ALIVE_CLEAR(a,i) ((a)[(size_t)(i) >> 3] &= (uint8)~(1U << ((i) & 7)))

/*--------------------------------------------------------------------*/
static void filter_read_lp_file_1pass(msieve_obj *obj,
                filter_t *filter,
                uint32 max_ideal_weight) {

    uint32 i, j, k;
    FILE *fp;
    char buf[256];
    uint32 num_relations = filter->num_relations;
    uint32 num_ideals = filter->num_ideals;
    uint32 *counts;
    relation_ideal_t *r, *r_old, *r_next;
    uint8 *end;
    size_t header_words = (sizeof(relation_ideal_t) -
            sizeof(((relation_ideal_t *)0)->ideal_list)) / sizeof(uint32);
    size_t header_bytes = header_words * sizeof(uint32);

    logprintf(obj, "reading all ideals from disk\n");

    sprintf(buf, "%s.lp", obj->savefile.name);
    fp = fopen(buf, "rb");
    if (fp == NULL) {
        logprintf(obj, "error: can't open LP file\n");
        exit(-1);
    }

    if (filter->lp_file_size > (uint64)SIZE_MAX) {
        logprintf(obj, "error: LP file is too large for this address space\n");
        exit(-1);
    }
    filter->relation_array = (relation_ideal_t *)xmalloc(
                    (size_t)MAX(filter->lp_file_size, (uint64)1));
    filter->relation_ptr = (relation_ideal_t **) xmalloc(
                    (size_t)num_relations * sizeof(relation_ideal_t *));
    if (fread(filter->relation_array, 1, (size_t)filter->lp_file_size, fp) !=
            (size_t)filter->lp_file_size || ferror(fp)) {
        logprintf(obj, "error: truncated LP file\n");
        exit(-1);
    }
    if (fgetc(fp) != EOF || ferror(fp)) {
        logprintf(obj, "error: LP file size changed while reading\n");
        exit(-1);
    }
    if (fclose(fp) != 0) {
        logprintf(obj, "error: can't close LP file\n");
        exit(-1);
    }
    logprintf(obj, "memory use: %.1f MB\n",
            ((double)filter->lp_file_size +
            (double)num_relations * sizeof(relation_ideal_t *))/ 1048576);

    counts = (uint32 *)xcalloc((size_t)num_ideals, sizeof(uint32));
    r = filter->relation_array;
    end = (uint8 *)filter->relation_array + (size_t)filter->lp_file_size;
    for (i = 0; i < num_relations; i++) {
        size_t bytes;
        if ((size_t)(end - (uint8 *)r) < header_bytes ||
            r->ideal_count > TEMP_FACTOR_LIST_SIZE) {
            logprintf(obj, "error: corrupt LP relation header at relation %u\n", i);
            exit(-1);
        }
        bytes = (header_words + r->ideal_count) * sizeof(uint32);
        if ((size_t)(end - (uint8 *)r) < bytes) {
            logprintf(obj, "error: truncated LP relation %u\n", i);
            exit(-1);
        }
        filter->relation_ptr[i] = r;
        for (j = 0; j < r->ideal_count; j++) {
            if (r->ideal_list[j] >= num_ideals) {
                logprintf(obj, "error: invalid ideal ID in LP relation %u\n", i);
                exit(-1);
            }
            counts[r->ideal_list[j]]++;
        }
        r = next_relation_ptr(r);
    }
    if ((uint8 *)r != end) {
        logprintf(obj, "error: LP file has trailing or mismatched relation data\n");
        exit(-1);
    }

    for (i = j = 0; i < num_ideals; i++) {
        if (counts[i] == 0 || counts[i] > max_ideal_weight)
            counts[i] = (uint32)(-1);
        else
            counts[i] = j++;
    }
    add_target_excess(obj, filter, i - j);
    filter->num_ideals = j;

    if (i != j) {
        logprintf(obj, "keeping %u ideals with weight <= %u, "
                "target excess is %u\n",
                j, max_ideal_weight,
                filter->target_excess);

        r = r_old = filter->relation_array;
        for (i = 0; i < num_relations; i++) {
            uint32 ideal_count = r->ideal_count;
            filter->relation_ptr[i] = r_old;
            r_next = next_relation_ptr(r);
            r_old->rel_index = r->rel_index;
            r_old->gf2_factors = r->gf2_factors;
            r_old->connected = 0;

            for (j = k = 0; j < ideal_count; j++) {
                uint32 ideal = counts[r->ideal_list[j]];
                if (ideal != (uint32)(-1))
                    r_old->ideal_list[k++] = ideal;
            }
            r_old->ideal_count = (uint8)k;
            r_old->gf2_factors += (uint8)(ideal_count - k);
            r_old = next_relation_ptr(r_old);
            r = r_next;
        }
        /* Do not shrink relation_array here: relation_ptr[] points into it,
           and a moving realloc would invalidate every pointer. The unused tail
           is released with relation_array after 2-way merge. */
    }

    free(counts);
}

/*--------------------------------------------------------------------*/
void filter_read_lp_file(msieve_obj *obj, filter_t *filter,
                uint32 max_ideal_weight) {
    uint32 i, j, k;
    FILE *fp;
    char buf[256];
    relation_ideal_t tmp;
    relation_ideal_t *relation_array;
    relation_ideal_t *r;
    uint32 *counts;
    uint32 num_relations = filter->num_relations;
    uint32 num_ideals = filter->num_ideals;
    uint64 out_words = 0;
    size_t header_words = (sizeof(relation_ideal_t) -
            sizeof(tmp.ideal_list)) / sizeof(uint32);
    lp32_reader_t reader;
    size_t mem_use;

    if (max_ideal_weight == 0) {
        filter_read_lp_file_1pass(obj, filter, 200);
        filter_purge_singletons_core(obj, filter);
        return;
    }

    logprintf(obj, "reading large ideals from disk\n");

    sprintf(buf, "%s.lp", obj->savefile.name);
    fp = fopen(buf, "rb");
    if (fp == NULL) {
        logprintf(obj, "error: singleton2 can't open LP file\n");
        exit(-1);
    }
    lp32_reader_init(&reader, fp);
    counts = (uint32 *)xcalloc((size_t)num_ideals, sizeof(uint32));

    for (i = 0; i < num_relations; i++) {
        if (lp32_read_record(&reader, &tmp, num_ideals, 0) != 1) {
            logprintf(obj, "error: truncated or corrupt LP file at relation %u\n", i);
            exit(-1);
        }
        for (j = 0; j < tmp.ideal_count; j++)
            counts[tmp.ideal_list[j]]++;
    }
    if (lp32_read_record(&reader, &tmp, num_ideals, 1) != 0) {
        logprintf(obj, "error: LP file contains trailing relation data\n");
        exit(-1);
    }

    for (i = j = 0; i < num_ideals; i++) {
        if (counts[i] <= max_ideal_weight)
            counts[i] = j++;
        else
            counts[i] = (uint32)(-1);
    }
    add_target_excess(obj, filter, i - j);
    filter->num_ideals = j;
    logprintf(obj, "keeping %u ideals with weight <= %u, "
            "target excess is %u\n", j, max_ideal_weight,
            filter->target_excess);

    /* A sizing pass avoids a giant grow/realloc loop and guarantees that
       relation_ptr[] is built only after the packed array reaches its final
       address. */
    if (lp32_reader_rewind(&reader) != 0) {
        logprintf(obj, "error: can't rewind LP file\n");
        exit(-1);
    }
    for (i = 0; i < num_relations; i++) {
        uint32 keep = 0;
        if (lp32_read_record(&reader, &tmp, num_ideals, 0) != 1) {
            logprintf(obj, "error: truncated or corrupt LP file during sizing\n");
            exit(-1);
        }
        for (j = 0; j < tmp.ideal_count; j++)
            if (counts[tmp.ideal_list[j]] != (uint32)(-1))
                keep++;
        out_words += header_words + keep;
    }
    if (out_words > SIZE_MAX / sizeof(uint32)) {
        logprintf(obj, "error: packed relation array exceeds address space\n");
        exit(-1);
    }

    relation_array = (relation_ideal_t *)xmalloc((size_t)MAX(out_words, (uint64)1) * sizeof(uint32));
    filter->relation_ptr = (relation_ideal_t **)xmalloc(
            (size_t)num_relations * sizeof(relation_ideal_t *));

    if (lp32_reader_rewind(&reader) != 0) {
        logprintf(obj, "error: can't rewind LP file\n");
        exit(-1);
    }
    r = relation_array;
    for (i = 0; i < num_relations; i++) {
        uint32 original_count;
        if (lp32_read_record(&reader, &tmp, num_ideals, 0) != 1) {
            logprintf(obj, "error: truncated or corrupt LP file during packing\n");
            exit(-1);
        }
        original_count = tmp.ideal_count;
        filter->relation_ptr[i] = r;
        r->rel_index = tmp.rel_index;
        r->gf2_factors = tmp.gf2_factors;
        r->connected = 0;
        for (j = k = 0; j < original_count; j++) {
            uint32 mapped = counts[tmp.ideal_list[j]];
            if (mapped != (uint32)(-1))
                r->ideal_list[k++] = mapped;
        }
        r->ideal_count = (uint8)k;
        r->gf2_factors += (uint8)(original_count - k);
        r = next_relation_ptr(r);
    }

    filter->relation_array = relation_array;
    free(counts);
    lp32_reader_free(&reader);
    if (fclose(fp) != 0) {
        logprintf(obj, "error: can't close LP file\n");
        exit(-1);
    }
    mem_use = (size_t)out_words * sizeof(uint32) +
            (size_t)num_relations * sizeof(relation_ideal_t *);
    logprintf(obj, "memory use: %.1f MB\n", (double)mem_use / 1048576);

    filter_purge_singletons_core(obj, filter);
}

/*--------------------------------------------------------------------*/
void filter_purge_lp_singletons(msieve_obj *obj,
                filter_t *filter,
                uint64 ram_size) {

    uint32 i, j, m;
    FILE *in_fp;
    FILE *out_fp;
    char buf[256];
    char buf2[256];
    relation_ideal_t tmp;
    uint8 *alive;
    uint32 *counts;
    uint32 num_singletons;
    uint32 num_relations = filter->num_relations;
    uint32 num_ideals = filter->num_ideals;
    uint32 start_relations = filter->num_relations;
    uint32 num_passes = 0;
    uint64 new_file_size;
    lp32_reader_t reader;
    lp32_writer_t writer;

    logprintf(obj, "removing singletons from LP file\n");
    logprintf(obj, "start with %u relations and %u ideals\n",
            num_relations, num_ideals);

    sprintf(buf, "%s.lp", obj->savefile.name);
    in_fp = fopen(buf, "rb");
    if (in_fp == NULL) {
        logprintf(obj, "error: can't open LP file\n");
        exit(-1);
    }
    lp32_reader_init(&reader, in_fp);
    sprintf(buf2, "%s.lp0", obj->savefile.name);
    out_fp = fopen(buf2, "wb");
    if (out_fp == NULL) {
        logprintf(obj, "error: can't open LP output file\n");
        exit(-1);
    }
    lp32_writer_init(&writer, out_fp);

    alive = (uint8 *)xmalloc(((size_t)start_relations + 7) / 8);
    memset(alive, 0xff, ((size_t)start_relations + 7) / 8);
    counts = (uint32 *)xcalloc((size_t)num_ideals, sizeof(uint32));

    for (i = 0; i < start_relations; i++) {
        if (lp32_read_record(&reader, &tmp, num_ideals, 0) != 1) {
            logprintf(obj, "error: truncated or corrupt LP file at relation %u\n", i);
            exit(-1);
        }
        for (j = 0; j < tmp.ideal_count; j++)
            counts[tmp.ideal_list[j]]++;
    }

    do {
        new_file_size = 0;
        num_singletons = 0;
        if (lp32_reader_rewind(&reader) != 0) {
            logprintf(obj, "error: can't rewind LP file\n");
            exit(-1);
        }
        for (i = 0; i < start_relations; i++) {
            if (lp32_read_record(&reader, &tmp, num_ideals, 0) != 1) {
                logprintf(obj, "error: truncated or corrupt LP file during singleton pass\n");
                exit(-1);
            }
            if (!LP32_ALIVE_TEST(alive, i))
                continue;
            for (m = 0; m < tmp.ideal_count; m++)
                if (counts[tmp.ideal_list[m]] < 2)
                    break;
            if (m == tmp.ideal_count) {
                size_t header_words = (sizeof(relation_ideal_t) -
                        sizeof(tmp.ideal_list)) / sizeof(uint32);
                new_file_size += (header_words + tmp.ideal_count) * sizeof(uint32);
            }
            else {
                LP32_ALIVE_CLEAR(alive, i);
                num_relations--;
                num_singletons++;
                for (m = 0; m < tmp.ideal_count; m++)
                    counts[tmp.ideal_list[m]]--;
            }
        }
        logprintf(obj, "pass %u: found %u singletons\n",
                ++num_passes, num_singletons);
    } while (num_relations > 2000000 &&
            num_singletons > 500000 &&
            new_file_size >= ram_size / 2);

    for (i = j = 0; i < num_ideals; i++) {
        if (counts[i] != 0)
            counts[i] = j++;
    }
    num_ideals = j;

    if (lp32_reader_rewind(&reader) != 0) {
        logprintf(obj, "error: can't rewind LP file\n");
        exit(-1);
    }
    for (i = 0; i < start_relations; i++) {
        if (lp32_read_record(&reader, &tmp, filter->num_ideals, 0) != 1) {
            logprintf(obj, "error: truncated or corrupt LP file during compaction\n");
            exit(-1);
        }
        if (!LP32_ALIVE_TEST(alive, i))
            continue;
        for (j = 0; j < tmp.ideal_count; j++)
            tmp.ideal_list[j] = counts[tmp.ideal_list[j]];
        if (lp32_write_record(&writer, &tmp) != 0) {
            logprintf(obj, "error: write failed compacting LP file\n");
            exit(-1);
        }
    }

    logprintf(obj, "pruned dataset has %u relations and "
            "%u large ideals\n", num_relations, num_ideals);

    filter->num_relations = num_relations;
    filter->num_ideals = num_ideals;
    filter->relation_array = NULL;
    free(counts);
    free(alive);
    lp32_reader_free(&reader);

    {
        int in_close_rc = fclose(in_fp);
        int writer_rc = lp32_writer_finish(&writer);
        int out_close_rc = fclose(out_fp);
        if (in_close_rc != 0 || writer_rc != 0 || out_close_rc != 0) {
            logprintf(obj, "error: can't finalize compacted LP file\n");
            exit(-1);
        }
    }
    if (remove(buf) != 0) {
        logprintf(obj, "error: can't delete LP file\n");
        exit(-1);
    }
    if (rename(buf2, buf) != 0) {
        logprintf(obj, "error: can't rename LP output file\n");
        exit(-1);
    }
    filter->lp_file_size = get_file_size(buf);
}

/*--------------------------------------------------------------------*/
void filter_purge_singletons_core(msieve_obj *obj,
				filter_t *filter) {

	/* main routine for performing in-memory singleton
	   removal. We iterate until there are no more singletons */

	uint32 i, j;
	uint32 *freqtable;
	relation_ideal_t *relation_array;
	relation_ideal_t **relation_ptr;
	relation_ideal_t *curr_relation;
	relation_ideal_t *old_relation;
	uint32 orig_num_ideals;
	uint32 num_passes;
	uint32 num_relations;
	uint32 num_ideals;
	uint32 orig_num_relations, new_num_relations;

	logprintf(obj, "commencing in-memory singleton removal\n");

	num_relations = filter->num_relations;
	orig_num_ideals = num_ideals = filter->num_ideals;
	relation_array = filter->relation_array;
	relation_ptr = filter->relation_ptr;
	freqtable = (uint32 *)xcalloc((size_t)num_ideals, sizeof(uint32));

	/* count the number of times each ideal occurs. Note
	   that since we know the exact number of ideals, we
	   don't need a hashtable to store the counts, just an
	   ordinary random-access array (i.e. a perfect hashtable) */

#pragma omp parallel for private(j)
	for (i = 0; i < num_relations; i++) {
		relation_ideal_t *my_relation = filter->relation_ptr[i];
		my_relation->connected = 0;
		for (j = 0; j < my_relation->ideal_count; j++) {
			uint32 ideal = my_relation->ideal_list[j];
#pragma omp atomic update
			freqtable[ideal]++;
		}
	}

	logprintf(obj, "begin with %u relations and %u unique ideals\n",
					num_relations, num_ideals);

	/* while singletons were found */

	num_passes = 0;
	orig_num_relations = new_num_relations = num_relations;
	do {
		num_relations = new_num_relations;
		new_num_relations = 0;

#pragma omp parallel for private(j) reduction(+:new_num_relations)
		for (i = 0; i < orig_num_relations; i++) {
			relation_ideal_t *my_relation = relation_ptr[i];
			uint32 curr_num_ideals = my_relation->ideal_count;
			uint32 ideal;

			/* check if relation is already marked for deletion */

			if (my_relation->connected == 0) {

				/* check the count of each ideal */

				for (j = 0; j < curr_num_ideals; j++) {
					ideal = my_relation->ideal_list[j];
					if (freqtable[ideal] <= 1) break;
				}

				if (j < curr_num_ideals) {

					/* relation is a singleton; decrement the
				   	count of each of its ideals and skip it */

					for (j = 0; j < curr_num_ideals; j++) {
						ideal = my_relation->ideal_list[j];
#pragma omp atomic update
						freqtable[ideal]--;
					}

					my_relation->connected = 1;
				}
				else new_num_relations++;
			}
		}

		num_passes++;
	} while (new_num_relations != num_relations);

	/* Now remove the relations that were marked for deletion */

	curr_relation = old_relation = relation_array;
	new_num_relations = 0;
	for (i = 0; i < orig_num_relations; i++) {
		relation_ideal_t *next_relation;

		/* the ideal count in curr_relation may get
			overwritten when writing old_relation, so
			cache the count and point to the next
			relation now */

		next_relation = next_relation_ptr(curr_relation);

		if (curr_relation->connected == 0) {
			filter->relation_ptr[new_num_relations] = old_relation;
			old_relation->rel_index =
					curr_relation->rel_index;
			old_relation->gf2_factors =
					curr_relation->gf2_factors;
			old_relation->ideal_count =
					curr_relation->ideal_count;
			for (j = 0; j < curr_relation->ideal_count; j++) {
				old_relation->ideal_list[j] =
					curr_relation->ideal_list[j];
			}
			new_num_relations++;
			old_relation = next_relation_ptr(old_relation);
		}

		curr_relation = next_relation;
	}

	/* find the ideal that occurs in the most
	   relations, and renumber the ideals to ignore
	   any that have a count of zero */

	num_ideals = 0;
	for (i = j = 0; i < orig_num_ideals; i++) {
		if (freqtable[i]) {
			j = MAX(j, freqtable[i]);
			freqtable[i] = num_ideals++;
		}
	}

	logprintf(obj, "reduce to %u relations and %u ideals in %u passes\n",
				num_relations, num_ideals, num_passes);
	logprintf(obj, "max relations containing the same ideal: %u\n", j);

	/* save the current state */

	filter->max_ideal_weight = j;
	filter->num_relations = num_relations;
	filter->num_ideals = num_ideals;

#pragma omp parallel for private(j)
	for (i = 0; i < num_relations; i++) {
		relation_ideal_t *my_relation = filter->relation_ptr[i];
		for (j = 0; j < my_relation->ideal_count; j++) {
			uint32 ideal = my_relation->ideal_list[j];
			my_relation->ideal_list[j] = freqtable[ideal];
		}
	}

	free(freqtable);

	{
		size_t relation_bytes = (size_t)((uint8 *)old_relation -
				(uint8 *)relation_array);

		/* relation_array contains variable-size packed records. Measure the
		   used span in bytes rather than subtracting relation_ideal_t pointers;
		   pointer subtraction would incorrectly scale by sizeof(relation_ideal_t)
		   and is not valid for these byte-packed records. */
		if (num_relations == 0) {
			free(relation_array);
			free(relation_ptr);
			filter->relation_array = NULL;
			filter->relation_ptr = NULL;
			return;
		}
		if (relation_bytes == 0) {
			logprintf(obj, "error: singleton compaction produced an empty "
					"relation array with %u survivors\n", num_relations);
			exit(-1);
		}

		filter->relation_array =
				(relation_ideal_t *)xrealloc(relation_array, relation_bytes);
		filter->relation_ptr =
				(relation_ideal_t **)xrealloc(relation_ptr,
					(size_t)num_relations * sizeof(relation_ideal_t *));

		if (filter->relation_array != relation_array) {
			/* the realloc moved the packed array; rebuild every interior
			   pointer from the packed records at their new base address */
			curr_relation = filter->relation_array;
			for (i = 0; i < num_relations; i++) {
				filter->relation_ptr[i] = curr_relation;
				curr_relation = next_relation_ptr(curr_relation);
			}
		}
	}
}
