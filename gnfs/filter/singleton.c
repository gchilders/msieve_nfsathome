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

/* The common filtering code deliberately uses 32-bit relation and ideal
   IDs. For very large NFS inputs, keep 64-bit IDs only in this initial,
   disk-based stage and compact them before handing the LP file to the
   common filter. */

#define NFS_LARGE_IO_BUFSIZE (8u * 1024u * 1024u)

typedef struct {
	FILE *fp;
	uint8 *buf;
	size_t pos;
	size_t len;
	int error;
} nfs_buffered_reader_t;

typedef struct {
	FILE *fp;
	uint8 *buf;
	size_t pos;
	int error;
} nfs_buffered_writer_t;

static void nfs_buffered_reader_init(nfs_buffered_reader_t *r, FILE *fp) {
	memset(r, 0, sizeof(*r));
	r->fp = fp;
	r->buf = (uint8 *)xmalloc(NFS_LARGE_IO_BUFSIZE);
}

/* Return 1 on success, 0 only for clean EOF before any requested byte,
   and -1 for a short/truncated read or I/O error. */
static int nfs_buffered_read(nfs_buffered_reader_t *r, void *dst,
			     size_t len, uint32 allow_eof) {
	uint8 *out = (uint8 *)dst;
	size_t copied = 0;
	while (copied < len) {
		size_t avail;
		if (r->pos == r->len) {
			if (r->error)
				return -1;
			if (feof(r->fp))
				return (copied == 0 && allow_eof) ? 0 : -1;
			r->len = fread(r->buf, 1, NFS_LARGE_IO_BUFSIZE, r->fp);
			r->pos = 0;
			if (ferror(r->fp)) {
				r->error = 1;
				return -1;
			}
			if (r->len == 0) {
				if (copied == 0 && allow_eof)
					return 0;
				return -1;
			}
		}
		avail = r->len - r->pos;
		if (avail > len - copied)
			avail = len - copied;
		memcpy(out + copied, r->buf + r->pos, avail);
		r->pos += avail;
		copied += avail;
	}
	return 1;
}

static int nfs_buffered_reader_rewind(nfs_buffered_reader_t *r) {
	if (fseeko(r->fp, 0, SEEK_SET) != 0)
		return -1;
	r->pos = r->len = 0;
	r->error = 0;
	clearerr(r->fp);
	return 0;
}

static void nfs_buffered_reader_free(nfs_buffered_reader_t *r) {
	free(r->buf);
	memset(r, 0, sizeof(*r));
}

static void nfs_buffered_writer_init(nfs_buffered_writer_t *w, FILE *fp) {
	memset(w, 0, sizeof(*w));
	w->fp = fp;
	w->buf = (uint8 *)xmalloc(NFS_LARGE_IO_BUFSIZE);
}

static int nfs_buffered_writer_flush(nfs_buffered_writer_t *w) {
	if (w->error)
		return -1;
	if (w->pos != 0) {
		if (fwrite(w->buf, 1, w->pos, w->fp) != w->pos) {
			w->error = 1;
			return -1;
		}
		w->pos = 0;
	}
	return 0;
}

static int nfs_buffered_write(nfs_buffered_writer_t *w,
			      const void *src, size_t len) {
	const uint8 *in = (const uint8 *)src;
	while (len != 0) {
		size_t room = NFS_LARGE_IO_BUFSIZE - w->pos;
		size_t chunk = len < room ? len : room;
		if (chunk != 0) {
			memcpy(w->buf + w->pos, in, chunk);
			w->pos += chunk;
			in += chunk;
			len -= chunk;
		}
		if (w->pos == NFS_LARGE_IO_BUFSIZE &&
		    nfs_buffered_writer_flush(w) != 0)
			return -1;
	}
	return 0;
}

static int nfs_buffered_writer_finish(nfs_buffered_writer_t *w) {
	int rc = 0;
	if (nfs_buffered_writer_flush(w) != 0 || fflush(w->fp) != 0 ||
	    ferror(w->fp))
		rc = -1;
	free(w->buf);
	w->buf = NULL;
	return rc;
}

/* The temporary LP64 record has a fixed byte representation independent
   of structure padding. Record packing/unpacking occurs entirely in RAM;
   the underlying file sees multi-megabyte block I/O. */
static int read_lp64_record(nfs_buffered_reader_t *r, uint64 *rel_index,
			    uint8 *ideal_count, uint8 *gf2_factors,
			    uint64 *ideal_list) {
	int rc = nfs_buffered_read(r, rel_index, sizeof(uint64), 1);
	if (rc <= 0)
		return rc;
	if (nfs_buffered_read(r, ideal_count, sizeof(uint8), 0) != 1 ||
	    nfs_buffered_read(r, gf2_factors, sizeof(uint8), 0) != 1)
		return -1;
	if (*ideal_count > TEMP_FACTOR_LIST_SIZE)
		return -1;
	if (*ideal_count != 0 &&
	    nfs_buffered_read(r, ideal_list,
			      (size_t)*ideal_count * sizeof(uint64), 0) != 1)
		return -1;
	return 1;
}

static int write_lp64_record(nfs_buffered_writer_t *w, uint64 rel_index,
			     uint8 ideal_count, uint8 gf2_factors,
			     const uint64 *ideal_list) {
	uint8 record[sizeof(uint64) + 2 * sizeof(uint8) +
		     TEMP_FACTOR_LIST_SIZE * sizeof(uint64)];
	size_t len = sizeof(uint64) + 2 * sizeof(uint8) +
		(size_t)ideal_count * sizeof(uint64);
	memcpy(record, &rel_index, sizeof(uint64));
	record[sizeof(uint64)] = ideal_count;
	record[sizeof(uint64) + sizeof(uint8)] = gf2_factors;
	if (ideal_count != 0)
		memcpy(record + sizeof(uint64) + 2 * sizeof(uint8), ideal_list,
			(size_t)ideal_count * sizeof(uint64));
	return nfs_buffered_write(w, record, len);
}


static int nfs_replace_file(const char *src, const char *dst) {
#if defined(WIN32) || defined(_WIN64)
	return MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
	return rename(src, dst);
#endif
}

static int nfs_close_output(FILE *fp) {
	int rc = 0;
	if (fflush(fp) != 0 || ferror(fp))
		rc = -1;
	if (fclose(fp) != 0)
		rc = -1;
	return rc;
}

/* Return 1 for a complete relation number, 0 for clean EOF, and -1 for
   a truncated record or I/O error. */
static int read_relation_number(FILE *fp, uint64 *relation_num) {
	size_t n;
	if (feof(fp)) {
		*relation_num = UINT64_MAX;
		return 0;
	}
	n = fread(relation_num, 1, sizeof(uint64), fp);
	if (n == sizeof(uint64))
		return 1;
	if (n == 0 && feof(fp) && !ferror(fp)) {
		*relation_num = UINT64_MAX;
		return 0;
	}
	return -1;
}

static void read_relation_number_checked(msieve_obj *obj, FILE *fp,
				uint64 *relation_num, const char *what) {
	if (read_relation_number(fp, relation_num) < 0) {
		logprintf(obj, "error: truncated or unreadable %s\n", what);
		exit(-1);
	}
}

/*--------------------------------------------------------------------*/
void nfs_write_lp_file(msieve_obj *obj, factor_base_t *fb,
			filter_t *filter, uint64 max_relations,
			uint32 pass) {

	uint32 i;
	savefile_t *savefile = &obj->savefile;
	FILE *relation_fp;
	FILE *final_fp;
	nfs_buffered_writer_t final_writer;
	char lp_name[256], lp_tmp_name[256];
	char *buf;
	uint64 next_relation;
	uint64 curr_relation;
	uint64 *my_curr_relation;
	uint64 num_relations;
	uint32 needs_rmap = 0;
	nfs_hashtable64_t unique_ideals;
	uint32 *tmp_factor_size;
	relation_t *tmp_relation;
	uint32 have_skip_list = (pass == 0);
	mpz_t *scratch;
	relation_lp_t *tmp_ideal;
	uint64 packed_ideal_ids[TEMP_FACTOR_LIST_SIZE];

	uint32 batch = 1024 * obj->num_threads;
	uint32 num_relations_read;
	int32 *status;

	if (batch < 1)
		batch = 1;

	my_curr_relation = (uint64 *)malloc(batch * sizeof(uint64));
	buf = (char *)malloc(batch * LINE_BUF_SIZE * sizeof(char));
	scratch = (mpz_t *)malloc(batch * sizeof(mpz_t));
	tmp_factor_size = (uint32 *)malloc(batch * sizeof(uint32));
	tmp_relation = (relation_t *)malloc(batch * sizeof(relation_t));
	tmp_ideal = (relation_lp_t *)malloc(batch * sizeof(relation_lp_t));
	status = (int32 *)malloc(batch * sizeof(int32));

	for (i = 0; i < batch; i++) {
		tmp_relation[i].factors = (uint8 *)malloc(
				COMPRESSED_P_MAX_SIZE * sizeof(uint8));
		mpz_init(scratch[i]);
	}

	logprintf(obj, "commencing singleton removal, initial pass\n");

	savefile_open(savefile, SAVEFILE_READ);
	sprintf(buf, "%s.d", savefile->name);
	relation_fp = fopen(buf, "rb");
	if (relation_fp == NULL) {
		logprintf(obj, "error: can't open dup file\n");
		exit(-1);
	}
	sprintf(lp_name, "%s.lp", savefile->name);
	sprintf(lp_tmp_name, "%s.lp0", savefile->name);
	final_fp = fopen(lp_tmp_name, "wb");
	if (final_fp == NULL) {
		logprintf(obj, "error: can't open output LP file\n");
		exit(-1);
	}
	nfs_buffered_writer_init(&final_writer, final_fp);

	nfs_hash64_init(obj, &unique_ideals, (uint32)WORDS_IN(ideal_t));

	curr_relation = UINT64_MAX;
	next_relation = UINT64_MAX;
	num_relations = 0;
	read_relation_number_checked(obj, relation_fp, &next_relation, "relation selection file");

	do {
		num_relations_read = 0;
		for (i = 0; i < batch; i++) {
			char *buf_i = buf + i * LINE_BUF_SIZE;
			savefile_read_line(buf_i, LINE_BUF_SIZE * sizeof(char), savefile);
			if (savefile_eof(savefile))
				break;
			if (buf_i[0] != '-' && !isdigit(buf_i[0])) {
				i--;
				continue;
			}
			curr_relation++;
			if (max_relations && curr_relation >= max_relations)
				break;

			if (have_skip_list) {
				if (curr_relation == next_relation) {
					read_relation_number_checked(obj, relation_fp, &next_relation, "relation selection file");
					i--;
					continue;
				}
			}
			else {
				if (curr_relation < next_relation) {
					i--;
					continue;
				}
				if (curr_relation != next_relation) {
					logprintf(obj, "error: relation keep-list is out of order\n");
					exit(-1);
				}
				read_relation_number_checked(obj, relation_fp, &next_relation, "relation selection file");
			}

			my_curr_relation[i] = curr_relation;
			if (curr_relation > UINT32_MAX)
				needs_rmap = 1;
			num_relations_read++;
		}

#pragma omp parallel for
		for (i = 0; i < num_relations_read; i++) {
			char *buf_i = buf + i * LINE_BUF_SIZE;
			status[i] = nfs_read_relation(buf_i, fb, &tmp_relation[i],
					&tmp_factor_size[i], 1, scratch[i], 0);
			if (status[i] == 0)
				find_large_ideals(&tmp_relation[i], &tmp_ideal[i],
						filter->filtmin_r, filter->filtmin_a);
		}

		for (i = 0; i < num_relations_read; i++) {
			if (status[i] == 0) {
				uint32 j;
				num_relations++;
				for (j = 0; j < tmp_ideal[i].ideal_count; j++) {
					packed_ideal_ids[j] = nfs_hash64_find(obj,
							&unique_ideals,
							tmp_ideal[i].ideal_list + j, NULL);
				}
				if (write_lp64_record(&final_writer, my_curr_relation[i],
						tmp_ideal[i].ideal_count,
						tmp_ideal[i].gf2_factors,
						packed_ideal_ids) != 0) {
					logprintf(obj, "error: write failed creating LP64 file\n");
					exit(-1);
				}
			}
		}
	} while (num_relations_read == batch);

	for (i = 0; i < batch; i++) {
		free(tmp_relation[i].factors);
		mpz_clear(scratch[i]);
	}
	free(my_curr_relation);
	free(scratch);
	free(tmp_factor_size);
	free(tmp_relation);
	free(tmp_ideal);
	free(status);

	filter->lp_num_relations = num_relations;
	filter->lp_num_ideals = unique_ideals.num_entries;
	filter->lp_is_64bit = 1;
	filter->lp_needs_rmap = needs_rmap;
	filter->num_relations = 0;
	filter->num_ideals = 0;
	filter->relation_array = NULL;
	logprintf(obj, "LP64 contains %" PRIu64 " relations and %" PRIu64
			" unique large ideals\n", num_relations,
			unique_ideals.num_entries);
	logprintf(obj, "memory use: %.1f MB\n",
			(double)nfs_hash64_sizeof(&unique_ideals) / 1048576);
	nfs_hash64_free(&unique_ideals);
	savefile_close(savefile);
	fclose(relation_fp);
	if (nfs_buffered_writer_finish(&final_writer) != 0 || fclose(final_fp) != 0) {
		logprintf(obj, "error: write failed finalizing LP64 file\n");
		exit(-1);
	}
	if (nfs_replace_file(lp_tmp_name, lp_name) != 0) {
		logprintf(obj, "error: can't install completed LP64 file\n");
		exit(-1);
	}

	filter->lp_file_size = get_file_size(lp_name);

	sprintf(buf, "%s.d", savefile->name);
	if (remove(buf) != 0) {
		logprintf(obj, "error: can't delete dup file\n");
		exit(-1);
	}
	free(buf);
}

static uint64 nfs_new_rmap_generation(msieve_obj *obj, uint64 count) {
	uint64 g = ((uint64)time(NULL) << 32) ^ ((uint64)(uint32)getpid() << 1) ^ count;
	g ^= get_file_size(obj->savefile.name);
	g ^= (uint64)clock() << 17;
	/* Mix the inexpensive process-local ingredients without consuming the
	   msieve PRNG state used by later numerical stages. */
	g ^= g >> 33;
	g *= 0xff51afd7ed558ccdULL;
	g ^= g >> 33;
	g *= 0xc4ceb9fe1a85ec53ULL;
	g ^= g >> 33;
	return g ? g : 1;
}

static int nfs_write_rmap_header(nfs_buffered_writer_t *w,
				 uint64 generation, uint64 count) {
	uint64 magic = NFS_RMAP_MAGIC;
	uint64 version = NFS_RMAP_VERSION;
	return nfs_buffered_write(w, &magic, sizeof(magic)) ||
		nfs_buffered_write(w, &version, sizeof(version)) ||
		nfs_buffered_write(w, &generation, sizeof(generation)) ||
		nfs_buffered_write(w, &count, sizeof(count)) ? -1 : 0;
}

static void nfs_install_compacted_files(msieve_obj *obj,
					const char *lp_tmp, const char *lp_name,
					const char *map_tmp, const char *map_name,
					uint32 needs_rmap, uint64 generation,
					uint64 map_count) {
	char commit_name[256], commit_tmp[256];
	FILE *fp;
	uint64 magic = NFS_RMAP_COMMIT_MAGIC;
	sprintf(commit_name, "%s.rmap.commit", obj->savefile.name);
	sprintf(commit_tmp, "%s.rmap.commit0", obj->savefile.name);

	/* Removing the commit marker first makes any crash during the two-file
	   replacement fail closed. Consumers accept a mapped dataset only after
	   a new marker matching the new map generation is installed. */
	remove(commit_name);
	remove(commit_tmp);

	if (needs_rmap) {
		if (nfs_replace_file(map_tmp, map_name) != 0) {
			logprintf(obj, "error: can't install relation map file\n");
			exit(-1);
		}
	}
	else {
		remove(map_name);
		remove(map_tmp);
	}

	if (nfs_replace_file(lp_tmp, lp_name) != 0) {
		logprintf(obj, "error: can't install compacted LP file\n");
		exit(-1);
	}

	if (!needs_rmap)
		return;

	fp = fopen(commit_tmp, "wb");
	if (fp == NULL ||
	    fwrite(&magic, sizeof(uint64), 1, fp) != 1 ||
	    fwrite(&generation, sizeof(uint64), 1, fp) != 1 ||
	    fwrite(&map_count, sizeof(uint64), 1, fp) != 1 ||
	    nfs_close_output(fp) != 0) {
		logprintf(obj, "error: can't commit relation-map transaction\n");
		exit(-1);
	}
	if (nfs_replace_file(commit_tmp, commit_name) != 0) {
		logprintf(obj, "error: can't install relation-map commit marker\n");
		exit(-1);
	}
}

#define ALIVE_TEST(a, i) ((a)[(size_t)((i) >> 3)] & (1u << ((i) & 7)))
#define ALIVE_CLEAR(a, i) do { \
	size_t _alive_byte = (size_t)((i) >> 3); \
	(a)[_alive_byte] = (uint8)((a)[_alive_byte] & \
		(uint8)~(1u << ((i) & 7))); \
} while (0)

/*--------------------------------------------------------------------*/
void nfs_compact_lp_file(msieve_obj *obj, filter_t *filter,
			uint64 ram_size) {
	uint64 i, j;
	uint64 start_relations = filter->lp_num_relations;
	uint64 num_relations = start_relations;
	uint64 num_ideals = filter->lp_num_ideals;
	uint64 active_ideals = num_ideals;
	uint64 num_singletons = 0;
	uint32 num_passes = 0;
	uint64 projected_file_size = filter->lp_file_size;
	uint64 *counts = NULL;
	uint8 *alive = NULL;
	FILE *in_fp, *out_fp, *map_fp = NULL;
	nfs_buffered_reader_t reader;
	nfs_buffered_writer_t out_writer, map_writer;
	char in_name[256], out_name[256], map_name[256], map_tmp_name[256];
	uint64 rel_index;
	uint64 ideal_list[TEMP_FACTOR_LIST_SIZE] = {0};
	uint8 ideal_count, gf2_factors;
	uint32 needs_rmap = filter->lp_needs_rmap;
	uint64 generation = needs_rmap ? nfs_new_rmap_generation(obj, start_relations) : 0;

	if (!filter->lp_is_64bit) {
		logprintf(obj, "error: NFS LP compaction called on a 32-bit LP file\n");
		exit(-1);
	}

	logprintf(obj, "preparing 64-bit LP file for common filtering\n");
	logprintf(obj, "start with %" PRIu64 " relations and %" PRIu64
			" ideals\n", num_relations, num_ideals);

	sprintf(in_name, "%s.lp", obj->savefile.name);
	sprintf(out_name, "%s.lp0", obj->savefile.name);
	sprintf(map_name, "%s.rmap", obj->savefile.name);
	sprintf(map_tmp_name, "%s.rmap0", obj->savefile.name);
	in_fp = fopen(in_name, "rb");
	if (in_fp == NULL) {
		logprintf(obj, "error: can't open LP file\n");
		exit(-1);
	}
	nfs_buffered_reader_init(&reader, in_fp);

	/* Preserve the historical fast path only with substantial headroom below
	   UINT32_MAX. Near the architectural ceiling, force disk singleton
	   pruning so common-filter arithmetic and merge counters have room. */
	if (filter->lp_file_size <= ram_size / 2 &&
	    start_relations <= NFS_COMMON_FILTER_SAFE_MAX &&
	    num_ideals <= NFS_COMMON_FILTER_SAFE_MAX) {
		uint64 dense_id = 0;
		out_fp = fopen(out_name, "wb");
		if (needs_rmap)
			map_fp = fopen(map_tmp_name, "wb");
		if (out_fp == NULL || (needs_rmap && map_fp == NULL)) {
			logprintf(obj, "error: can't open LP compaction output files\n");
			exit(-1);
		}
		nfs_buffered_writer_init(&out_writer, out_fp);
		if (needs_rmap) {
			nfs_buffered_writer_init(&map_writer, map_fp);
			if (nfs_write_rmap_header(&map_writer, generation, start_relations) != 0) {
				logprintf(obj, "error: can't write relation map header\n");
				exit(-1);
			}
		}

		for (i = 0; i < start_relations; i++) {
			relation_ideal_t packed;
			uint32 k;
			size_t header_words, bytes;
			int rc = read_lp64_record(&reader, &rel_index, &ideal_count,
						  &gf2_factors, ideal_list);
			if (rc != 1) {
				logprintf(obj, "error: truncated LP64 file during compaction\n");
				exit(-1);
			}
			packed.rel_index = needs_rmap ? (uint32)dense_id : (uint32)rel_index;
			packed.ideal_count = ideal_count;
			packed.gf2_factors = gf2_factors;
			packed.connected = 0;
			for (k = 0; k < ideal_count; k++) {
				if (ideal_list[k] >= num_ideals || ideal_list[k] > UINT32_MAX) {
					logprintf(obj, "error: invalid ideal ID in direct LP compaction\n");
					exit(-1);
				}
				packed.ideal_list[k] = (uint32)ideal_list[k];
			}
			header_words = (sizeof(relation_ideal_t) - sizeof(packed.ideal_list)) /
					sizeof(uint32);
			bytes = (header_words + ideal_count) * sizeof(uint32);
			if (nfs_buffered_write(&out_writer, &packed, bytes) != 0 ||
			    (needs_rmap && nfs_buffered_write(&map_writer, &rel_index,
							      sizeof(uint64)) != 0)) {
				logprintf(obj, "error: write failed during LP compaction\n");
				exit(-1);
			}
			dense_id++;
		}

		nfs_buffered_reader_free(&reader);
		fclose(in_fp);
		if (nfs_buffered_writer_finish(&out_writer) != 0 || fclose(out_fp) != 0 ||
		    (needs_rmap && (nfs_buffered_writer_finish(&map_writer) != 0 ||
				      fclose(map_fp) != 0))) {
			logprintf(obj, "error: write failed finalizing LP compaction\n");
			exit(-1);
		}
		nfs_install_compacted_files(obj, out_name, in_name, map_tmp_name,
					    map_name, needs_rmap, generation, start_relations);

		filter->num_relations = (uint32)start_relations;
		filter->num_ideals = (uint32)num_ideals;
		filter->lp_num_relations = start_relations;
		filter->lp_num_ideals = num_ideals;
		filter->lp_is_64bit = 0;
		filter->lp_needs_rmap = needs_rmap;
		filter->lp_file_size = get_file_size(in_name);
		filter->relation_array = NULL;
		logprintf(obj, "compacted LP dataset has %u relations and %u large ideals%s\n",
				filter->num_relations, filter->num_ideals,
				needs_rmap ? " (64-bit source relation map active)" : "");
		return;
	}

	counts = (uint64 *)xcalloc(nfs_checked_array_size(obj, num_ideals,
							sizeof(uint64),
							"large-ideal frequency table"), 1);
	alive = (uint8 *)xmalloc(nfs_checked_array_size(obj,
			(start_relations + 7) / 8, sizeof(uint8),
			"relation survivor bitmap"));
	memset(alive, 0xff, nfs_checked_array_size(obj,
			(start_relations + 7) / 8, sizeof(uint8),
			"relation survivor bitmap"));

	for (i = 0; i < start_relations; i++) {
		int rc = read_lp64_record(&reader, &rel_index, &ideal_count,
					  &gf2_factors, ideal_list);
		if (rc != 1) {
			logprintf(obj, "error: truncated LP64 file at relation %" PRIu64 "\n", i);
			exit(-1);
		}
		for (j = 0; j < ideal_count; j++) {
			if (ideal_list[j] >= num_ideals) {
				logprintf(obj, "error: invalid 64-bit ideal ID in LP file\n");
				exit(-1);
			}
			counts[ideal_list[j]]++;
		}
	}

	do {
		uint32 must_fit_common = (num_relations > NFS_COMMON_FILTER_SAFE_MAX ||
					    active_ideals > NFS_COMMON_FILTER_SAFE_MAX);
		uint32 should_prune = must_fit_common ||
			(num_relations > 2000000 && num_singletons > 500000 &&
			 projected_file_size >= ram_size / 2);

		if (num_passes != 0 && !should_prune)
			break;
		num_singletons = 0;
		projected_file_size = 0;
		if (nfs_buffered_reader_rewind(&reader) != 0) {
			logprintf(obj, "error: can't rewind LP64 file\n");
			exit(-1);
		}

		for (i = 0; i < start_relations; i++) {
			uint32 is_singleton = 0;
			int rc = read_lp64_record(&reader, &rel_index, &ideal_count,
						  &gf2_factors, ideal_list);
			if (rc != 1) {
				logprintf(obj, "error: truncated LP64 file during singleton pass\n");
				exit(-1);
			}
			if (!ALIVE_TEST(alive, i))
				continue;
			for (j = 0; j < ideal_count; j++) {
				if (counts[ideal_list[j]] < 2) {
					is_singleton = 1;
					break;
				}
			}
			if (is_singleton) {
				ALIVE_CLEAR(alive, i);
				num_relations--;
				num_singletons++;
				for (j = 0; j < ideal_count; j++) {
					uint64 id = ideal_list[j];
					if (counts[id] != 0 && --counts[id] == 0)
						active_ideals--;
				}
			}
			else {
				projected_file_size += sizeof(uint64) + 2 * sizeof(uint8) +
					(uint64)ideal_count * sizeof(uint64);
			}
		}
		logprintf(obj, "pass %u: found %" PRIu64 " singletons; %" PRIu64
				" relations and %" PRIu64 " ideals remain\n",
				++num_passes, num_singletons, num_relations, active_ideals);
		if (num_singletons == 0 &&
		    (num_relations > NFS_COMMON_FILTER_SAFE_MAX ||
		     active_ideals > NFS_COMMON_FILTER_SAFE_MAX)) {
			logprintf(obj, "error: singleton removal reached a fixed point but "
				"the dataset remains above the safe 32-bit common-filter handoff (%" PRIu64 ")\n",
				NFS_COMMON_FILTER_SAFE_MAX);
			exit(-1);
		}
	} while (1);

	if (num_relations > NFS_COMMON_FILTER_SAFE_MAX ||
	    active_ideals > NFS_COMMON_FILTER_SAFE_MAX) {
		logprintf(obj, "error: LP compaction did not reach the safe common-filter limit\n");
		exit(-1);
	}

	for (i = j = 0; i < num_ideals; i++) {
		if (counts[i] != 0)
			counts[i] = j++;
		else
			counts[i] = UINT64_MAX;
	}
	if (j != active_ideals) {
		logprintf(obj, "error: ideal-count mismatch during LP compaction\n");
		exit(-1);
	}

	out_fp = fopen(out_name, "wb");
	if (needs_rmap)
		map_fp = fopen(map_tmp_name, "wb");
	if (out_fp == NULL || (needs_rmap && map_fp == NULL)) {
		logprintf(obj, "error: can't open LP compaction output files\n");
		exit(-1);
	}
	nfs_buffered_writer_init(&out_writer, out_fp);
	if (needs_rmap) {
		nfs_buffered_writer_init(&map_writer, map_fp);
		generation = nfs_new_rmap_generation(obj, num_relations);
		if (nfs_write_rmap_header(&map_writer, generation, num_relations) != 0) {
			logprintf(obj, "error: can't write relation map header\n");
			exit(-1);
		}
	}
	if (nfs_buffered_reader_rewind(&reader) != 0) {
		logprintf(obj, "error: can't rewind LP64 file\n");
		exit(-1);
	}
	j = 0;
	for (i = 0; i < start_relations; i++) {
		relation_ideal_t packed;
		uint32 k;
		size_t header_words, bytes;
		int rc = read_lp64_record(&reader, &rel_index, &ideal_count,
					  &gf2_factors, ideal_list);
		if (rc != 1) {
			logprintf(obj, "error: truncated LP64 file during compaction\n");
			exit(-1);
		}
		if (!ALIVE_TEST(alive, i))
			continue;
		packed.rel_index = needs_rmap ? (uint32)j : (uint32)rel_index;
		j++;
		packed.ideal_count = ideal_count;
		packed.gf2_factors = gf2_factors;
		packed.connected = 0;
		for (k = 0; k < ideal_count; k++) {
			uint64 mapped = counts[ideal_list[k]];
			if (mapped == UINT64_MAX || mapped > UINT32_MAX) {
				logprintf(obj, "error: invalid ideal remap during LP compaction\n");
				exit(-1);
			}
			packed.ideal_list[k] = (uint32)mapped;
		}
		header_words = (sizeof(relation_ideal_t) - sizeof(packed.ideal_list)) /
			sizeof(uint32);
		bytes = (header_words + ideal_count) * sizeof(uint32);
		if (nfs_buffered_write(&out_writer, &packed, bytes) != 0 ||
		    (needs_rmap && nfs_buffered_write(&map_writer, &rel_index,
						      sizeof(uint64)) != 0)) {
			logprintf(obj, "error: write failed during final LP compaction\n");
			exit(-1);
		}
	}
	if (j != num_relations) {
		logprintf(obj, "error: relation-count mismatch during LP compaction\n");
		exit(-1);
	}

	nfs_buffered_reader_free(&reader);
	fclose(in_fp);
	if (nfs_buffered_writer_finish(&out_writer) != 0 || fclose(out_fp) != 0 ||
	    (needs_rmap && (nfs_buffered_writer_finish(&map_writer) != 0 ||
			      fclose(map_fp) != 0))) {
		logprintf(obj, "error: write failed finalizing LP compaction\n");
		exit(-1);
	}
	free(counts);
	free(alive);

	nfs_install_compacted_files(obj, out_name, in_name, map_tmp_name,
				    map_name, needs_rmap, generation, num_relations);

	filter->num_relations = (uint32)num_relations;
	filter->num_ideals = (uint32)active_ideals;
	filter->lp_num_relations = num_relations;
	filter->lp_num_ideals = active_ideals;
	filter->lp_is_64bit = 0;
	filter->lp_needs_rmap = needs_rmap;
	filter->lp_file_size = get_file_size(in_name);
	filter->relation_array = NULL;
	logprintf(obj, "compacted LP dataset has %u relations and %u large ideals%s\n",
			filter->num_relations, filter->num_ideals,
			needs_rmap ? " (64-bit source relation map active)" : "");
}
