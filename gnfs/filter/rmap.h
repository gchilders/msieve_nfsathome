#ifndef _GNFS_FILTER_RMAP_H_
#define _GNFS_FILTER_RMAP_H_

#define NFS_RMAP_CACHE_ENTRIES (1024u * 1024u)

typedef struct {
	FILE *fp;
	uint64 count;
	uint64 generation;
	uint64 data_offset;
	uint64 cache_first;
	uint32 cache_count;
	uint64 *cache;
} nfs_rmap_reader_t;

static int nfs_rmap_read_header(FILE *fp, uint64 *generation,
				uint64 *count, uint64 *data_offset) {
	uint64 magic, version;
	if (fread(&magic, sizeof(uint64), 1, fp) != 1 ||
	    magic != NFS_RMAP_MAGIC ||
	    fread(&version, sizeof(uint64), 1, fp) != 1 ||
	    version != NFS_RMAP_VERSION ||
	    fread(generation, sizeof(uint64), 1, fp) != 1 ||
	    fread(count, sizeof(uint64), 1, fp) != 1)
		return -1;
	*data_offset = 4 * sizeof(uint64);
	return 0;
}

static int nfs_rmap_check_commit(msieve_obj *obj, uint64 generation,
				 uint64 count) {
	char filename[256];
	FILE *fp;
	uint64 magic, commit_generation, commit_count;
	sprintf(filename, "%s.rmap.commit", obj->savefile.name);
	fp = fopen(filename, "rb");
	if (fp == NULL)
		return -1;
	if (fread(&magic, sizeof(uint64), 1, fp) != 1 ||
	    fread(&commit_generation, sizeof(uint64), 1, fp) != 1 ||
	    fread(&commit_count, sizeof(uint64), 1, fp) != 1) {
		fclose(fp);
		return -1;
	}
	if (fclose(fp) != 0 || magic != NFS_RMAP_COMMIT_MAGIC ||
	    commit_generation != generation || commit_count != count)
		return -1;
	return 0;
}

static int nfs_rmap_reader_open(msieve_obj *obj, nfs_rmap_reader_t *r,
				uint32 require_committed,
				uint64 expected_generation) {
	char filename[256];
	memset(r, 0, sizeof(*r));
	sprintf(filename, "%s.rmap", obj->savefile.name);
	r->fp = fopen(filename, "rb");
	if (r->fp == NULL)
		return -1;
	if (nfs_rmap_read_header(r->fp, &r->generation,
				 &r->count, &r->data_offset) != 0 ||
	    (expected_generation != UINT64_MAX &&
	     r->generation != expected_generation) ||
	    (require_committed &&
	     nfs_rmap_check_commit(obj, r->generation, r->count) != 0)) {
		fclose(r->fp);
		memset(r, 0, sizeof(*r));
		return -1;
	}
	r->cache = (uint64 *)xmalloc((size_t)NFS_RMAP_CACHE_ENTRIES *
				      sizeof(uint64));
	r->cache_first = UINT64_MAX;
	return 0;
}

static uint64 nfs_rmap_get(msieve_obj *obj, nfs_rmap_reader_t *r,
			   uint32 dense_id) {
	uint64 id = dense_id;
	if (id >= r->count) {
		logprintf(obj, "error: dense relation ID %u is outside relation map\n",
				dense_id);
		exit(-1);
	}
	if (r->cache_first == UINT64_MAX || id < r->cache_first ||
	    id >= r->cache_first + r->cache_count) {
		uint64 first = id & ~((uint64)NFS_RMAP_CACHE_ENTRIES - 1);
		uint64 remain = r->count - first;
		uint32 want = (uint32)MIN(remain, (uint64)NFS_RMAP_CACHE_ENTRIES);
		uint64 offset = r->data_offset + first * sizeof(uint64);
		if (fseeko(r->fp, (int64)offset, SEEK_SET) != 0 ||
		    fread(r->cache, sizeof(uint64), want, r->fp) != want) {
			logprintf(obj, "error: cannot bulk-read relation map near %u\n",
					dense_id);
			exit(-1);
		}
		r->cache_first = first;
		r->cache_count = want;
	}
	return r->cache[id - r->cache_first];
}

static void nfs_rmap_reader_close(nfs_rmap_reader_t *r) {
	if (r->fp != NULL)
		fclose(r->fp);
	free(r->cache);
	memset(r, 0, sizeof(*r));
}

#endif
