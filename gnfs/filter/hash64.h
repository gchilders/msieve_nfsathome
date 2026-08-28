#ifndef _GNFS_FILTER_HASH64_H_
#define _GNFS_FILTER_HASH64_H_

/* Segmented 64-bit-ordinal hash table for the very-large NFS prefilter.
   Bucket heads and chain links are uint64 ordinals. Entry storage grows in
   fixed-size segments so adding an entry never requires reallocating or
   copying a multi-gigabyte key/link array. */

#define NFS_HASH64_SEGMENT_LOG2 20
#define NFS_HASH64_SEGMENT_SIZE ((uint64)1 << NFS_HASH64_SEGMENT_LOG2)
#define NFS_HASH64_SEGMENT_MASK (NFS_HASH64_SEGMENT_SIZE - 1)

typedef struct {
	uint32 key_words;
	uint32 log2_bucket_count;
	uint64 num_entries;
	uint64 num_segments;
	uint64 segment_ptr_alloc;
	uint64 *buckets;
	uint64 **next_segments;
	uint32 **key_segments;
} nfs_hashtable64_t;

static size_t nfs_checked_array_size(msieve_obj *obj, uint64 count,
				     size_t elem_size, const char *what) {
	if (count > (uint64)((size_t)-1 / elem_size)) {
		logprintf(obj, "error: %s is too large for this build\n", what);
		exit(-1);
	}
	return (size_t)count * elem_size;
}

static uint32 nfs_hash64_bucket(const uint32 *key, uint32 key_words,
				uint32 log2_bucket_count) {
	uint32 h = hash_function((uint32 *)key, key_words);
	if (log2_bucket_count == 32)
		return h;
	return h >> (32 - log2_bucket_count);
}

static uint64 *nfs_hash64_next_ptr(nfs_hashtable64_t *h, uint64 entry) {
	return h->next_segments[(size_t)(entry >> NFS_HASH64_SEGMENT_LOG2)] +
		(size_t)(entry & NFS_HASH64_SEGMENT_MASK);
}

static uint32 *nfs_hash64_key_ptr(nfs_hashtable64_t *h, uint64 entry) {
	return h->key_segments[(size_t)(entry >> NFS_HASH64_SEGMENT_LOG2)] +
		(size_t)(entry & NFS_HASH64_SEGMENT_MASK) * h->key_words;
}

static void nfs_hash64_rehash(msieve_obj *obj, nfs_hashtable64_t *h,
			      uint32 new_log2) {
	uint64 i;
	uint64 bucket_count = (uint64)1 << new_log2;
	uint64 *buckets = (uint64 *)xcalloc(
			nfs_checked_array_size(obj, bucket_count, sizeof(uint64),
					       "64-bit NFS hash buckets"), 1);

	for (i = 0; i < h->num_entries; i++) {
		uint32 *key = nfs_hash64_key_ptr(h, i);
		uint64 *next = nfs_hash64_next_ptr(h, i);
		uint32 bucket = nfs_hash64_bucket(key, h->key_words, new_log2);
		*next = buckets[bucket];
		buckets[bucket] = i + 1;
	}

	free(h->buckets);
	h->buckets = buckets;
	h->log2_bucket_count = new_log2;
}

static void nfs_hash64_init(msieve_obj *obj, nfs_hashtable64_t *h,
			    uint32 key_words) {
	memset(h, 0, sizeof(*h));
	h->key_words = key_words;
	h->log2_bucket_count = 14;
	h->segment_ptr_alloc = 16;
	h->buckets = (uint64 *)xcalloc(
			nfs_checked_array_size(obj,
				(uint64)1 << h->log2_bucket_count,
				sizeof(uint64), "64-bit NFS hash buckets"), 1);
	h->next_segments = (uint64 **)xcalloc(
			nfs_checked_array_size(obj, h->segment_ptr_alloc,
					       sizeof(uint64 *), "NFS hash segment table"), 1);
	h->key_segments = (uint32 **)xcalloc(
			nfs_checked_array_size(obj, h->segment_ptr_alloc,
					       sizeof(uint32 *), "NFS hash segment table"), 1);
}

static void nfs_hash64_add_segment(msieve_obj *obj, nfs_hashtable64_t *h) {
	uint64 new_alloc;
	if (h->num_segments == h->segment_ptr_alloc) {
		if (h->segment_ptr_alloc > UINT64_MAX / 2) {
			logprintf(obj, "error: too many NFS hash segments\n");
			exit(-1);
		}
		new_alloc = h->segment_ptr_alloc * 2;
		h->next_segments = (uint64 **)xrealloc(h->next_segments,
			nfs_checked_array_size(obj, new_alloc, sizeof(uint64 *),
					       "NFS hash segment table"));
		h->key_segments = (uint32 **)xrealloc(h->key_segments,
			nfs_checked_array_size(obj, new_alloc, sizeof(uint32 *),
					       "NFS hash segment table"));
		memset(h->next_segments + h->segment_ptr_alloc, 0,
			(size_t)(new_alloc - h->segment_ptr_alloc) * sizeof(uint64 *));
		memset(h->key_segments + h->segment_ptr_alloc, 0,
			(size_t)(new_alloc - h->segment_ptr_alloc) * sizeof(uint32 *));
		h->segment_ptr_alloc = new_alloc;
	}

	h->next_segments[h->num_segments] = (uint64 *)xmalloc(
		nfs_checked_array_size(obj, NFS_HASH64_SEGMENT_SIZE, sizeof(uint64),
				       "NFS hash link segment"));
	h->key_segments[h->num_segments] = (uint32 *)xmalloc(
		nfs_checked_array_size(obj, NFS_HASH64_SEGMENT_SIZE * h->key_words,
				       sizeof(uint32), "NFS hash key segment"));
	h->num_segments++;
}

/* Return the stable uint64 ordinal for blob. If is_new is non-NULL, set it
   to one only when a new entry was inserted. */
static uint64 nfs_hash64_find(msieve_obj *obj, nfs_hashtable64_t *h,
			      const void *blob, uint32 *is_new) {
	uint32 i;
	const uint32 *key = (const uint32 *)blob;
	uint32 bucket;
	uint64 link;
	uint64 bucket_count = (uint64)1 << h->log2_bucket_count;

	if (h->num_entries + 1 >= (bucket_count * 4) / 5 &&
	    h->log2_bucket_count < 32)
		nfs_hash64_rehash(obj, h, h->log2_bucket_count + 1);

	bucket = nfs_hash64_bucket(key, h->key_words, h->log2_bucket_count);
	link = h->buckets[bucket];
	while (link != 0) {
		uint64 entry_num = link - 1;
		uint32 *entry = nfs_hash64_key_ptr(h, entry_num);
		for (i = 0; i < h->key_words; i++) {
			if (entry[i] != key[i])
				break;
		}
		if (i == h->key_words) {
			if (is_new != NULL)
				*is_new = 0;
			return entry_num;
		}
		link = *nfs_hash64_next_ptr(h, entry_num);
	}

	if ((h->num_entries & NFS_HASH64_SEGMENT_MASK) == 0)
		nfs_hash64_add_segment(obj, h);

	{
		uint64 entry_num = h->num_entries++;
		uint32 *entry = nfs_hash64_key_ptr(h, entry_num);
		uint64 *next = nfs_hash64_next_ptr(h, entry_num);
		for (i = 0; i < h->key_words; i++)
			entry[i] = key[i];
		*next = h->buckets[bucket];
		h->buckets[bucket] = entry_num + 1;
		if (is_new != NULL)
			*is_new = 1;
		return entry_num;
	}
}

static uint64 nfs_hash64_sizeof(const nfs_hashtable64_t *h) {
	uint64 bucket_count = (uint64)1 << h->log2_bucket_count;
	uint64 per_segment = NFS_HASH64_SEGMENT_SIZE *
		(sizeof(uint64) + (uint64)h->key_words * sizeof(uint32));
	return bucket_count * sizeof(uint64) + h->num_segments * per_segment +
		h->segment_ptr_alloc * (sizeof(uint64 *) + sizeof(uint32 *));
}

static void nfs_hash64_free(nfs_hashtable64_t *h) {
	uint64 i;
	free(h->buckets);
	for (i = 0; i < h->num_segments; i++) {
		free(h->next_segments[i]);
		free(h->key_segments[i]);
	}
	free(h->next_segments);
	free(h->key_segments);
	memset(h, 0, sizeof(*h));
}

#endif
