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

#include "lanczos.h"

/*--------------------------------------------------------------------*/
#ifdef HAVE_MPI

typedef struct {
	uint32 col_start;
	uint64 mat_file_offset;
} mat_block_t;

typedef struct {
	uint64 sparse_per_proc;
	uint64 curr_sparse;
	uint64 target_sparse;
	uint32 curr_mpi;
	uint32 curr_col;
	mat_block_t idx_entries[MAX_MPI_GRID_DIM + 1];
} mat_idx_t;

static mat_idx_t * mat_idx_init(uint64 num_sparse) {

	uint32 i;
	mat_idx_t *m = (mat_idx_t *)xcalloc(MAX_MPI_GRID_DIM,
					sizeof(mat_idx_t));

	for (i = 1; i <= MAX_MPI_GRID_DIM; i++)
		m[i-1].sparse_per_proc = num_sparse / i + 100;

	return m;
}

static void mat_idx_update(mat_idx_t *m, FILE *mat_fp,
			uint32 curr_sparse) {

	uint32 i;

	for (i = 0; i < MAX_MPI_GRID_DIM; i++) {
		mat_idx_t *curr_m = m + i;

		if (curr_m->curr_sparse >= curr_m->target_sparse) {
			mat_block_t *curr_block = curr_m->idx_entries +
							curr_m->curr_mpi++;
			curr_block->col_start = curr_m->curr_col;
			curr_block->mat_file_offset = ftello(mat_fp);

			curr_m->target_sparse = curr_m->curr_sparse +
						curr_m->sparse_per_proc;
		}

		curr_m->curr_col++;
		curr_m->curr_sparse += curr_sparse;
	}
}

static void mat_idx_final(msieve_obj *obj, mat_idx_t *m,
			uint32 ncols, uint64 mat_file_size) {

	uint32 i;
	char buf[256];
	FILE *idx_fp;

	sprintf(buf, "%s.mat.idx", obj->savefile.name);
	idx_fp = fopen(buf, "wb");
	if (idx_fp == NULL) {
		logprintf(obj, "error: can't open matrix index file\n");
		exit(-1);
	}

	i = MAX_MPI_GRID_DIM;
	if (fwrite(&i, sizeof(uint32), (size_t)1, idx_fp) != 1) {
		logprintf(obj, "error: can't write matrix index header\n");
		exit(-1);
	}

	for (i = 1; i <= MAX_MPI_GRID_DIM; i++) {
		mat_idx_t *curr_m = m + (i-1);

		curr_m->idx_entries[i].col_start = ncols;
		curr_m->idx_entries[i].mat_file_offset = mat_file_size;

		if (fwrite(curr_m->idx_entries, sizeof(mat_block_t),
					(size_t)(i+1), idx_fp) != (size_t)(i+1)) {
			logprintf(obj, "error: can't write matrix index\n");
			exit(-1);
		}

	}

	if (fflush(idx_fp) != 0 || ferror(idx_fp) || fclose(idx_fp) != 0) {
		logprintf(obj, "error: can't finalize matrix index\n");
		exit(-1);
	}
	free(m);
}

static void find_submatrix_bounds(msieve_obj *obj, uint32 *ncols,
			uint32 *start_col, uint64 *mat_file_offset) {

	mat_block_t mat_block;
	mat_block_t next_mat_block;
	char buf[256];
	FILE *matrix_idx_fp;
	uint32 max_grid_cols;

	sprintf(buf, "%s.mat.idx", obj->savefile.name);
	matrix_idx_fp = fopen(buf, "rb");
	if (matrix_idx_fp == NULL) {
		logprintf(obj, "error: can't open matrix index file\n");
		exit(-1);
	}

	if (fread(&max_grid_cols, sizeof(uint32), (size_t)1, matrix_idx_fp) != 1) {
		logprintf(obj, "error: truncated matrix index header\n");
		exit(-1);
	}
	if (max_grid_cols < obj->mpi_ncols) {
		logprintf(obj, "error: matrix expects MPI cols <= %u\n",
				max_grid_cols);
		exit(-1);
	}

	if (fseek(matrix_idx_fp,
		(long)((obj->mpi_ncols *
		        (obj->mpi_ncols + 1) / 2 - 1 +
			obj->mpi_la_col_rank) * sizeof(mat_block_t)),
		SEEK_CUR) != 0) {
		logprintf(obj, "error: can't seek matrix index\n");
		exit(-1);
	}

	if (fread(&mat_block, sizeof(mat_block_t), 1, matrix_idx_fp) != 1 ||
	    fread(&next_mat_block, sizeof(mat_block_t), 1, matrix_idx_fp) != 1 ||
	    fclose(matrix_idx_fp) != 0) {
		logprintf(obj, "error: truncated matrix index\n");
		exit(-1);
	}

	*start_col = mat_block.col_start;
	*ncols = next_mat_block.col_start - mat_block.col_start;
	*mat_file_offset = mat_block.mat_file_offset;
}

#endif

static uint32 cycle_header_rmap_info(msieve_obj *obj, uint64 *generation) {
	char buf[256];
	FILE *fp;
	uint32 first, version, flags, ncycles;
	uint64 gen;
	*generation = 0;
	sprintf(buf, "%s.cyc", obj->savefile.name);
	fp = fopen(buf, "rb");
	if (fp == NULL)
		return 0;
	if (fread(&first, sizeof(uint32), 1, fp) != 1 || first != CYCLE_FILE_MAGIC) {
		fclose(fp);
		return 0;
	}
	if (fread(&version, sizeof(uint32), 1, fp) != 1 ||
	    fread(&flags, sizeof(uint32), 1, fp) != 1 ||
	    fread(&ncycles, sizeof(uint32), 1, fp) != 1 ||
	    fread(&gen, sizeof(uint64), 1, fp) != 1 ||
	    fclose(fp) != 0 || version != CYCLE_FILE_VERSION) {
		logprintf(obj, "error: invalid versioned cycle header\n");
		exit(-1);
	}
	(void)ncycles;
	*generation = gen;
	return (flags & CYCLE_FLAG_RMAP_REQUIRED) != 0;
}

static void validate_cycle_rmap(msieve_obj *obj, uint64 expected_generation) {
	char buf[256];
	FILE *fp;
	uint64 magic, version, generation, count;
	uint64 cmagic, cgen, ccount;
	sprintf(buf, "%s.rmap", obj->savefile.name);
	fp = fopen(buf, "rb");
	if (fp == NULL || fread(&magic, sizeof(uint64), 1, fp) != 1 ||
	    fread(&version, sizeof(uint64), 1, fp) != 1 ||
	    fread(&generation, sizeof(uint64), 1, fp) != 1 ||
	    fread(&count, sizeof(uint64), 1, fp) != 1 ||
	    fclose(fp) != 0 || magic != NFS_RMAP_MAGIC ||
	    version != NFS_RMAP_VERSION || generation != expected_generation) {
		logprintf(obj, "error: cycle file requires a matching relation map\n");
		exit(-1);
	}
	sprintf(buf, "%s.rmap.commit", obj->savefile.name);
	fp = fopen(buf, "rb");
	if (fp == NULL || fread(&cmagic, sizeof(uint64), 1, fp) != 1 ||
	    fread(&cgen, sizeof(uint64), 1, fp) != 1 ||
	    fread(&ccount, sizeof(uint64), 1, fp) != 1 ||
	    fclose(fp) != 0 || cmagic != NFS_RMAP_COMMIT_MAGIC ||
	    cgen != generation || ccount != count) {
		logprintf(obj, "error: required relation map is not transactionally committed\n");
		exit(-1);
	}
}

/*--------------------------------------------------------------------*/
void dump_cycles(msieve_obj *obj, la_col_t *cols, uint32 ncols) {

	uint32 i;
	char buf[256];
	FILE *cycle_fp;
	uint64 rmap_generation = 0;
	uint32 require_rmap;

	sprintf(buf, "%s.cyc", obj->savefile.name);
	require_rmap = cycle_header_rmap_info(obj, &rmap_generation);
	if (require_rmap)
		validate_cycle_rmap(obj, rmap_generation);
	cycle_fp = fopen(buf, "wb");
	if (cycle_fp == NULL) {
		logprintf(obj, "error: can't open cycle file\n");
		exit(-1);
	}

	if (require_rmap) {
		uint32 magic = CYCLE_FILE_MAGIC, version = CYCLE_FILE_VERSION;
		uint32 flags = CYCLE_FLAG_RMAP_REQUIRED;
		if (fwrite(&magic, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&version, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&flags, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&ncols, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(&rmap_generation, sizeof(uint64), 1, cycle_fp) != 1) {
			logprintf(obj, "error: can't write cycle header\n");
			exit(-1);
		}
	}
	else if (fwrite(&ncols, sizeof(uint32), 1, cycle_fp) != 1) {
		logprintf(obj, "error: can't write cycle header\n");
		exit(-1);
	}

	for (i = 0; i < ncols; i++) {
		la_col_t *c = cols + i;
		uint32 num = c->cycle.num_relations;

		if (fwrite(&num, sizeof(uint32), 1, cycle_fp) != 1 ||
		    fwrite(c->cycle.list, sizeof(uint32), (size_t)num, cycle_fp) != num) {
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
void dump_matrix(msieve_obj *obj,
		uint32 nrows, uint32 num_dense_rows,
		uint32 ncols, la_col_t *cols,
		uint64 sparse_weight) {

	uint32 i;
	uint32 dense_row_words;
	char buf[256];
	FILE *matrix_fp;
#ifdef HAVE_MPI
	mat_idx_t *mpi_idx_data = mat_idx_init(sparse_weight);
#endif

	dump_cycles(obj, cols, ncols);

	sprintf(buf, "%s.mat", obj->savefile.name);
	matrix_fp = fopen(buf, "wb");
	if (matrix_fp == NULL) {
		logprintf(obj, "error: can't open matrix file\n");
		exit(-1);
	}

	if (fwrite(&nrows, sizeof(uint32), 1, matrix_fp) != 1 ||
	    fwrite(&num_dense_rows, sizeof(uint32), 1, matrix_fp) != 1 ||
	    fwrite(&ncols, sizeof(uint32), 1, matrix_fp) != 1) {
		logprintf(obj, "error: can't write matrix header\n");
		exit(-1);
	}
	dense_row_words = (num_dense_rows + 31) / 32;

	for (i = 0; i < ncols; i++) {
		la_col_t *c = cols + i;
		uint32 num = c->weight + dense_row_words;

#ifdef HAVE_MPI
		mat_idx_update(mpi_idx_data, matrix_fp, c->weight);
#endif
		if (fwrite(&c->weight, sizeof(uint32), 1, matrix_fp) != 1 ||
		    fwrite(c->data, sizeof(uint32), (size_t)num, matrix_fp) != num) {
			logprintf(obj, "error: can't write matrix column %u\n", i);
			exit(-1);
		}
	}

#ifdef HAVE_MPI
	mat_idx_final(obj, mpi_idx_data, ncols, ftello(matrix_fp));
#endif
	if (fflush(matrix_fp) != 0 || ferror(matrix_fp) || fclose(matrix_fp) != 0) {
		logprintf(obj, "error: can't finalize matrix file\n");
		exit(-1);
	}
}

/*--------------------------------------------------------------------*/
void read_cycles(msieve_obj *obj,
		uint32 *num_cycles_out,
		la_col_t **cycle_list_out,
		uint32 dependency,
		uint32 *colperm) {

	uint32 i;
	uint32 num_cycles;
	uint32 curr_cycle;
	uint32 rel_index[MAX_COL_IDEALS];
	char buf[256];
	FILE *cycle_fp;
	FILE *dep_fp = NULL;
	la_col_t *cycle_list = *cycle_list_out;
	uint64 mask = 0;

	if (dependency > 0 && colperm != NULL) {
		logprintf(obj, "error: cannot read dependency with permute\n");
		exit(-1);
	}

	sprintf(buf, "%s.cyc", obj->savefile.name);
	cycle_fp = fopen(buf, "rb");
	if (cycle_fp == NULL) {
		logprintf(obj, "error: read_cycles can't open cycle file\n");
		exit(-1);
	}

	if (dependency) {
		sprintf(buf, "%s.dep", obj->savefile.name);
		dep_fp = fopen(buf, "rb");
		if (dep_fp == NULL) {
			logprintf(obj, "error: read_cycles can't "
					"open dependency file\n");
			exit(-1);
		}
		mask = (uint64)1 << (dependency - 1);
	}

	/* read the number of cycles to expect. If necessary,
	   allocate space for them */

	{
		uint32 first;
		if (fread(&first, sizeof(uint32), 1, cycle_fp) != 1) {
			logprintf(obj, "error: empty cycle file\n");
			exit(-1);
		}
		if (first == CYCLE_FILE_MAGIC) {
			uint32 version, flags;
			uint64 generation;
			if (fread(&version, sizeof(uint32), 1, cycle_fp) != 1 ||
			    fread(&flags, sizeof(uint32), 1, cycle_fp) != 1 ||
			    fread(&num_cycles, sizeof(uint32), 1, cycle_fp) != 1 ||
			    fread(&generation, sizeof(uint64), 1, cycle_fp) != 1 ||
			    version != CYCLE_FILE_VERSION) {
				logprintf(obj, "error: invalid versioned cycle file\n");
				exit(-1);
			}
			if (flags & CYCLE_FLAG_RMAP_REQUIRED)
				validate_cycle_rmap(obj, generation);
		}
		else {
			num_cycles = first;
		}
	}
	if (cycle_list == NULL) {
		cycle_list = (la_col_t *)xcalloc((size_t)num_cycles,
						sizeof(la_col_t));
	}

	/* read the relation numbers for each cycle */

	for (i = curr_cycle = 0; i < num_cycles; i++) {

		la_col_t *c;
		uint32 num_relations;

		if (fread(&num_relations, sizeof(uint32), 1, cycle_fp) != 1) {
			logprintf(obj, "error: truncated cycle file at cycle %u of %u\n",
				i, num_cycles);
			exit(-1);
		}

		if (num_relations > MAX_COL_IDEALS) {
			printf("error: cycle too large; corrupt file?\n");
			exit(-1);
		}

		if (fread(rel_index, sizeof(uint32), (size_t)num_relations,
					cycle_fp) != num_relations) {
			logprintf(obj, "error: truncated relation list in cycle %u\n", i);
			exit(-1);
		}

		/* all the relation numbers for this cycle
		   have been read; save them and start the
		   count for the next cycle. If reading in
		   relations to produce a particular dependency
		   from the linear algebra phase, skip any
		   cycles that will not appear in the dependency */

		if (dependency) {
			uint64 curr_dep;

			if (fread(&curr_dep, sizeof(uint64), 1, dep_fp) != 1) {
				printf("dependency file corrupt\n");
				exit(-1);
			}
			if (!(curr_dep & mask))
				continue;
		}

		if (colperm != NULL)
			c = cycle_list + colperm[i];
		else
			c = cycle_list + curr_cycle;

		curr_cycle++;
		c->cycle.num_relations = num_relations;
		c->cycle.list = (uint32 *)xmalloc(num_relations *
						sizeof(uint32));
		memcpy(c->cycle.list, rel_index,
				num_relations * sizeof(uint32));
	}
	logprintf(obj, "read %u cycles\n", curr_cycle);
	num_cycles = curr_cycle;

	/* check that all cycles have a nonzero number of relations */
	for (i = 0; i < num_cycles; i++) {
		if (cycle_list[i].cycle.num_relations == 0) {
			logprintf(obj, "error: empty cycle encountered\n");
			exit(-1);
		}
	}

	if (ferror(cycle_fp) || fclose(cycle_fp) != 0) {
		logprintf(obj, "error: can't finalize cycle-file read\n");
		exit(-1);
	}
	if (dep_fp) {
		if (ferror(dep_fp) || fclose(dep_fp) != 0) {
			logprintf(obj, "error: can't finalize dependency-file read\n");
			exit(-1);
		}
	}
	if (num_cycles == 0) {
		free(cycle_list);
		*num_cycles_out = 0;
		*cycle_list_out = NULL;
		return;
	}

	*num_cycles_out = num_cycles;
	*cycle_list_out = (la_col_t *)xrealloc(cycle_list,
				num_cycles * sizeof(la_col_t));
}
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
#define FILE_CACHE_WORDS 20000

typedef struct {
	uint32 read_ptr;
	uint32 num_valid;
	uint32 *cache;
} file_cache_t;

static void file_cache_init(file_cache_t *f) {

	f->read_ptr = 0;
	f->num_valid = 0;
	f->cache = (uint32 *)xmalloc(FILE_CACHE_WORDS * sizeof(uint32));
}

static void file_cache_free(file_cache_t *f) {

	free(f->cache);
}

static void file_cache_get_next(msieve_obj *obj, FILE *fp,
				file_cache_t *f, uint32 dense_row_words,
				uint32 *num_out, uint32 *entries,
				uint32 read_submatrix) {

	uint32 num;
	uint32 need;
	uint32 words_left;

	/* First obtain a complete column header, then obtain the complete
	   column payload. Never inspect cache words that were not read. */
	for (;;) {
		words_left = f->num_valid - f->read_ptr;
		if (words_left >= 1)
			break;
		memmove(f->cache, f->cache + f->read_ptr,
				words_left * sizeof(uint32));
#ifdef HAVE_MPI
		/* only the top MPI row reads from disk */

		if (obj->mpi_la_row_rank == 0) {
#endif
		if (feof(fp)) {
			logprintf(obj, "error: truncated matrix column\n");
			exit(-1);
		}
		f->num_valid = words_left + (uint32)
			fread(f->cache + words_left, sizeof(uint32),
				FILE_CACHE_WORDS - words_left, fp);
		if (ferror(fp)) {
			logprintf(obj, "error: matrix read failed\n");
			exit(-1);
		}
#ifdef HAVE_MPI
		}

		if (read_submatrix && obj->mpi_nrows > 1) {
			/* broadcast the new cache size and new data
			   (if any) down the column */

			MPI_TRY(MPI_Bcast(&f->num_valid, 1, MPI_INT, 0,
						obj->mpi_la_col_grid))

			if (f->num_valid > words_left) {
				MPI_TRY(MPI_Bcast(f->cache + words_left,
						f->num_valid - words_left,
						MPI_INT, 0, obj->mpi_la_col_grid))
			}
		}
#endif
		f->read_ptr = 0;
		if (f->num_valid < 1) {
			logprintf(obj, "error: truncated matrix column header\n");
			exit(-1);
		}
	}

	num = f->cache[f->read_ptr];
	if ((uint64)num + dense_row_words > MAX_COL_IDEALS) {
		printf("error: column too large; corrupt file?\n");
		exit(-1);
	}
	need = num + dense_row_words + 1;
	if (need > FILE_CACHE_WORDS) {
		logprintf(obj, "error: matrix column exceeds I/O cache capacity\n");
		exit(-1);
	}
	while (f->num_valid - f->read_ptr < need) {
		words_left = f->num_valid - f->read_ptr;
		memmove(f->cache, f->cache + f->read_ptr,
				words_left * sizeof(uint32));
#ifdef HAVE_MPI
		if (obj->mpi_la_row_rank == 0) {
#endif
			if (feof(fp)) {
				logprintf(obj, "error: truncated matrix column\n");
				exit(-1);
			}
			f->num_valid = words_left + (uint32)
				fread(f->cache + words_left, sizeof(uint32),
					FILE_CACHE_WORDS - words_left, fp);
			if (ferror(fp)) {
				logprintf(obj, "error: matrix read failed\n");
				exit(-1);
			}
#ifdef HAVE_MPI
		}
		if (read_submatrix && obj->mpi_nrows > 1) {
			MPI_TRY(MPI_Bcast(&f->num_valid, 1, MPI_INT, 0,
					obj->mpi_la_col_grid))
			if (f->num_valid > words_left) {
				MPI_TRY(MPI_Bcast(f->cache + words_left,
					f->num_valid - words_left, MPI_INT, 0,
					obj->mpi_la_col_grid))
			}
		}
#endif
		f->read_ptr = 0;
		if (f->num_valid < need) {
			logprintf(obj, "error: truncated matrix column\n");
			exit(-1);
		}
	}

	*num_out = num;
	memcpy(entries, f->cache + f->read_ptr + 1,
			(num + dense_row_words) * sizeof(uint32));
	f->read_ptr += num + dense_row_words + 1;
}

/*--------------------------------------------------------------------*/
void read_matrix(msieve_obj *obj,
		uint32 *nrows_out, uint32 *max_nrows_out,
		uint32 *start_row_out,
		uint32 *dense_rows_out,
		uint32 *ncols_out, uint32 *max_ncols_out,
		uint32 *start_col_out,
		la_col_t **cols_out, uint32 *rowperm, uint32 *colperm) {

	uint32 i, j, k;
	uint32 dense_rows, dense_row_words;
	uint32 ncols, max_ncols, start_col;
	uint32 nrows, max_nrows, start_row;
	uint32 mpi_resclass, mpi_nrows;
	la_col_t *cols;
	char buf[256];
	FILE *matrix_fp;
	uint32 read_submatrix = (start_row_out != NULL &&
				start_col_out != NULL);
	file_cache_t file_cache;
#ifdef HAVE_MPI
	uint32 num_static_rows = 0;
#endif

	if (read_submatrix && colperm != NULL) {
		logprintf(obj, "error: cannot read submatrix with permute\n");
		exit(-1);
	}

	sprintf(buf, "%s.mat", obj->savefile.name);
	matrix_fp = fopen(buf, "rb");
	if (matrix_fp == NULL) {
		logprintf(obj, "error: cannot open matrix file\n");
		exit(-1);
	}

	if (fread(&max_nrows, sizeof(uint32), 1, matrix_fp) != 1 ||
	    fread(&dense_rows, sizeof(uint32), 1, matrix_fp) != 1 ||
	    fread(&max_ncols, sizeof(uint32), 1, matrix_fp) != 1) {
		logprintf(obj, "error: truncated matrix header\n");
		exit(-1);
	}
	if (dense_rows > max_nrows || (uint64)dense_rows + 31 >
			(uint64)MAX_COL_IDEALS * 32 + 31) {
		logprintf(obj, "error: invalid dense-row count in matrix\n");
		exit(-1);
	}

	/* default bounding rectangle on matrix read in */

	dense_row_words = (dense_rows + 31) / 32;
	nrows = max_nrows;
	ncols = max_ncols;
	start_row = start_col = 0;
	mpi_resclass = 0;
	mpi_nrows = 1;

#ifdef HAVE_MPI
	if (read_submatrix) {
		/* read in only a subset of the matrix */

		uint64 mat_file_offset;

		find_submatrix_bounds(obj, &ncols, &start_col,
					&mat_file_offset);
		fseeko(matrix_fp, mat_file_offset, SEEK_SET);

		mpi_resclass = obj->mpi_la_row_rank;
		mpi_nrows = obj->mpi_nrows;

		/* we perform an on-the-fly permutation of the rows,
		   so that row i winds up in MPI row (i % mpi_nrows).
		   This is basically a scatter of the initial row
		   ordering across all the MPI rows.

		   While this will distribute the nonzeros across
		   the MPI rows approximately evenly, we can only
		   remove the densest rows from the top row of MPI
		   processes. Hence the first few row numbers must
		   not be permuted. Actually this isn't strictly
		   necessary and we can scatter all the rows, whether
		   sparse or dense, but only a very few rows really
		   benefit from being packed so it's not critical
		   to give every MPI some dense rows */

		num_static_rows = POST_LANCZOS_ROWS;
		while (num_static_rows < dense_rows)
			num_static_rows += 64;
		num_static_rows = MAX(64, num_static_rows);

		/* increase the number of static rows until the
		   remaining number of rows is a multiple of mpi_nrows */

		num_static_rows += (nrows - num_static_rows) % mpi_nrows;

		/* finally, compute the starting row number for the
		   current MPI process */

		nrows = (nrows - num_static_rows) / mpi_nrows;
		if (mpi_resclass == 0)
			nrows += num_static_rows;
		else
			start_row = num_static_rows + mpi_resclass * nrows;
	}
#endif
	cols = (la_col_t *)xcalloc((size_t)ncols, sizeof(la_col_t));

	file_cache_init(&file_cache);

	for (i = 0; i < ncols; i++) {
		la_col_t *c;
		uint32 tmp_col[MAX_COL_IDEALS];
		uint32 num;

		if (colperm != NULL)
			c = cols + colperm[i];
		else
			c = cols + i;

		/* read the whole column */

		file_cache_get_next(obj, matrix_fp, &file_cache,
				dense_row_words, &num, tmp_col,
				read_submatrix);
		k = num + dense_row_words;
		c->data = NULL;
		c->weight = num;
		for (j = 0; j < num; j++) {
			if (tmp_col[j] >= max_nrows) {
				logprintf(obj, "error: matrix column %u contains row %u >= %u\n",
					i, tmp_col[j], max_nrows);
				exit(-1);
			}
		}

		/* possibly permute the row numbers */

		if (rowperm != NULL) {
			for (j = 0; j < num; j++)
				tmp_col[j] = rowperm[tmp_col[j]];

			if (num > 1) {
				qsort(tmp_col, (size_t)num,
					sizeof(uint32), compare_uint32);
			}
		}

#ifdef HAVE_MPI
		/* pull out the row numbers that belong in this MPI process */

		for (j = k = 0; j < num; j++) {
			uint32 curr_row = tmp_col[j];

			if (curr_row < num_static_rows) {
				if (start_row == 0)
					tmp_col[k++] = curr_row;
			}
			else {
				uint32 curr_resclass;

				curr_row -= num_static_rows;
				curr_resclass = curr_row % mpi_nrows;

				if (curr_resclass == mpi_resclass) {
					tmp_col[k] = curr_row / mpi_nrows;
					if (start_row == 0)
						tmp_col[k] += num_static_rows;
					k++;
				}
			}
		}
		c->weight = k;

		if (start_row == 0) {
			for (j = 0; j < dense_row_words; j++)
				tmp_col[k + j] = tmp_col[num + j];
			k += dense_row_words;
		}
#endif
		if (k > 0) {
			c->data = (uint32 *)xmalloc(k * sizeof(uint32));
			memcpy(c->data, tmp_col, k * sizeof(uint32));
		}
	}

	file_cache_free(&file_cache);
	if (ferror(matrix_fp) || fclose(matrix_fp) != 0) {
		logprintf(obj, "error: can't finalize matrix read\n");
		exit(-1);
	}
	*cols_out = cols;
	*ncols_out = ncols;
	*nrows_out = nrows;
	*dense_rows_out = (start_row == 0) ? dense_rows : 0;
	if (read_submatrix) {
		*max_nrows_out = max_nrows;
		*start_row_out = start_row;
		*max_ncols_out = max_ncols;
		*start_col_out = start_col;
	}
}

/*--------------------------------------------------------------------*/
void dump_dependencies(msieve_obj *obj,
			uint64 *deps, uint32 ncols) {

	char buf[256];
	FILE *deps_fp;

	/* we allow up to 64 dependencies, even though the
	   average case will have (64 - POST_LANCZOS_ROWS) */

	sprintf(buf, "%s.dep", obj->savefile.name);
	deps_fp = fopen(buf, "wb");
	if (deps_fp == NULL) {
		logprintf(obj, "error: can't open deps file\n");
		exit(-1);
	}

	if (fwrite(deps, sizeof(uint64), (size_t)ncols, deps_fp) != ncols ||
	    fflush(deps_fp) != 0 || ferror(deps_fp) || fclose(deps_fp) != 0) {
		logprintf(obj, "error: can't finalize deps file\n");
		exit(-1);
	}
}

