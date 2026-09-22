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

typedef struct {
	uint32 index;
	uint32 count;
} row_count_t;


static int compare_row_count(const void *x, const void *y) {
	row_count_t *xx = (row_count_t *)x;
	row_count_t *yy = (row_count_t *)y;
	return yy->count - xx->count;
}

static int compare_weight(const void *x, const void *y) {
	la_col_t *xx = (la_col_t *)x;
	la_col_t *yy = (la_col_t *)y;
	return xx->weight - yy->weight;
}

/* Sorting each column's row numbers is the single most expensive thing
   reduce_matrix() does: qsort() costs about 7000 cycles to order the ~90
   entries of one column, nearly all of it the indirect call per comparison
   and the size-generic element swap. A specialised sort of uint32 with the
   comparison inlined does the same job for a fraction of that, and the
   result is identical -- these are just row numbers in increasing order. */

#define SORT_INSERT_CUTOFF 20

static void sort_uint32(uint32 *a, uint32 n) {

	while (n > SORT_INSERT_CUTOFF) {
		uint32 pivot, i, j;
		uint32 m = n / 2;

		/* median of first, middle and last, moved to a[0] */

		if (a[m] < a[0]) { pivot = a[m]; a[m] = a[0]; a[0] = pivot; }
		if (a[n-1] < a[0]) { pivot = a[n-1]; a[n-1] = a[0]; a[0] = pivot; }
		if (a[m] < a[n-1]) { pivot = a[m]; a[m] = a[n-1]; a[n-1] = pivot; }
		pivot = a[n-1];

		i = 0;
		j = n - 1;
		for (;;) {
			while (a[++i] < pivot)
				;
			while (a[--j] > pivot)
				;
			if (i >= j)
				break;
			{ uint32 t = a[i]; a[i] = a[j]; a[j] = t; }
		}
		{ uint32 t = a[i]; a[i] = a[n-1]; a[n-1] = t; }

		/* recurse into the smaller side, loop on the larger, so the
		   stack stays O(log n) whatever the input looks like */

		if (i < n - i - 1) {
			sort_uint32(a, i);
			a += i + 1;
			n -= i + 1;
		}
		else {
			sort_uint32(a + i + 1, n - i - 1);
			n = i;
		}
	}

	{
		uint32 i, j;

		for (i = 1; i < n; i++) {
			uint32 v = a[i];
			for (j = i; j > 0 && a[j-1] > v; j--)
				a[j] = a[j-1];
			a[j] = v;
		}
	}
}

/*------------------------------------------------------------------*/
uint64 count_matrix_nonzero(msieve_obj *obj,
			uint32 nrows, uint32 num_dense_rows,
			uint32 ncols, la_col_t *cols) {

	uint32 i, j;
	uint64 total_weight;
	uint64 sparse_weight;
	size_t mem_use;

	mem_use = ncols * (sizeof(la_col_t) +
		sizeof(uint32) * ((num_dense_rows + 31) / 32));

	for (i = total_weight = sparse_weight = 0; i < ncols; i++) {
		uint32 w = cols[i].weight;
		total_weight += w;
		sparse_weight += w;
		mem_use += w * sizeof(uint32);
	}

	if (num_dense_rows > 0) {
		for (i = 0; i < ncols; i++) {
			uint32 *dense_rows = cols[i].data + cols[i].weight;
			for (j = 0; j < num_dense_rows; j++) {
				if (dense_rows[j / 32] & (1 << (j % 32)))
					total_weight++;
			}
		}
	}

	logprintf(obj, "matrix is %u x %u (%.1lf MB) with "
			"weight %" PRIu64 " (%5.2lf/col)\n", 
				nrows, ncols, 
				(double)mem_use / 1048576,
				total_weight, 
				(double)total_weight / ncols);
	logprintf(obj, "sparse part has weight %" PRIu64 " (%5.2lf/col)\n", 
				sparse_weight, 
				(double)sparse_weight / ncols);
	return sparse_weight;
}

/*------------------------------------------------------------------*/
#define MAX_COL_WEIGHT 1000

static void combine_cliques(uint32 num_dense_rows, 
			uint32 *ncols_out, la_col_t *cols, 
			row_count_t *counts, uint32 num_rows,
			uint32 *witness, uint32 *col_id, uint8 *col_bad) {

	uint32 i, j;
	uint32 ncols = *ncols_out;
	uint32 dense_row_words = (num_dense_rows + 31) / 32;

	uint32 num_merged;
	uint32 merge_array[MAX_COL_WEIGHT];
	uint32 touched[MAX_COL_WEIGHT];
	uint32 num_touched;
	uint32 id0, id1;

	/* For each row, mark the last column encountered that contains a
	   nonzero entry in that row. Writing i in increasing order leaves
	   each row holding the largest column that contains it, so the
	   same answer comes out of a parallel maximum -- but only if the
	   starting value cannot win, and the prologue's row sort leaves
	   index holding a stale permutation, so clear it first. */

	{
		int32 ci;
		int32 num_cols = (int32)ncols;
		int32 ri;
		int32 nrows = (int32)num_rows;

#pragma omp parallel for schedule(static)
		for (ri = 0; ri < nrows; ri++)
			counts[ri].index = 0;

#pragma omp parallel for schedule(dynamic, 256)
		for (ci = 0; ci < num_cols; ci++) {
			la_col_t *c = cols + ci;
			uint32 k2;

			for (k2 = 0; k2 < c->weight; k2++) {
				uint32 *p = &counts[c->data[k2]].index;
				uint32 seen = *p;

				while (seen < (uint32)ci) {
#if defined(__GNUC__) || defined(__clang__)
					if (__sync_bool_compare_and_swap(p,
							seen, (uint32)ci))
						break;
					seen = *p;
#else
					*p = (uint32)ci;
					break;
#endif
				}
			}
		}
	}

	/* for each column */

	for (i = 0; i < ncols; i++) {
		la_col_t *c0;
		la_col_t *c1 = cols + i;
		uint32 clique_base = (uint32)(-1);

		if (c1->data == NULL)
			continue;

		/* if the column hits to a row of weight 2, and the
		   other column containing this row is distinct and
		   not previously merged */

		for (j = 0; j < c1->weight; j++) {
			row_count_t *curr_clique = counts + c1->data[j];
			if (curr_clique->count == 2) {
				clique_base = curr_clique->index;
				break;
			}
		}

		if (clique_base == (uint32)(-1) || clique_base == i)
			continue;

		c0 = cols + clique_base;
		if (c0->data == NULL || 
		    c0->weight + c1->weight >= MAX_COL_WEIGHT)
			continue;

		/* remove c0 and c1 from the row counts. Both columns stop
		   holding these rows, so both drop out of their witnesses;
		   note which rows to look at once the counts have settled */

		id0 = col_id[clique_base];
		id1 = col_id[i];
		num_touched = 0;
		for (j = 0; j < c0->weight; j++) {
			counts[c0->data[j]].count--;
			witness[c0->data[j]] ^= id0;
			touched[num_touched++] = c0->data[j];
		}
		for (j = 0; j < c1->weight; j++) {
			counts[c1->data[j]].count--;
			witness[c1->data[j]] ^= id1;
			touched[num_touched++] = c1->data[j];
		}

		/* merge column c1 into column c0. First merge the
		   nonzero entries */
		num_merged = merge_relations(merge_array, 
						c0->data, c0->weight,
						c1->data, c1->weight);
		for (j = 0; j < dense_row_words; j++) {
			merge_array[num_merged + j] = c0->data[c0->weight+j] ^
						      c1->data[c1->weight+j];
		}
		free(c0->data);
		c0->data = (uint32 *)xmalloc((num_merged + 
					dense_row_words) * sizeof(uint32));
		memcpy(c0->data, merge_array, (num_merged + 
					dense_row_words) * sizeof(uint32));
		c0->weight = num_merged;

		/* then combine the two lists of relation numbers */

		c0->cycle.list = (uint32 *)xrealloc(c0->cycle.list, 
					(c0->cycle.num_relations +
					 c1->cycle.num_relations) *
					sizeof(uint32));
		memcpy(c0->cycle.list + c0->cycle.num_relations,
			c1->cycle.list, c1->cycle.num_relations * 
					sizeof(uint32));
		c0->cycle.num_relations += c1->cycle.num_relations;

		/* add c0 back into the row counts */

		for (j = 0; j < c0->weight; j++) {
			row_count_t *curr_row = counts + c0->data[j];
			curr_row->count++;
			curr_row->index = clique_base;
			witness[c0->data[j]] ^= id0;
		}

		/* the counts have settled: a row left with one column makes
		   that column deletable, and its witness names it. c0 also
		   has different rows now, so its own test has to be redone */

		for (j = 0; j < num_touched; j++) {
			uint32 r = touched[j];
			if (counts[r].count == 1)
				col_bad[witness[r]] = 1;
		}

		/* c0's flag is recomputed rather than added to. Marks made
		   from row counts can never go stale, because a count only
		   ever falls, but the test on c0's largest row can: a column
		   marked for holding nothing past POST_LANCZOS_ROWS stops
		   qualifying the moment a merge gives it a larger row, and
		   leaving the mark set would delete a column the original
		   would have kept */

		col_bad[id0] = 0;
		if (c0->weight == 0 ||
		    c0->data[c0->weight - 1] < POST_LANCZOS_ROWS) {
			col_bad[id0] = 1;
		}
		else {
			for (j = 0; j < c0->weight; j++) {
				if (counts[c0->data[j]].count == 1) {
					col_bad[id0] = 1;
					break;
				}
			}
		}

		/* kill off c1 */

		free(c1->data);
		c1->data = NULL;
		free(c1->cycle.list);
		c1->cycle.list = NULL;
	}


	/* squeeze out the merged columns from the list */

	for (i = j = 0; i < ncols; i++) {
		if (cols[i].data != NULL) {
			col_id[j] = col_id[i];
			cols[j++] = cols[i];
		}
	}
	*ncols_out = j;
}

/*------------------------------------------------------------------*/
uint64 reduce_matrix(msieve_obj *obj, uint32 *nrows, 
		uint32 num_dense_rows, uint32 *ncols, 
		la_col_t *cols, uint32 num_excess) {

	/* Perform light filtering on the nrows x ncols
	   matrix specified by cols[]. The processing here is
	   limited to collapsing cliques, deleting columns that 
	   contain a singleton row, deleting empty rows, and then 
	   deleting the heaviest columns until the matrix has a 
	   few more columns than rows. Because deleting a column 
	   reduces the counts in several different rows, the process 
	   must iterate to convergence.
	   
	   Note that deleting singleton rows is not intended to 
	   make the Lanczos iteration run any faster (though it will); 
	   it's just that if we don't go to this trouble and the matrix
	   has many zero rows, then Lanczos iteration could fail 
	   to find any nontrivial dependencies. I've also seen cases
	   where cliques *must* be merged in order to find nontrivial
	   dependencies; this seems to happen for matrices that are large
	   and very sparse */

	uint32 r, c, i, j, k;
	uint32 passes;
	row_count_t *counts, *old_counts;
	uint32 reduced_rows;
	uint32 reduced_cols;
	uint32 *witness;
	uint32 *col_id;
	uint8 *col_bad;
	uint32 prune_cliques = (*ncols >= MIN_POST_LANCZOS_DIM);
	uint64 sparse_weight;

	/* sort the columns in order of increasing weight */

	qsort(cols, (size_t)(*ncols), sizeof(la_col_t), compare_weight);

	/* count the number of nonzero entries in each row */

	reduced_rows = *nrows;
	reduced_cols = *ncols;
	passes = 0;

	witness = (uint32 *)xcalloc((size_t)reduced_rows, sizeof(uint32));
	col_id = (uint32 *)xmalloc((size_t)reduced_cols * sizeof(uint32));
	col_bad = (uint8 *)xcalloc((size_t)reduced_cols, sizeof(uint8));
	for (i = 0; i < reduced_cols; i++)
		col_id[i] = i;

	old_counts = (row_count_t *)xmalloc((size_t)reduced_rows *
					sizeof(row_count_t));
	counts = (row_count_t *)xmalloc((size_t)reduced_rows *
					sizeof(row_count_t));
	for (i = 0; i < reduced_rows; i++) {
		old_counts[i].index = i;
		old_counts[i].count = 0;
	}
	for (i = 0; i < reduced_cols; i++) {
		for (j = 0; j < cols[i].weight; j++)
			old_counts[cols[i].data[j]].count++;
	}
	/* permute the row numbers so that the most dense rows
	   are first, empty rows are squeezed out, and the remaining
	   rows are sorted within each column. Doing this here is 
	   not very intuitive, because rows and columns have yet to 
	   actually be pruned. We do this here because we will also 
	   be deleting some almost-empty columns, and cannot do so 
	   if the row numbers change afterwards. The dense rows are
	   unaffected */

	memcpy(counts, old_counts, reduced_rows * sizeof(row_count_t));
	qsort(counts + num_dense_rows, 
			(size_t)(reduced_rows - num_dense_rows), 
			sizeof(row_count_t), compare_row_count);
	for (i = j = num_dense_rows; i < reduced_rows; i++) {
		if (counts[i].count) {
			counts[j].count = counts[i].count;
			old_counts[counts[i].index].index = j;
			j++;
		}
	}
	reduced_rows = j;

	/* each column is renumbered and sorted independently of every
	   other, and old_counts is only read here */

	{
		int32 ci;
		int32 num_cols = (int32)reduced_cols;

#pragma omp parallel for schedule(dynamic, 256)
		for (ci = 0; ci < num_cols; ci++) {
			la_col_t *col = cols + ci;
			uint32 k2;

			for (k2 = 0; k2 < col->weight; k2++) {
				uint32 r = old_counts[col->data[k2]].index;

				col->data[k2] = r;

				/* XOR of the columns holding this row. Once only
				   one is left the XOR is that column, which is
				   what makes a singleton row name its own
				   column without a reverse index */

#if defined(__GNUC__) || defined(__clang__)
				__sync_fetch_and_xor(witness + r, (uint32)ci);
#else
#pragma omp atomic update
				witness[r] ^= (uint32)ci;
#endif
			}
			sort_uint32(col->data, col->weight);
		}
	}
	free(old_counts);

	/* seed the deletable set: a row held by exactly one column makes
	   that column deletable, and so do the two tests that depend only
	   on a column's own contents */

	for (i = 0; i < *nrows; i++) {
		if (counts[i].count == 1)
			col_bad[witness[i]] = 1;
	}
	for (i = 0; i < reduced_cols; i++) {
		la_col_t *col = cols + i;

		if (col->weight == 0 || (prune_cliques &&
		    col->data[col->weight - 1] < POST_LANCZOS_ROWS))
			col_bad[col_id[i]] = 1;
	}

	/* prune rows and columns iteratively until the matrix 
	   has the correct aspect ratio */

	do {
		r = reduced_rows;

		/* remove any bad columns, then update the row counts
		   to reflect the missing column. Iterate until
		   no more columns can be deleted */

		do {
			c = reduced_cols;

			/* delete columns that are empty or contain 
			   a singleton row */

			/* Which columns these are is already known: a column
			   became deletable at the moment one of its rows was
			   left with nowhere else to live, and that row named it
			   through its witness. Counts here only ever fall, so a
			   column that was marked stays deletable, and the marks
			   made behind the sweep are picked up on the next one --
			   which is how the original, re-reading every entry,
			   behaved as well */

			for (i = j = 0; i < reduced_cols; i++) {
				la_col_t *col = cols + i;
				uint32 weight = col->weight;
				uint32 id = col_id[i];

				if (col_bad[id]) {

					for (k = 0; k < weight; k++) {
						uint32 r = col->data[k];

						counts[r].count--;
						witness[r] ^= id;
						if (counts[r].count == 1)
							col_bad[witness[r]] = 1;
					}
					free(col->data);
					free(col->cycle.list);
				}
				else {
					col_id[j] = id;
					cols[j++] = cols[i];
				}
			}
			reduced_cols = j;

			/* if the matrix is big enough, collapse most 
			   of the cliques that it contains */

			if (prune_cliques) {
				combine_cliques(num_dense_rows, 
						&reduced_cols, 
						cols, counts, *nrows,
						witness, col_id, col_bad);
			}
		} while (c != reduced_cols);
	
		/* count the number of rows that contain a
		   nonzero entry. Ignore the row indices associated
		   with the dense rows */

		for (i = reduced_rows = num_dense_rows; i < *nrows; i++) {
			if (counts[i].count)
				reduced_rows++;
		}

		/* Because deleting a column reduces the weight
		   of many rows, the number of nonzero rows may
		   be much less than the number of columns. Delete
		   more columns until the matrix has the correct
		   aspect ratio. Columns at the end of cols[] are
		   the heaviest, so delete those (and update the 
		   row counts again) */

		if (reduced_cols > reduced_rows + num_excess) {
			for (i = reduced_rows + num_excess;
					i < reduced_cols; i++) {

				la_col_t *col = cols + i;
				uint32 id = col_id[i];

				for (j = 0; j < col->weight; j++) {
					uint32 r = col->data[j];

					counts[r].count--;
					witness[r] ^= id;
					if (counts[r].count == 1)
						col_bad[witness[r]] = 1;
				}
				free(col->data);
				free(col->cycle.list);
			}
			reduced_cols = reduced_rows + num_excess;
		}

		/* if any columns were deleted in the previous step,
		   then the matrix is less dense and more columns
		   can be deleted; iterate until no further deletions
		   are possible */

		passes++;

	} while (r != reduced_rows);

	/* if the linear system was underdetermined going
	   into this routine, the pruning above will likely
	   have destroyed the matrix. Linear algebra clearly
	   cannot proceed in this case */

	if (reduced_cols == 0) {
		free(counts);
		free(witness);
		free(col_id);
		free(col_bad);
		*nrows = reduced_rows;
		*ncols = reduced_cols;
		return 0;
	}

	logprintf(obj, "filtering completed in %u passes\n", passes);
	sparse_weight = count_matrix_nonzero(obj, reduced_rows, num_dense_rows,
				reduced_cols, cols);

	/* change the row indices to remove rows with zero weight */
	
	for (i = j = num_dense_rows; i < *nrows; i++) {
		if (counts[i].count)
			counts[i].index = j++;
	}
	for (i = 0; i < reduced_cols; i++) {
		la_col_t *col = cols + i;
		for (j = 0; j < col->weight; j++) {
			col->data[j] = counts[col->data[j]].index;
		}
	}

	/* make heavy columns alternate with light columns, to
	   balance the distribution of nonzeros */

	for (i = 1, j = reduced_cols - 2; i < j; i += 2, j -= 2) {
		la_col_t tmp = cols[i];
		cols[i] = cols[j];
		cols[j] = tmp;
	}

	free(counts);
	free(witness);
	free(col_id);
	free(col_bad);
	*nrows = reduced_rows;
	*ncols = reduced_cols;
	return sparse_weight;
}
