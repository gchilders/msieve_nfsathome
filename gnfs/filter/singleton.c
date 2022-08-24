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

/*--------------------------------------------------------------------*/
void nfs_write_lp_file(msieve_obj *obj, factor_base_t *fb,
			filter_t *filter, uint32 max_relations,
			uint32 pass) {

	/* read through the relation file and form a packed 
	   array of relation_ideal_t structures. This is the
	   first step to get the NFS relations into the 
	   algorithm-independent form that the rest of the
	   filtering will use */

	uint32 i;
	savefile_t *savefile = &obj->savefile;
	FILE *relation_fp;
	FILE *final_fp;
	char *buf;
	size_t header_words;
	uint32 next_relation;
	uint32 curr_relation;
	uint32 *my_curr_relation;
	uint32 num_relations;
	hashtable_t unique_ideals;
	uint32 *tmp_factor_size;
	relation_t *tmp_relation;
	relation_ideal_t packed_ideal;
	uint32 have_skip_list = (pass == 0);
	mpz_t *scratch;
	relation_lp_t *tmp_ideal;
	
	uint32 batch = 1024 * obj->num_threads;
	uint32 num_relations_read;
	int32 *status;

	if (batch < 1) batch = 1;

	/* per thread variables */
	my_curr_relation = (uint32 *)malloc(batch * sizeof(uint32));
	buf = (char *)malloc(batch * LINE_BUF_SIZE * sizeof(char));
	scratch = (mpz_t *)malloc(batch * sizeof(mpz_t));
	tmp_factor_size = (uint32 *)malloc(batch * sizeof(uint32));
	tmp_relation = (relation_t *)malloc(batch * sizeof(relation_t));
	tmp_ideal = (relation_lp_t *)malloc(batch * sizeof(relation_lp_t));
	status = (int32 *)malloc(batch * sizeof(int32));

	for (i = 0; i < batch; i++) {
		tmp_relation[i].factors = (uint8 *)malloc(COMPRESSED_P_MAX_SIZE * sizeof(uint8));
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
	sprintf(buf, "%s.lp", savefile->name);
	final_fp = fopen(buf, "wb");
	if (final_fp == NULL) {
		logprintf(obj, "error: can't open output LP file\n");
		exit(-1);
	}

	hashtable_init(&unique_ideals, (uint32)WORDS_IN(ideal_t), 0);
	header_words = (sizeof(relation_ideal_t) - 
			sizeof(packed_ideal.ideal_list)) / sizeof(uint32);

	/* for each relation that survived the duplicate removal */

	curr_relation = (uint32)(-1);
	next_relation = (uint32)(-1);
	num_relations = 0;
	fread(&next_relation, (size_t)1, 
			sizeof(uint32), relation_fp);
	
	do {
		num_relations_read = 0;
		for(i = 0; i < batch; i++) {
			char *buf_i = buf + i * LINE_BUF_SIZE;
			savefile_read_line(buf_i, 
					LINE_BUF_SIZE * sizeof(char), savefile);
			if (savefile_eof(savefile)) break;
			if (buf_i[0] != '-' && !isdigit(buf_i[0])) {
				/* no relation on this line */
				i--;
				continue;
			} 
			num_relations_read++;
			curr_relation++;
			my_curr_relation[i] = curr_relation;
			if (max_relations && curr_relation >= max_relations) break;
			if (have_skip_list) {
				if (curr_relation == next_relation) {
					fread(&next_relation, sizeof(uint32), 
							(size_t)1, relation_fp);
					i--;
					num_relations_read--;
					continue;
				}
			}
			else {
				if (curr_relation < next_relation) {
					i--;
					num_relations_read--;
					continue;
				} else
					fread(&next_relation, sizeof(uint32), 
						(size_t)1, relation_fp);
			}
		}

		/* read it in */

#pragma omp parallel for
		for (i = 0; i < num_relations_read; i++) {
			char *buf_i = buf + i * LINE_BUF_SIZE;
			status[i] = nfs_read_relation(buf_i, fb, &tmp_relation[i], 
						&tmp_factor_size[i], 1,
						scratch[i], 0);

			/* get the large ideals */
			if (status[i] == 0)
				find_large_ideals(&tmp_relation[i], &tmp_ideal[i], 
							filter->filtmin_r,
							filter->filtmin_a);
		}

		for (i = 0; i < num_relations_read; i++) {			
			if (status[i] == 0) {
				uint32 j;
				num_relations++;

				packed_ideal.rel_index = my_curr_relation[i];
				packed_ideal.gf2_factors = tmp_ideal[i].gf2_factors;
				packed_ideal.ideal_count = tmp_ideal[i].ideal_count;

				/* map each ideal to a unique integer */

				for (j = 0; j < tmp_ideal[i].ideal_count; j++) {
					ideal_t *ideal = tmp_ideal[i].ideal_list + j;

					hashtable_find(&unique_ideals, ideal,
							packed_ideal.ideal_list + j,
							NULL);
				}

				/* dump the relation to disk */

				fwrite(&packed_ideal, sizeof(uint32),
					header_words + tmp_ideal[i].ideal_count, 
					final_fp);
			}
		}
	} while (num_relations_read == batch);

	/* free per thread variables */

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

	filter->num_relations = num_relations;
	filter->num_ideals = hashtable_get_num(&unique_ideals);
	filter->relation_array = NULL;
	logprintf(obj, "memory use: %.1f MB\n",
			(double)hashtable_sizeof(&unique_ideals) / 1048576);
	hashtable_free(&unique_ideals);
	savefile_close(savefile);
	fclose(relation_fp);
	fclose(final_fp);

	sprintf(buf, "%s.lp", savefile->name);
	filter->lp_file_size = get_file_size(buf);

	sprintf(buf, "%s.d", savefile->name);
	if (remove(buf) != 0) {
		logprintf(obj, "error: can't delete dup file\n");
		exit(-1);
	}
	free(buf);
}
