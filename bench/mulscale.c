/* Does GMP multiplication throughput on this machine keep rising past the
   handful of concurrent multiplies msieve's square root can already issue?
 *
 * The Newton iteration in the NFS square root parallelises only across the
 * degree of the algebraic polynomial -- 6 for a sextic -- so it keeps about
 * 6 large multiplications in flight and no more. Splitting each one with a
 * level of Karatsuba would put ~18 in flight, at the cost of doing 1.5x the
 * total work. That trade only pays if 18-way throughput beats 6-way by more
 * than 1.5x, which depends entirely on how many cores the machine has.
 *
 *   cc -O2 -fopenmp -o mulscale mulscale.c -lgmp
 *   ./mulscale [bits] [maxthreads]
 *
 * bits defaults to 64,000,000 (8 MB operands), which is the right order for
 * the later Newton iterations of a large job without needing the memory a
 * real one would. maxthreads defaults to the core count.
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

static double run(mpz_t *a, mpz_t *b, mpz_t *r, int np, int reps) {
	int rep, i;
	double best = 1e30;
	for (rep = 0; rep < reps; rep++) {
		double t = omp_get_wtime();
#pragma omp parallel for schedule(dynamic,1) num_threads(np)
		for (i = 0; i < np; i++)
			mpz_mul(r[i], a[i], b[i]);
		t = omp_get_wtime() - t;
		if (t < best) best = t;
	}
	return best;
}

int main(int argc, char **argv) {
	unsigned long bits = (argc > 1) ? strtoul(argv[1], NULL, 10) : 64000000UL;
	int maxt = (argc > 2) ? atoi(argv[2]) : omp_get_num_procs();
	int i, np, reps = 3;
	double base, t6, t18;
	mpz_t *a, *b, *r;
	gmp_randstate_t st;

	if (maxt < 18) maxt = 18;            /* need 18 for the verdict */
	a = malloc(maxt * sizeof(mpz_t));
	b = malloc(maxt * sizeof(mpz_t));
	r = malloc(maxt * sizeof(mpz_t));
	gmp_randinit_default(st); gmp_randseed_ui(st, 7);
	for (i = 0; i < maxt; i++) {
		mpz_init(a[i]); mpz_init(b[i]); mpz_init(r[i]);
		mpz_urandomb(a[i], st, bits); mpz_urandomb(b[i], st, bits);
	}
	printf("cores reported: %d    operands: %lu bits (%.1f MB each)"
		"    memory: ~%.1f GB\n\n",
		omp_get_num_procs(), bits, bits/8.0/1048576.0,
		maxt * 4.0 * bits / 8.0 / (1024.0*1024*1024));

	printf("concurrent   wall      throughput vs 1\n");
	base = run(a, b, r, 1, reps);
	for (np = 1; np <= maxt; np *= 2) {
		double t = (np == 1) ? base : run(a, b, r, np, reps);
		printf("%8d   %7.3fs   %6.2fx\n", np, t, (base * np) / t);
	}

	t6  = run(a, b, r, 6,  reps);
	t18 = run(a, b, r, 18, reps);
	printf("\n 6 concurrent (what the Newton iteration issues today): "
		"%.2fx throughput\n", (base * 6) / t6);
	printf("18 concurrent (what one Karatsuba level would issue):   "
		"%.2fx throughput\n", (base * 18) / t18);
	printf("\ngain from 6 -> 18 in flight: %.2fx   (needs to beat 1.50x, "
		"the extra work Karatsuba costs)\n", ((base*18)/t18) / ((base*6)/t6));
	printf("VERDICT: %s\n", (((base*18)/t18) / ((base*6)/t6) > 1.5)
		? "worth wiring in -- more cores than the square root can currently use"
		: "not worth it on this machine -- the extra work exceeds the extra throughput");
	return 0;
}
