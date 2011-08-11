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

#ifndef _STAGE1_CORE_CPU_H_
#define _STAGE1_CORE_CPU_H_

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)1 << 27)
#define SPECIAL_Q_SCALE 5

uint32 sieve_lattice_cpu_core(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);

#endif /* !_STAGE1_CORE_CPU_H_ */
