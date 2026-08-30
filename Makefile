# --------------------------------------------------------------------
# This source distribution is placed in the public domain by its author,
# Jason Papadopoulos. You may use it for any purpose, free of charge,
# without having to notify anyone. I disclaim any responsibility for any
# errors.
# 
# Optionally, please be nice and tell me if you find this source to be
# useful. Again optionally, if you add to the functionality present here
# please consider making those additions public too, so that others may 
# benefit from your work.	
#
#  $Id$
# --------------------------------------------------------------------

# override from command line
WIN = 0
WIN64 = 0
VBITS = 64
OMP = 1

# gcc with basic optimization (-march flag could
# get overridden by architecture-specific builds)
CC = gcc
CXX = g++
HOSTCC ?= $(CC)
WARN_FLAGS = -Wall -W
OPT_FLAGS = -O3 -g -march=native \
	    -D_FILE_OFFSET_BITS=64 -DNDEBUG -D_LARGEFILE64_SOURCE -DVBITS=$(VBITS)

# use := instead of = so we only run the following once
SVN_VERSION := $(shell svnversion .)
ifeq ($(SVN_VERSION),)
	SVN_VERSION := unknown
endif

CFLAGS = $(OPT_FLAGS) $(MACHINE_FLAGS) $(WARN_FLAGS) \
	 	-DMSIEVE_SVN_VERSION="\"$(SVN_VERSION)\"" \
		-I. -Iaprcl -Iinclude -Ignfs -Ignfs/poly -Ignfs/poly/stage1

# tweak the compile flags

ifeq ($(OMP),1)
	CFLAGS += -fopenmp -DHAVE_OMP
endif
ifeq ($(ECM),1)
	CFLAGS += -DHAVE_GMP_ECM
	LIBS += -lecm
endif
ifeq ($(WIN),1)

else
	LIBS += -ldl
endif
ifdef CUDA
# Preserve the historical CUDA=cc interface for single-architecture builds,
# and allow CUDA_ARCHS to request one executable containing several native
# cubins. CUDA_ARCHS may be space- or comma-separated. The first architecture
# is also the default PTX virtual architecture unless CUDA_PTX_ARCH is set.
ifeq ($(CUDA),1)
	CUDA_ARCHS ?= 80 86 89 90 120
else
	CUDA_ARCHS ?= $(CUDA)
endif
comma := ,
CUDA_ARCH_LIST := $(strip $(subst $(comma), ,$(CUDA_ARCHS)))
CUDA_PTX_ARCH ?= $(firstword $(CUDA_ARCH_LIST))
CUDA_GENCODE := $(foreach arch,$(CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))
CUDA_FATBIN_GENCODE := $(CUDA_GENCODE) \
	-gencode arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)

ifeq ($(WIN),1)
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)/bin/nvcc"

ifeq ($(WIN64),1)
	CUDA_DRIVER_LIBS = "$(CUDA_ROOT)/lib/x64/cuda.lib"
	CUDA_RUNTIME_LIBS = "$(CUDA_ROOT)/lib/x64/cudart_static.lib"
else
	CUDA_DRIVER_LIBS = "$(CUDA_ROOT)/lib/win32/cuda.lib"
	CUDA_RUNTIME_LIBS = "$(CUDA_ROOT)/lib/win32/cudart_static.lib"
endif
	CUDA_HOST_FLAGS = -Xcompiler /fp:strict
	BIN2C = msieve_bin2c.exe
	# The historical WIN=1 Makefile mixes MinGW C with NVCC/MSVC C++.
	# Keep its DSO-based CUDA packaging by default; Visual Studio projects
	# are also unchanged. CUDA_SINGLE_BINARY=1 may be used with a compatible
	# all-MSVC/COFF toolchain.
	CUDA_SINGLE_BINARY ?= 0
else
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	CUDA_DRIVER_LIBS = -lcuda
	CUDA_RUNTIME_LIBS = -L"$(CUDA_ROOT)/lib64" -lcudart_static -ldl -lrt
	CUDA_HOST_FLAGS = -Xcompiler -ffloat-store
	BIN2C = msieve_bin2c
	CUDA_SINGLE_BINARY ?= 1
endif
	CFLAGS += -I"$(CUDA_ROOT)/include" -Icub -DHAVE_CUDA
ifeq ($(CUDA_SINGLE_BINARY),1)
	CFLAGS += -DMSIEVE_CUDA_SINGLE_BINARY
	LIBS += $(CUDA_RUNTIME_LIBS) $(CUDA_DRIVER_LIBS)
else
	LIBS += $(CUDA_DRIVER_LIBS)
endif

ifeq ($(CUDAAWARE),1)
	CFLAGS += -DHAVE_CUDAAWARE_MPI
endif
endif
ifeq ($(MPI),1)
	CC = mpicc
	CXX = mpicxx
	CFLAGS += -DHAVE_MPI
endif

ifdef CUDA
ifeq ($(CUDA_SINGLE_BINARY),1)
	LINKER = $(CXX)
else
	LINKER = $(CC)
endif
else
	LINKER = $(CC)
endif
ifeq ($(BOINC),1)
	# fill in as appropriate
	BOINC_INC_DIR = .
	BOINC_LIB_DIR = .
	CFLAGS += -I$(BOINC_INC_DIR) -DHAVE_BOINC
	LIBS += -L$(BOINC_LIB_DIR) -lboinc_api -lboinc
endif
ifeq ($(NO_ZLIB),1)
	CFLAGS += -DNO_ZLIB
else
	LIBS += -lz
endif


# Note to MinGW users: the library does not use pthreads calls in
# win32 or win64, so it's safe to pull libpthread into the link line.
# Of course this does mean you have to install the minGW pthreads bundle...

LIBS += -lgmp -lm -lpthread

#---------------------------------- Generic file lists -------------------

COMMON_HDR = \
	aprcl/mpz_aprcl32.h \
	common/lanczos/lanczos.h \
	common/filter/filter.h \
	common/filter/filter_priv.h \
	common/filter/merge_util.h \
	include/batch_factor.h \
	include/common.h \
	include/cuda_xface.h \
	include/dd.h \
	include/ddcomplex.h \
	include/gmp_xface.h \
	include/integrate.h \
	include/msieve.h \
	include/mp.h \
	include/polyroot.h \
	include/thread.h \
	include/util.h

COMMON_GPU_HDR = \
	common/lanczos/gpu/lanczos_kernel.cu \
	common/lanczos/gpu/lanczos_gpu.h \
	common/lanczos/gpu/lanczos_gpu_core.h

COMMON_NOGPU_HDR = \
	common/lanczos/cpu/lanczos_cpu.h

COMMON_SRCS = \
	aprcl/mpz_aprcl32.c \
	common/filter/clique.c \
	common/filter/filter.c \
	common/filter/merge.c \
	common/filter/merge_post.c \
	common/filter/merge_pre.c \
	common/filter/merge_util.c \
	common/filter/singleton.c \
	common/lanczos/lanczos.c \
	common/lanczos/lanczos_io.c \
	common/lanczos/lanczos_matmul.c \
	common/lanczos/lanczos_pre.c \
	common/lanczos/matmul_util.c \
	common/smallfact/gmp_ecm.c \
	common/smallfact/smallfact.c \
	common/smallfact/squfof.c \
	common/smallfact/tinyqs.c \
	common/batch_factor.c \
	common/cuda_xface.c \
	common/dickman.c \
	common/driver.c \
	common/expr_eval.c \
	common/hashtable.c \
	common/integrate.c \
	common/minimize.c \
	common/minimize_global.c \
	common/mp.c \
	common/polyroot.c \
	common/prime_delta.c \
	common/prime_sieve.c \
	common/savefile.c \
	common/strtoll.c \
	common/thread.c \
	common/util.c

COMMON_GPU_SRCS = \
		common/lanczos/gpu/lanczos_matmul_gpu.c \
		common/lanczos/gpu/lanczos_vv.c

COMMON_NOGPU_SRCS = \
		common/lanczos/cpu/lanczos_matmul0.c \
		common/lanczos/cpu/lanczos_matmul1.c \
		common/lanczos/cpu/lanczos_matmul2.c \
		common/lanczos/cpu/lanczos_vv.c

ifdef CUDA
	COMMON_SRCS += $(COMMON_GPU_SRCS)
	COMMON_HDR += $(COMMON_GPU_HDR)
else
	COMMON_SRCS += $(COMMON_NOGPU_SRCS)
	COMMON_HDR += $(COMMON_NOGPU_HDR)
endif

COMMON_OBJS = $(COMMON_SRCS:.c=.o)
COMMON_GPU_OBJS = $(COMMON_GPU_SRCS:.c=.o)
COMMON_NOGPU_OBJS = $(COMMON_NOGPU_SRCS:.c=.o)

#---------------------------------- QS file lists -------------------------

QS_HDR = mpqs/mpqs.h

QS_SRCS = \
	mpqs/gf2.c \
	mpqs/mpqs.c \
	mpqs/poly.c \
	mpqs/relation.c \
	mpqs/sieve.c \
	mpqs/sieve_core.c \
	mpqs/sqrt.c

QS_OBJS = \
	mpqs/gf2.qo \
	mpqs/mpqs.qo \
	mpqs/poly.qo \
	mpqs/relation.qo \
	mpqs/sieve.qo \
	mpqs/sqrt.qo \
	mpqs/sieve_core_generic_32k.qo \
	mpqs/sieve_core_generic_64k.qo

#---------------------------------- GPU file lists -------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
CUB_DEPS := $(call rwildcard,cub/cub/,*.cuh)

# Define these unconditionally so plain `make clean` removes artifacts left
# by an earlier CUDA build even when CUDA is not specified on the clean command.
CUDA_ENGINE_OBJS = \
	cub/sort_engine.o \
	cub/spmv_engine.o

CUDA_EMBED_OBJS = \
	stage1_core_fatbin_embed.o \
	stage1_core_ptx_embed.o \
	lanczos_kernel_fatbin_embed.o \
	lanczos_kernel_ptx_embed.o

ifdef CUDA
ifeq ($(CUDA_SINGLE_BINARY),1)
GPU_OBJS = $(CUDA_ENGINE_OBJS) $(CUDA_EMBED_OBJS)
CUDA_ARCHIVE_OBJS = $(GPU_OBJS)
else
GPU_OBJS = \
	stage1_core.ptx stage1_core.fatbin \
	lanczos_kernel.ptx lanczos_kernel.fatbin \
	cub/built
CUDA_ARCHIVE_OBJS =
endif
else
GPU_OBJS =
CUDA_ARCHIVE_OBJS =
endif

#---------------------------------- NFS file lists -------------------------

NFS_HDR = \
	gnfs/filter/filter.h \
	gnfs/filter/hash64.h \
	gnfs/filter/rmap.h \
	gnfs/poly/poly.h \
	gnfs/poly/poly_skew.h \
	gnfs/poly/stage1/stage1.h \
	gnfs/poly/stage2/stage2.h \
	gnfs/sieve/sieve.h \
	gnfs/sqrt/sqrt.h \
	gnfs/gnfs.h

NFS_GPU_HDR = \
	gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu \
	gnfs/poly/stage1/stage1_core_gpu/cuda_intrinsics.h \
	gnfs/poly/stage1/stage1_core_gpu/stage1_core.h

NFS_NOGPU_HDR = \
	gnfs/poly/stage1/cpu_intrinsics.h

NFS_SRCS = \
	gnfs/poly/poly.c \
	gnfs/poly/poly_param.c \
	gnfs/poly/poly_skew.c \
	gnfs/poly/polyutil.c \
	gnfs/poly/root_score.c \
	gnfs/poly/size_score.c \
	gnfs/poly/stage1/stage1.c \
	gnfs/poly/stage1/stage1_roots.c \
	gnfs/poly/stage2/optimize.c \
	gnfs/poly/stage2/optimize_deg6.c \
	gnfs/poly/stage2/root_sieve.c \
	gnfs/poly/stage2/root_sieve_deg45_x.c \
	gnfs/poly/stage2/root_sieve_deg5_xy.c \
	gnfs/poly/stage2/root_sieve_deg6_x.c \
	gnfs/poly/stage2/root_sieve_deg6_xy.c \
	gnfs/poly/stage2/root_sieve_deg6_xyz.c \
	gnfs/poly/stage2/root_sieve_line.c \
	gnfs/poly/stage2/root_sieve_util.c \
	gnfs/poly/stage2/stage2.c \
	gnfs/filter/duplicate.c \
	gnfs/filter/filter.c \
	gnfs/filter/singleton.c \
	gnfs/sieve/sieve_line.c \
	gnfs/sieve/sieve_util.c \
	gnfs/sqrt/sqrt.c \
	gnfs/sqrt/sqrt_a.c \
	gnfs/fb.c \
	gnfs/ffpoly.c \
	gnfs/gf2.c \
	gnfs/gnfs.c \
	gnfs/relation.c

NFS_OBJS = $(NFS_SRCS:.c=.no)

NFS_GPU_SRCS = \
	gnfs/poly/stage1/stage1_sieve_gpu.c

NFS_GPU_OBJS = $(NFS_GPU_SRCS:.c=.no)

NFS_NOGPU_SRCS = \
	gnfs/poly/stage1/stage1_sieve_cpu.c

NFS_NOGPU_OBJS = $(NFS_NOGPU_SRCS:.c=.no)

ifdef CUDA
	NFS_HDR += $(NFS_GPU_HDR)
	NFS_SRCS += $(NFS_GPU_SRCS)
	NFS_OBJS += $(NFS_GPU_OBJS)
else
	NFS_HDR += $(NFS_NOGPU_HDR)
	NFS_SRCS += $(NFS_NOGPU_SRCS)
	NFS_OBJS += $(NFS_NOGPU_OBJS)
endif

#---------------------------------- make targets -------------------------

help:
	@echo "to build:"
	@echo "make all"
	@echo "add 'WIN=1 if building on windows"
	@echo "add 'WIN64=1 if building on 64-bit windows"
	@echo "add 'ECM=1' if GMP-ECM is available (enables ECM)"
	@echo "add 'CUDA=1' for Nvidia graphics card support"
	@echo "     CUDA=1 defaults to 80 86 89 90 120"
	@echo "     use CUDA_ARCHS=\"80 86 89 90\" to embed several native architectures"
	@echo "     CUDA_PTX_ARCH defaults to the first CUDA_ARCHS entry and supplies PTX fallback"
	@echo "     CUDA_SINGLE_BINARY=1 is the Unix default; =0 selects legacy external CUDA files"
	@echo "add 'MPI=1' for parallel processing using MPI"
	@echo "     add 'CUDAAWARE=1' if using CUDA-Aware MPI"
	@echo "add 'BOINC=1' to add BOINC wrapper"
	@echo "add 'NO_ZLIB=1' if you don't have zlib"
	@echo "add 'VBITS=X' for linear algebra with X-bit vectors"
	@echo "     (64, 128, 192, 256, 320, 384, 448, 512)"

all: demo.o $(COMMON_OBJS) $(QS_OBJS) $(NFS_OBJS) $(GPU_OBJS)
	rm -f libmsieve.a
	ar r libmsieve.a $(COMMON_OBJS) $(QS_OBJS) $(NFS_OBJS) $(CUDA_ARCHIVE_OBJS)
	ranlib libmsieve.a
	$(LINKER) $(CFLAGS) -o msieve $(LDFLAGS) demo.o libmsieve.a $(LIBS)

clean:
	cd cub && make clean WIN=$(WIN) WIN64=$(WIN64) && cd ..
	rm -f msieve msieve.exe demo.o libmsieve.a $(COMMON_OBJS) $(QS_OBJS) \
		$(COMMON_GPU_OBJS) $(NFS_OBJS) $(NFS_GPU_OBJS) $(NFS_NOGPU_OBJS) \
		$(CUDA_ENGINE_OBJS) $(CUDA_EMBED_OBJS) *.ptx *.fatbin *_embed.c \
		msieve_bin2c msieve_bin2c.exe

#----------------------------------------- build rules ----------------------

# common file build rules

%.o: %.c $(COMMON_HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

# QS build rules

mpqs/sieve_core_generic_32k.qo: mpqs/sieve_core.c $(COMMON_HDR) $(QS_HDR)
	$(CC) $(CFLAGS) -DBLOCK_KB=32 -DHAS_SSE2 \
		-DROUTINE_NAME=qs_core_sieve_generic_32k \
		-c -o $@ mpqs/sieve_core.c

mpqs/sieve_core_generic_64k.qo: mpqs/sieve_core.c $(COMMON_HDR) $(QS_HDR)
	$(CC) $(CFLAGS) -DBLOCK_KB=64 -DHAS_SSE2 \
		-DROUTINE_NAME=qs_core_sieve_generic_64k \
		-c -o $@ mpqs/sieve_core.c

%.qo: %.c $(COMMON_HDR) $(QS_HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

# NFS build rules

%.no: %.c $(COMMON_HDR) $(NFS_HDR)
	$(CC) $(CFLAGS) -Ignfs -c -o $@ $<

# GPU build rules

ifdef CUDA
# The two Driver-API kernel modules are built twice: a multi-architecture
# fatbin with native SASS plus PTX, and a standalone PTX image. Both are
# converted to C arrays and linked into msieve. The standalone PTX is kept
# separately because some driver/toolkit combinations have accepted raw PTX
# after rejecting an otherwise valid fatbin container.
stage1_core.ptx: $(NFS_GPU_HDR)
	$(NVCC) -arch compute_$(CUDA_PTX_ARCH) -ptx -o $@ $<

stage1_core.fatbin: $(NFS_GPU_HDR)
	$(NVCC) $(CUDA_FATBIN_GENCODE) -fatbin -o $@ $<

lanczos_kernel.ptx: $(COMMON_GPU_HDR)
	$(NVCC) -arch compute_$(CUDA_PTX_ARCH) -ptx -DVBITS=$(VBITS) -o $@ $<

lanczos_kernel.fatbin: $(COMMON_GPU_HDR)
	$(NVCC) $(CUDA_FATBIN_GENCODE) -fatbin -DVBITS=$(VBITS) -o $@ $<

ifeq ($(CUDA_SINGLE_BINARY),1)
# CUB engines are ordinary CUDA objects linked directly into the executable.
# Include native SASS for every requested architecture and PTX for forward
# compatibility, matching the module fatbins above.
cub/sort_engine.o: cub/sort_engine.cu cub/sort_engine.h $(CUB_DEPS)
	$(NVCC) $(CUDA_FATBIN_GENCODE) $(CUDA_HOST_FLAGS) -O3 \
		-I. -Icub -I"$(CUDA_ROOT)/include" -c -o $@ $<

cub/spmv_engine.o: cub/spmv_engine.cu cub/spmv_engine.h $(CUB_DEPS)
	$(NVCC) $(CUDA_FATBIN_GENCODE) $(CUDA_HOST_FLAGS) -O3 -DVBITS=$(VBITS) \
		-I. -Icub -I"$(CUDA_ROOT)/include" -c -o $@ $<

$(BIN2C): tools/bin2c.c
	$(HOSTCC) -O2 -o $@ $<

stage1_core_fatbin_embed.c: stage1_core.fatbin $(BIN2C)
	./$(BIN2C) msieve_stage1_core_fatbin $< > $@

stage1_core_ptx_embed.c: stage1_core.ptx $(BIN2C)
	./$(BIN2C) --text msieve_stage1_core_ptx $< > $@

lanczos_kernel_fatbin_embed.c: lanczos_kernel.fatbin $(BIN2C)
	./$(BIN2C) msieve_lanczos_kernel_fatbin $< > $@

lanczos_kernel_ptx_embed.c: lanczos_kernel.ptx $(BIN2C)
	./$(BIN2C) --text msieve_lanczos_kernel_ptx $< > $@

%_embed.o: %_embed.c
	$(CC) $(MACHINE_FLAGS) -c -o $@ $<
else
cub/built:
	cd cub && make WIN=$(WIN) WIN64=$(WIN64) VBITS=$(VBITS) \
		sm=$(firstword $(CUDA_ARCH_LIST))0 && cd ..
endif
endif
