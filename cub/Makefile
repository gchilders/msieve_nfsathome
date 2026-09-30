
WIN = 0
WIN64 = 0

ifeq ($(WIN),1)
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)/bin/nvcc"
	EXT = dll
ifeq ($(WIN64),1)
	NVCCFLAGS += -m64
else
	NVCCFLAGS += -m32
endif
	NVCCFLAGS += -Xptxas -v -Xcudafe -\# -shared -Xcompiler /fp:strict
else
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	EXT = so
	NVCCFLAGS += -Xptxas -v -Xcudafe -\# -shared -Xcompiler -ffloat-store \
			-Xcompiler -fPIC -Xcompiler -fvisibility=hidden
endif

# sm is the SM number(s) to build for, e.g. sm=60, sm=86, sm=90, sm=120.
# For multiple architectures, pass a comma-separated list: sm=86,90,120.
#
# This must be the bare SM number (matching nvcc's `sm_NN` naming), not the
# old trailing-zero format. Compute capability 12.0 (Blackwell, e.g. RTX 5070)
# is sm=120, not sm=1200.

COMMA = ,
ifdef sm
    SM_LIST := $(subst $(COMMA), ,$(sm))
else
    SM_LIST := 60
endif

SM_TARGETS := $(foreach s,$(SM_LIST),-gencode=arch=compute_$(s),code=\"sm_$(s),compute_$(s)\")

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

CUB_DEPS = 	$(call rwildcard, cub,*.cuh)

INC = -I. -I"$(CUDA_ROOT)/include"

DEPS = ./Makefile \
	sort_engine.cu \
	sort_engine.h \
	spmv_engine.cu \
	spmv_engine.h \
	$(CUB_DEPS)

LIBNAME = sort_engine
LIBNAME2 = spmv_engine

all: $(LIBNAME).$(EXT) $(LIBNAME2).$(EXT)
	touch built

clean :
	rm -f  *.dll *.so *.lib *.exp built

$(LIBNAME).$(EXT) : $(DEPS)
	$(NVCC) $(SM_TARGETS) -o $@ sort_engine.cu $(NVCCFLAGS) $(INC) -O3

$(LIBNAME2).$(EXT) : $(DEPS)
	$(NVCC) $(SM_TARGETS) -o $@ spmv_engine.cu $(NVCCFLAGS) -DVBITS=$(VBITS) $(INC) -O3

