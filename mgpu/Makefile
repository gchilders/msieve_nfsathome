
WIN = 0

ifeq ($(WIN),1)
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)/bin/nvcc"
	CUDA_LIBS = "$(CUDA_ROOT)/lib/win32/cuda.lib"
	EXT = dll
	NVCCFLAGS = -Xptxas -v -Xcudafe -\# -shared 
else
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	CUDA_LIBS = -lcuda
	EXT = so
	NVCCFLAGS = -Xptxas -v -Xcudafe -\# -shared \
			-Xcompiler -fPIC -Xcompiler -fvisibility=hidden
endif

GEN_ARCH = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 

INC = -I"$(CUDA_ROOT)/include" -Iinclude -I. 

SRC = \
      spmv_engine.cu \
	$(wildcard src/*.cpp) \
	$(wildcard src/*.cu)

DEPS = ./Makefile $(SRC) \
	spmv_engine.h \
	$(wildcard include/*.h) \
	$(wildcard include/*.cuh)

LIBNAME = spmv_engine

all: $(LIBNAME).$(EXT)
	touch built

clean :
	rm -f  *.$(EXT) *.lib *.exp *.dll built

$(LIBNAME).$(EXT) : $(DEPS)
	$(NVCC) $(GEN_ARCH) -o $@ $(SRC) $(NVCCFLAGS) $(INC) -O3  

