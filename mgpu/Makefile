
WIN = 0

ifeq ($(WIN),1)
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)bin/nvcc"
	CUDA_LIBS = "$(CUDA_ROOT)lib/win32/cuda.lib"
	EXT = dll
	NVCCFLAGS = -Xptxas -v -Xcudafe -\# -shared -Xptxas -abi=no
else
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	CUDA_LIBS = -lcuda
	EXT = so
	NVCCFLAGS = -Xptxas -v -Xcudafe -\# -shared -Xptxas -abi=no \
			-Xcompiler -fPIC -Xcompiler -fvisibility=hidden
endif

GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" 
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 

INC = -I"$(CUDA_ROOT)include" -Iinclude -I. 

SRC = \
      scan_engine.cu \
	$(wildcard src/*.cpp)

DEPS = ./Makefile $(SRC) \
	scan_engine.h \
	$(wildcard include/*.h) \
	$(wildcard include/*.cuh)

LIBNAME = scan_engine

all: $(LIBNAME)_sm20.$(EXT) $(LIBNAME)_sm30.$(EXT) $(LIBNAME)_sm35.$(EXT)
	touch built

clean :
	rm -f  *.$(EXT) *.lib *.exp *.dll built

$(LIBNAME)_sm20.$(EXT) : $(DEPS)
	$(NVCC) $(GEN_SM20) -o $@ $(SRC) $(NVCCFLAGS) $(INC) -O3  

$(LIBNAME)_sm30.$(EXT) : $(DEPS)
	$(NVCC) $(GEN_SM30) -o $@ $(SRC) $(NVCCFLAGS) $(INC) -O3  

$(LIBNAME)_sm35.$(EXT) : $(DEPS)
	$(NVCC) $(GEN_SM35) -o $@ $(SRC) $(NVCCFLAGS) $(INC) -O3  

