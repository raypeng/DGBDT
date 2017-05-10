
EXECUTABLE := main

CU_FILES   := decision_tree_util.cu

CU_DEPS    :=

CC_FILES   := main.cpp decision_tree.cpp dataset.cpp

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -std=c++11 -fopenmp -I/usr/local/depot/cuda-8.0/include/ -I/usr/local/cuda/include/ -g


NVCCFLAGS=-O3 -m64 -std=c++11 --gpu-architecture compute_35 -I/usr/local/depot/cuda-8.0/include/ -I/usr/local/cuda/include/ -g

ifneq ($(wildcard /opt/cuda-8.0/.*),)
# Latedays
LDFLAGS=-L/opt/cuda-8.0/lib64/ -lcudart
else
# GHC
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/decision_tree.o $(OBJDIR)/dataset.o $(OBJDIR)/decision_tree_util.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
