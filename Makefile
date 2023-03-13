NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
LDFLAGS  := -lm
EXES     := hw3-2 hw3-1 hw3-3

CC = gcc
CXX = g++
CXXFLAGS = -O3 -fopenmp
CFLAGS = -O3 -lm -fopenmp

alls: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
