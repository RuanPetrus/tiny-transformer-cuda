.PHONY: all test clean

all: build/transformer

CFLAGS="-Wall,-Wextra"
NVCC_FLAGS=-std=c++17

build/transformer: transformer.cu
	@mkdir -p $(dir $@)
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) $^ -o $@

build/test_against_numpy: test_against_numpy.cu transformer.cu
	@mkdir -p $(dir $@)
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) test_against_numpy.cu -o $@

test: build/test_against_numpy test_against_numpy.py
	python3 test_against_numpy.py
	./build/test_against_numpy

clean:
	rm -rf build
