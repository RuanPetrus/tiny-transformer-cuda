.PHONY: all test

all: test

test: build/test_tensor

build/test_tensor: test_tensor.cu tensor.cu
	@mkdir -p $(dir $@)
	nvcc test_tensor.cu -o $@

clean:
	rm -rf build
