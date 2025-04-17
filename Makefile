.PHONY: all test test_run

all: test build/run

test_run: test
	build/test_tensor
	build/test_nn

test: build/test_tensor build/test_nn

build/test_tensor: test_tensor.cu tensor.cu
	@mkdir -p $(dir $@)
	nvcc test_tensor.cu -o $@

build/test_nn: test_nn.cu tensor.cu nn.cu
	@mkdir -p $(dir $@)
	nvcc test_nn.cu -o $@

build/run: run.cu tensor.cu nn.cu
	@mkdir -p $(dir $@)
	nvcc run.cu -o $@

clean:
	rm -rf build
