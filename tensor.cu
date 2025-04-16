#include <assert.h>

__global__ void vector_add_kernel(float *c, float * a, float *b, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		c[id] = a[id] + b[id];
	}
}

struct Tensor3 
{
	int d1, d2, d3;
	float *data; // Gpu data
	float *cpu_data;
};

Tensor3 tensor3_new(int d1, int d2, int d3, float *v)  // (d3, d2, d1)
{
	size_t size = d1 * d2 * d3  * sizeof(float);
	float *cpu_data =(float *) malloc(size);
	float *data; cudaMalloc(&data, size);
	if (v != NULL) {
		cudaMemcpy(data, v, size, cudaMemcpyHostToDevice);
	}
	return {
		d1, d2, d3,
		data,
		cpu_data
	};
}

Tensor3 tensor3_new(int d1, int d2, int d3) 
{
	return tensor3_new(d1, d2, d3, NULL);
}

void tensor3_free(Tensor3 t) 
{
	free(t.cpu_data);
	cudaFree(t.data);
}

void tensor3_copy_gpu_to_cpu (Tensor3 t) 
{
	size_t size = t.d1 * t.d2 * t.d3  * sizeof(float);
	cudaMemcpy(t.cpu_data, t.data, size, cudaMemcpyDeviceToHost);

}

void tensor3_copy_cpu_to_gpu (Tensor3 t) 
{
	size_t size = t.d1 * t.d2 * t.d3  * sizeof(float);
	cudaMemcpy(t.data, t.cpu_data, size, cudaMemcpyHostToDevice);
}


int int_ceil(int a, int b) {
	return (a + b-1) / b;
}

void vector_add(float *dest, float *a, float *b, int n)
{
	static const int BLOCK_SIZE = 32;
	int n_block = int_ceil(n, BLOCK_SIZE);

	vector_add_kernel<<<n_block, BLOCK_SIZE>>>(dest, a, b, n);
}

Tensor3 tensor3_add(Tensor3 a, Tensor3 b) 
{
	assert(a.d1 == b.d1 && a.d2 == b.d2 && a.d3 == b.d3);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);
	vector_add(c.data, a.data, b.data, a.d1 * a.d2 * a.d3);
	return c;
}
