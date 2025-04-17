#include <assert.h>
#include <random>
#include "kernels.cu"

#define TODO(message) do { fprintf(stderr, "%s:%d: TODO: %s\n", __FILE__, __LINE__, message); abort(); } while(0)
#define ERROR(...) do { fprintf(stderr, __VA_ARGS__); abort(); } while(0)

#define THREAD_PER_BLOCK_X (1 << 4)
#define THREAD_PER_BLOCK_Y (1 << 3)
#define THREAD_PER_BLOCK_Z (1 << 3)

static int TENSOR_GPU_LAST_SYNC = 0;

float random_normal_distribution_float()
{
	static std::random_device random_device{};
	static std::mt19937 random_generator{random_device()};
	static std::normal_distribution random_normal_distribution{0.0f, 1.0f};
	return random_normal_distribution(random_generator);
}

int int_ceil(int a, int b) 
{
	return (a + b-1) / b;
}

struct Tensor3 
{
	int d1, d2, d3;
	float *data; // Gpu data
	float *cpu_data;
	int sync;
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
		cpu_data,
		-1
	};
}

Tensor3 tensor3_new(int d1, int d2, int d3) 
{
	return tensor3_new(d1, d2, d3, NULL);
}

Tensor3 tensor3_randn(int d1, int d2, int d3) 
{

	size_t size = d1 * d2 * d3  * sizeof(float);
	float *data =(float *) malloc(size);
	for (int i = 0; i < d1*d2*d3; i++) {
		data[i] = random_normal_distribution_float();
	}
	return tensor3_new(d1, d2, d3, data);
}

void tensor3_sync(Tensor3 t) 
{
	if (t.sync >= TENSOR_GPU_LAST_SYNC) {
		cudaDeviceSynchronize();
		TENSOR_GPU_LAST_SYNC++;
	}
}

void tensor3_unsync(Tensor3 *t)
{
	t->sync = TENSOR_GPU_LAST_SYNC;
}

void tensor3_free(Tensor3 t) 
{
	tensor3_sync(t);
	free(t.cpu_data);
	cudaFree(t.data);
}

void tensor3_copy_gpu_to_cpu (Tensor3 t) 
{
	tensor3_sync(t);
	size_t size = t.d1 * t.d2 * t.d3  * sizeof(float);
	cudaMemcpy(t.cpu_data, t.data, size, cudaMemcpyDeviceToHost);

}

void tensor3_copy_cpu_to_gpu (Tensor3 t) 
{
	tensor3_sync(t);
	size_t size = t.d1 * t.d2 * t.d3  * sizeof(float);
	cudaMemcpy(t.data, t.cpu_data, size, cudaMemcpyHostToDevice);
}

void tensor3_show(Tensor3 t) 
{
	tensor3_copy_gpu_to_cpu(t);
	printf("Tensor (%d, %d, %d):\n", t.d1, t.d2, t.d3);
	for (int k = 0; k < t.d3; k++) {
		printf("[ ");
		for (int j = 0; j < t.d2; j++) {
			printf("[ ");
			for (int i = 0; i < t.d1; i++) {
				printf("%.4f ", t.cpu_data[KEY(k, j, i, t.d3, t.d2, t.d1)]);
			}
			printf("] ");
		}
		printf("]\n");
	}
}

bool tensor3_same_shape(Tensor3 a, Tensor3 b)
{
	return (a.d1 == b.d1 && a.d2 == b.d2 && a.d3 == b.d3);
}

bool tensor3_is_broadcastable(Tensor3 a, int d1, int d2, int d3)
{
	return ((a.d1 == d1 || a.d1 == 1) &&
			(a.d2 == d2 || a.d2 == 1) &&
			(a.d3 == d3 || a.d3 == 1));
}

bool tensor3_is_summable(Tensor3 a, int d1, int d2, int d3)
{
	return ((a.d1 == d1 || d1 == 1) &&
			(a.d2 == d2 || d2 == 1) &&
			(a.d3 == d3 || d3 == 1));
}

Tensor3 tensor3_broadcast(Tensor3 a, int d1, int d2, int d3) 
{
	assert(tensor3_is_broadcastable(a, d1, d2, d3));
	tensor3_sync(a);
	Tensor3 c = tensor3_new(d1, d2, d3);

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_broadcast<<<num_blocks, threads_per_block>>>(
		c.data, a.data, 
		c.d1, c.d2, c.d3, // c dimensions
		a.d1, a.d2, a.d3  // a dimensions
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_sum(Tensor3 a, int d1, int d2, int d3) 
{
	assert(tensor3_is_summable(a, d1, d2, d3));
	tensor3_sync(a);
	Tensor3 c = tensor3_new(d1, d2, d3);

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_sum<<<num_blocks, threads_per_block>>>(
		c.data, a.data, 
		c.d1, c.d2, c.d3, // c dimensions
		a.d1, a.d2, a.d3  // a dimensions
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

// op(1) == (1, 2)
// op(2) == (1, 3)
// op(3) == (2, 3)
Tensor3 tensor3_transpose(Tensor3 a, int dx, int dy) 
{
	tensor3_sync(a);
	int op = dx ^ dy;
	int d1 = a.d1, d2 = a.d2, d3 = a.d3;
	if (op == 3) {
		SWAP(d1, d2, int);
	}
	else if (op == 2) {
		SWAP(d1, d3, int);
	}
	else if (op == 1) {
		SWAP(d2, d3, int);
	}
	Tensor3 c = tensor3_new(d1, d2, d3);

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(a.d1, threads_per_block.x),
		int_ceil(a.d2, threads_per_block.y),
		int_ceil(a.d3, threads_per_block.z)
	);

	kernel_transpose<<<num_blocks, threads_per_block>>>(
		c.data, a.data, 
		a.d1, a.d2, a.d3,
		op          
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_view(Tensor3 a, int d1, int d2, int d3) 
{
	assert(d1*d2*d3 == a.d1*a.d2*a.d3);
	tensor3_sync(a);
	Tensor3 c = tensor3_new(d1, d2, d3);

	cudaMemcpy(c.data, a.data, d1*d2*d3*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_add(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	Tensor3 c = tensor3_new(max(a.d1, b.d1), max(a.d2, b.d2), max(a.d3, b.d3));
	if (!tensor3_same_shape(a, b)) {
		if (tensor3_is_broadcastable(a, b.d1, b.d2, b.d3)) {
			a = tensor3_broadcast(a, b.d1, b.d2, b.d3);
		}
		if (tensor3_is_broadcastable(b, a.d1, a.d2, a.d3)) {
			b = tensor3_broadcast(b, a.d1, a.d2, a.d3);
		}
		else {
			ERROR("Shapes (%d, %d, %d) and (%d, %d, %d) are not broadcastable\n",
			      a.d1, a.d2, a.d3, b.d1, b.d2, b.d3);
		}
	}

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_add<<<num_blocks, threads_per_block>>>(
		c.data, a.data, b.data,
		c.d1, c.d2, c.d3
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}


Tensor3 tensor3_div(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	Tensor3 c = tensor3_new(max(a.d1, b.d1), max(a.d2, b.d2), max(a.d3, b.d3));
	if (!tensor3_same_shape(a, b)) {
		if (tensor3_is_broadcastable(a, b.d1, b.d2, b.d3)) {
			a = tensor3_broadcast(a, b.d1, b.d2, b.d3);
		}
		if (tensor3_is_broadcastable(b, a.d1, a.d2, a.d3)) {
			b = tensor3_broadcast(b, a.d1, a.d2, a.d3);
		}
		else {
			ERROR("Shapes (%d, %d, %d) and (%d, %d, %d) are not broadcastable\n",
			      a.d1, a.d2, a.d3, b.d1, b.d2, b.d3);
		}
	}

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_div<<<num_blocks, threads_per_block>>>(
		c.data, a.data, b.data,
		c.d1, c.d2, c.d3
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_mul(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	Tensor3 c = tensor3_new(max(a.d1, b.d1), max(a.d2, b.d2), max(a.d3, b.d3));
	if (!tensor3_same_shape(a, b)) {
		if (tensor3_is_broadcastable(a, b.d1, b.d2, b.d3)) {
			a = tensor3_broadcast(a, b.d1, b.d2, b.d3);
		}
		if (tensor3_is_broadcastable(b, a.d1, a.d2, a.d3)) {
			b = tensor3_broadcast(b, a.d1, a.d2, a.d3);
		}
		else {
			ERROR("Shapes (%d, %d, %d) and (%d, %d, %d) are not broadcastable\n",
			      a.d1, a.d2, a.d3, b.d1, b.d2, b.d3);
		}
	}

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_mul<<<num_blocks, threads_per_block>>>(
		c.data, a.data, b.data,
		c.d1, c.d2, c.d3
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_exp(Tensor3 a) 
{
	tensor3_sync(a);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);
	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_exp<<<num_blocks, threads_per_block>>>(
		c.data, a.data,
		c.d1, c.d2, c.d3
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_log(Tensor3 a) 
{
	tensor3_sync(a);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);
	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_log<<<num_blocks, threads_per_block>>>(
		c.data, a.data,
		c.d1, c.d2, c.d3
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_matmul(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	assert(a.d1 == b.d2);
	Tensor3 c = tensor3_new(b.d1, a.d2, max(a.d3, b.d3));
	if (b.d3 != a.d3) {
		if (a.d3 == 1) {
			a = tensor3_broadcast(a, a.d1, a.d2, b.d3);
		}
		if (b.d3 == 1) {
			b = tensor3_broadcast(b, b.d1, b.d2, a.d3);
		}
		else {
			ERROR("Shapes (%d, %d, %d) and (%d, %d, %d) are not broadcastable\n",
			      a.d1, a.d2, a.d3, b.d1, b.d2, b.d3);
		}
	}

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_matmul<<<num_blocks, threads_per_block>>>(
		c.data, a.data, b.data,
		c.d1, c.d2, c.d3,
		a.d1
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_mul_scalar(Tensor3 a, float b) 
{
	tensor3_sync(a);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);

	dim3 threads_per_block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, THREAD_PER_BLOCK_Z);
	dim3 num_blocks(
		int_ceil(c.d1, threads_per_block.x),
		int_ceil(c.d2, threads_per_block.y),
		int_ceil(c.d3, threads_per_block.z)
	);

	kernel_mul_scalar<<<num_blocks, threads_per_block>>>(
		c.data, a.data, b,
		c.d1, c.d2, c.d3
	);             

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_one_hot_cpu(int *arr, int B, int T, int A)
{
	float *res = (float *) malloc(B * T * A * sizeof(float));
	for (int k = 0; k < B; k++)
	for (int j = 0; j < T; j++)
	for (int i = 0; i < A; i++) {
		res[KEY(k, j, i, B, T, A)] = (arr[KEY(0, k, j, 0, B, T)] == i ? 1.0 : 0.0);
	}
	return tensor3_new(A, T, B, res);
}
