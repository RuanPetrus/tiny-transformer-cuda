#include <assert.h>
#include <random>

#define TODO(message) do { fprintf(stderr, "%s:%d: TODO: %s\n", __FILE__, __LINE__, message); abort(); } while(0)
#define ERROR(...) do { fprintf(stderr, __VA_ARGS__); abort(); } while(0)
#define KEY(i3, i2, i1, d3, d2, d1) (i3*d2*d1 + i2*d1 + i1)

#define THREAD_PER_BLOCK_X (1 << 4)
#define THREAD_PER_BLOCK_Y (1 << 3)
#define THREAD_PER_BLOCK_Z (1 << 3)

__global__ void kernel_vector_add(float *c, float * a, float *b, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		c[id] = a[id] + b[id];
	}
}

__global__ void kernel_vector_mul(float *c, float * a, float *b, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		c[id] = a[id] * b[id];
	}
}

__global__ void kernel_vector_div(float *c, float * a, float *b, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		c[id] = a[id] / b[id];
	}
}

__global__ void kernel_vector_exp(float *c, float *a, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		c[id] = exp(a[id]);
	}
}

__global__ void kernel_mat_mul(float *c, float *a, float *b, int N, int M, int T)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < M) {
		float sum = 0;
		for (int k = 0; k < T; k++) {
			 sum += a[row*T + k] * b[k * M + col];
		}
		c[row*M + col] = sum;
	}
}

// Sum over axis
__global__ void kernel_sum(float *c, float *a, 
							int cd1, int cd2, int cd3, // c dimensions
							int ad1, int ad2, int ad3) // a dimensions
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	int kn = ad3 / cd3;
	int jn = ad2 / cd2;
	int in = ad1 / cd1;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		float sum = 0;
		for (int k = 0; k < kn; k++)
		for (int j = 0; j < jn; j++)
		for (int i = 0; i < in; i++) {
			int ai3 = ci3 + k;
			int ai2 = ci2 + j;
			int ai1 = ci1 + i;
			sum += a[KEY(ai3, ai2, ai1, ad3, ad2, ad1)];
		}
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = sum;
	}
}

// Brodcast a to c
__global__ void kernel_broadcast(float *c, float *a, 
								 int cd1, int cd2, int cd3,
								 int ad1, int ad2, int ad3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	int f1 = cd1 / ad1;
	int f2 = cd2 / ad2;
	int f3 = cd3 / ad3;

	int ai3 = ci3 / f3;
	int ai2 = ci2 / f2;
	int ai1 = ci1 / f1;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		// c[id3][id2][id1] = a[ai3][ai2][ai1];
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = a[KEY(ai3, ai2, ai1, ad3, ad2, ad1)];
	}
}


static int TENSOR_GPU_LAST_SYNC = 0;

float random_normal_distribution_float()
{
	static std::random_device random_device{};
	static std::mt19937 random_generator{random_device()};
	static std::normal_distribution random_normal_distribution{0.0f, 1.0f};
	return random_normal_distribution(random_generator);
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
	printf("Tensor:\n");
	for (int k = 0; k < t.d3; k++) {
		printf("[ ");
		for (int j = 0; j < t.d2; j++) {
			printf("[ ");
			for (int i = 0; i < t.d1; i++) {
				printf("%.2f ", t.cpu_data[KEY(k, j, i, t.d3, t.d2, t.d1)]);
			}
			printf("] ");
		}
		printf("]\n");
	}
}

int int_ceil(int a, int b) 
{
	return (a + b-1) / b;
}

static const int BLOCK_SIZE = 32;

void vector_add(float *dest, float *a, float *b, int n)
{
	int n_block = int_ceil(n, BLOCK_SIZE);
	kernel_vector_add<<<n_block, BLOCK_SIZE>>>(dest, a, b, n);
}

void vector_mul(float *dest, float *a, float *b, int n)
{
	int n_block = int_ceil(n, BLOCK_SIZE);
	kernel_vector_mul<<<n_block, BLOCK_SIZE>>>(dest, a, b, n);
}

void vector_exp(float *dest, float *a, int n)
{
	int n_block = int_ceil(n, BLOCK_SIZE);
	kernel_vector_exp<<<n_block, BLOCK_SIZE>>>(dest, a, n);
}


Tensor3 tensor3_add(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	assert(a.d1 == b.d1 && a.d2 == b.d2 && a.d3 == b.d3);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);
	vector_add(c.data, a.data, b.data, a.d1 * a.d2 * a.d3);
	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_sum1(Tensor3 a) 
{
	tensor3_sync(a);
	Tensor3 c = tensor3_new(1, a.d2, a.d3);

	dim3 threads_per_block(1, BLOCK_SIZE/2, BLOCK_SIZE/2);
	dim3 num_blocks(1, int_ceil(c.d2, BLOCK_SIZE/2), int_ceil(c.d3, BLOCK_SIZE/2));

	kernel_sum<<<num_blocks, threads_per_block>>>(
		c.data, a.data, 
		c.d1, c.d2, c.d3, // c dimensions
		a.d1, a.d2, a.d3 // a dimensions
	);             

	tensor3_unsync(&c);
	return c;
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

// Broadcast A tensor to dimensions (d1, d2, d3)
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

	int n = c.d1 * c.d2 * c.d3;
	int n_block = int_ceil(n, BLOCK_SIZE);
	kernel_vector_div<<<n_block, BLOCK_SIZE>>>(c.data, a.data, b.data, n);
	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_mul(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	assert(a.d1 == b.d1 && a.d2 == b.d2 && a.d3 == b.d3);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);
	vector_mul(c.data, a.data, b.data, a.d1 * a.d2 * a.d3);
	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_exp(Tensor3 a) 
{
	tensor3_sync(a);
	Tensor3 c = tensor3_new(a.d1, a.d2, a.d3);
	vector_exp(c.data, a.data, a.d1 * a.d2 * a.d3);
	tensor3_unsync(&c);
	return c;
}

Tensor3 tensor3_cpu_naive_matmul(Tensor3 a, Tensor3 b)
{
	tensor3_sync(a); tensor3_sync(b);
	assert(a.d1 == b.d2);
	int N = a.d2;
	int M = b.d1;
	int T = a.d1;
	Tensor3 c = tensor3_new(M, N, max(a.d3, b.d3));

	float *A = a.cpu_data;
	float *B = b.cpu_data;
	float *C = c.cpu_data;

	if (a.d3 == b.d3){
		if (a.d3 == 1) {
			for(size_t i = 0; i < N; i++) {
				for(size_t k = 0; k < T; k++) {
					for(size_t j = 0; j < M; j++) {
						C[i*M + j] += A[i*T + k] * B[k*M + j];
					}
				}
			}
		}
		else {
			TODO("Not implemented");
		}
	}
	else {
		TODO("Not implemented");
	}
	return c;
}

// TODO: implement more paralel version
Tensor3 tensor3_matmul(Tensor3 a, Tensor3 b) 
{
	tensor3_sync(a); tensor3_sync(b);
	assert(a.d1 == b.d2);
	assert(a.d3 == b.d3 || a.d3 == 1 || b.d3 == 1);
	int N = a.d2;
	int M = b.d1;
	int T = a.d1;
	Tensor3 c = tensor3_new(M, N, max(a.d3, b.d3));
	dim3 threads_per_block(BLOCK_SIZE/2, BLOCK_SIZE/2);
	dim3 num_blocks(int_ceil(M, BLOCK_SIZE/2), int_ceil(N, BLOCK_SIZE/2));
	for (int i = 0; i < c.d3; i++) {
		kernel_mat_mul<<<num_blocks, threads_per_block>>>(
			c.data + i*(c.d1 * c.d2), 
			a.data + i*(a.d1 * a.d2) * (a.d3 > 1), 
			b.data + i*(b.d1 * b.d2) * (b.d3 > 1), 
			N, M, T
		);
	}
	tensor3_unsync(&c);
	return c;
}
