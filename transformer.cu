#include <stdio.h>
#include <assert.h>

#define ERROR(message, ...) do { fprintf(stderr, message, ##__VA_ARGS__); abort(); } while(0)

// x         @  w     + b = c
// (B, N, K) @ (K, M) + (N) = (B, N, M)
__global__ void kernel_matmul_bias_ll(
		float *out, 
		const float *x, 
		const float *w,
		const float *b,
		int N, int K, int M)
{
	int in = threadIdx.y + blockIdx.y * blockDim.y;
	int im = threadIdx.x + blockIdx.x * blockDim.x;
	if (in < N && im < M) {
		float sum = b[im];
		for (int ik = 0; ik < K; ik++) {
			sum += x[in*K + ik] * w[ik * M + im];
		}
		out[in * M + im] = sum;
	}
}

int int_ceil(int a, int b) 
{
	return (a +b -1) / b;
}

/*
   Simple allocator that receives a buffer and manages allocations within the buffer with
   a stack
*/

#define MAX_CNT_ALLOCATION 512
struct StaticStackAllocator 
{
	char* buff; // ptr to end of the memory
	size_t buff_cnt, buff_capacity; 
	size_t st[MAX_CNT_ALLOCATION];
	size_t st_cnt;

	StaticStackAllocator() {}
	StaticStackAllocator(char *buff_, size_t buff_capacity_) 
	{
		buff = buff_;
		buff_capacity = buff_capacity_;
		buff_cnt = 0;
		st_cnt = 0;
	}
	void *alloc(size_t sz)
	{
		assert(buff_cnt + sz <= buff_capacity && "ERROR: not enought memory");
		st[st_cnt++] = sz;
		char *ptr = buff + buff_cnt;
		buff_cnt += sz;
		return ptr;
	}
	void pop()
	{
		assert(st_cnt > 0 && "ERROR: Poping empty stack");
		size_t sz = st[--st_cnt];
		buff_cnt -= sz;
	}
	void *peek()
	{
		assert(st_cnt > 0 && "ERROR: Peeking empty stack");
		size_t sz = st[st_cnt - 1];
		return buff + buff_cnt - sz;
	}
	void clean()
	{
		buff_cnt = 0;
		st_cnt = 0;
	}
	float *alloc_float(size_t sz) 
	{
		return (float *) alloc(sz * sizeof(float));
	}
	float *peek_float()
	{
		return (float *) peek();
	}
};

struct ModelConfig 
{
	int dmodel;
	int dff;
	int max_sequence_len;
	int vocab_size;
	int n_block_layers;
	int n_heads;
};

struct Model 
{
	ModelConfig config;
	StaticStackAllocator activation_allocator;
};

/* Layer functions:
   This functions should have the return values in activation_allocator.peek()
 */
void matmul_forward(Model &model,
		const float *x, const float *w, const float*b, 
		int N, int K, int M)
{
	float *out = model.activation_allocator.alloc_float(N*M);
	dim3 block_dim(32, 32);
	dim3 grid_dim(int_ceil(M, 32), int_ceil(N, 32));
	kernel_matmul_bias_ll<<<grid_dim, block_dim>>>(out, x, w, b, N, K, M);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}

size_t matmul_temp_size(int B, int N, int K, int M)
{
	return (size_t) B * N * K * M;
}

void feed_forward_forward(Model &model, 
		const float *x, const float *w, const float*b, 
		int B, int sequence_len)
{
	matmul_forward(
		model, x, w, b,
		B*sequence_len, model.config.dmodel, model.config.dff
	);
	float *out1 = (float *) model.activation_allocator.peek();
}

size_t feed_forward_temp_size(int B, int sequence_len, ModelConfig config)
{
	return matmul_temp_size(B, sequence_len, config.dmodel, config.dff) +
		   matmul_temp_size(B, sequence_len, config.dff, config.dmodel);

}
