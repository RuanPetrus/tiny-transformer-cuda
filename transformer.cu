#include <stdio.h>
#include <assert.h>
#include <float.h>

#define ERROR(message, ...) do { fprintf(stderr, message, ##__VA_ARGS__); abort(); } while(0)

// x         @  w     + b = c
// (B, N, K) @ (K, M) + (N) = (B, N, M)
__global__ void kernel_matmul_bias(
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

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void kernel_gelu_forward(float* out, const float* x, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = x[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

__global__ void kernel_softmax_forward(float* out, const float* x, int N, int M) 
{
    int in = blockIdx.y * blockDim.y + threadIdx.y;
    int im = blockIdx.x * blockDim.x + threadIdx.x;
    if (in < N && im < M) {
		float mx =  FLT_MIN;
		for (int i = 0; i < M; i++) {
			mx = max(mx, x[in*M + i]);
		}
		float sum = 0.0;
		for (int i = 0; i < M; i++) {
			sum += exp(x[in*M + i] - mx);
		}
		out[in * M + im] = exp(x[in * M + im] - mx) / sum;
    }
}

__global__ void kernel_crossentropy_forward(float* out, const float *x, const int *y, int N, int M) 
{
    int in = blockIdx.y * blockDim.y + threadIdx.y;
    int im = blockIdx.x * blockDim.x + threadIdx.x;
    if (in < N && im < M) {
		int ix = y[in];
		out[in * M + im] = -log(x[in * M + im]) * (ix == im);
    }
}

#define LAYER_NORM_EPS 1e-5
__global__ void kernel_layernorm_forward(float* out, 
										 const float *x, const float *w, const float *b,
		                                int N, int M) 
{
    int in = blockIdx.y * blockDim.y + threadIdx.y;
    int im = blockIdx.x * blockDim.x + threadIdx.x;
    if (in < N && im < M) {
		float m = 0;
		for (int i = 0; i < M; i++) {
			m += x[in * M + i];
		}
		m /= M;
		float var = 0;
		for (int i = 0; i < M; i++) {
			float diff = x[in * M + i] - m;
			var += diff * diff;
		}
		var /= M;
		float s = rsqrtf(var + LAYER_NORM_EPS);

		float nr = (x[in * M + im] - m) * s;
		out[in * M + im] = nr * w[im] + b[im];
    }
}

__global__ void kernel_encoder_forward(float* out, 
									   const int *x, const float *w, const float *wp,
		                               int B, int T, int C) 
{
	// out = (B, T, C) x = (B, T), w = (A, C), wp = (T, C)
    int ib = blockIdx.z * blockDim.z + threadIdx.z;
    int it = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib < B && it < T && ic < C) {
		int ix = x[ib * T + it];
		out[ib*T*C + it*C + ic] = w[ix * C + ic] + wp[it * C + ic];
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

struct Parameters
{
	float *ffw0; // (n_block_layers, dmodel, dff)
	float *ffb0; // (n_block_layers, dff)
	float *ffw1; // (n_block_layers, dmff, dmodel)
	float *ffb1; // (n_block_layers, dmodel)
	float *ln1w; // (n_block_layers, dmodel)
	float *ln1b; // (n_block_layers, dmodel)
};

struct Model 
{
	ModelConfig config;
	StaticStackAllocator activation_allocator;
	Parameters params;
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
	kernel_matmul_bias<<<grid_dim, block_dim>>>(out, x, w, b, N, K, M);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

}

void gelu_forward(Model &model,
		const float *x, int N)
{
	float *out = model.activation_allocator.alloc_float(N);
	dim3 block_dim(32);
	dim3 grid_dim(int_ceil(N, 32));
	kernel_gelu_forward<<<grid_dim, block_dim>>>(out, x, N);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}

void softmax_forward(Model &model,
		const float *x, int N, int M)
{
	float *out = model.activation_allocator.alloc_float(N * M);
	dim3 block_dim(32, 32);
	dim3 grid_dim(int_ceil(M, 32), int_ceil(N, 32));
	kernel_softmax_forward<<<grid_dim, block_dim>>>(out, x, N, M);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}

void feed_forward_forward(Model &model, 
		const float *x, int block_id, 
		int B, int sequence_len)
{
	int T = sequence_len;
	int C = model.config.dmodel;
	int Cff = model.config.dff;
	// Loading parameters
	float *w0 = model.params.ffw0 + block_id * C * Cff;
	float *b0 = model.params.ffb0 + block_id * Cff;
	float *w1 = model.params.ffw1 + block_id * Cff * C;
	float *b1 = model.params.ffb1 + block_id * C;

	matmul_forward(
		model, x, w0, b0,
		B*T, C, Cff
	);
	float *out0 = (float *) model.activation_allocator.peek(); // (B, T, Cff)
	cudaDeviceSynchronize();

	gelu_forward(model, out0, B*T*Cff);
	float *out1 = (float *) model.activation_allocator.peek(); // (B, T, Cff)
	cudaDeviceSynchronize();

	matmul_forward(
		model, out1, w1, b1,
		B*T, Cff, C
	);
}

void crossentropy_forward(Model &model, 
		const float *x, const int *y, 
		int N, int M)
{
	
	softmax_forward(
		model, x,
		N, M
	);
	float *probs = (float *) model.activation_allocator.peek(); // (B, T, Cff)
	cudaDeviceSynchronize();

	float *out = model.activation_allocator.alloc_float(N * M);
	dim3 block_dim(32, 32);
	dim3 grid_dim(int_ceil(M, 32), int_ceil(N, 32));
	kernel_crossentropy_forward<<<grid_dim, block_dim>>>(out, probs, y, N, M);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}

void layernorm_forward(Model &model, 
		const float *x, const float *w, const float *b,
		int N, int M)
{
	
	float *out = model.activation_allocator.alloc_float(N * M);
	dim3 block_dim(32, 32);
	dim3 grid_dim(int_ceil(M, 32), int_ceil(N, 32));
	kernel_layernorm_forward<<<grid_dim, block_dim>>>(out, x, w, b, N, M);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}

void encoder_forward(Model &model, 
		const int *x, const float *w, const float *wp,
		int B, int T)
{
	
	int C = model.config.dmodel;
	float *out = model.activation_allocator.alloc_float(B * T * C);
	dim3 block_dim(1 << 3, 1 << 3, 1 << 4);
	dim3 grid_dim(int_ceil(C, block_dim.x), int_ceil(T, block_dim.y), int_ceil(B, block_dim.z));
	kernel_encoder_forward<<<grid_dim, block_dim>>>(out, x, w, wp, B, T, C);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}
