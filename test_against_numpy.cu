#include <stdio.h>
#include "transformer.cu"

#define TEMP_PATH "/tmp/tiny_transformer_test_"
#define TEST_ASSERT(expr, message, ...) do  { \
if (!(expr)) { \
	fprintf(stderr, message, ##__VA_ARGS__); \
	return false; \
} \
} while(0)

#define LOAD_VAR(x)   TEST_ASSERT(sizeof(x) == fread(&x, 1, sizeof(x), f), "Test bin format is wrong\n");
#define LOAD_ARRAY(x) TEST_ASSERT(sizeof(x) == fread(x, 1, sizeof(x), f), "Test bin format is wrong\n");

#define TEST_COPY_ARRAY(x, x_exp) cudaMemcpy(x, x_exp, sizeof(x_exp), cudaMemcpyHostToDevice)

static Model test_model;
StaticStackAllocator temp_cpu_allocator;

#define CLOSE_EPS 1e-4
bool assert_close(float *gpu_data, float *exp, int n)
{
	float *data = temp_cpu_allocator.alloc_float(n);
	cudaMemcpy(data, gpu_data, sizeof(float) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		float diff = abs(data[i] - exp[i]);
		TEST_ASSERT(diff < CLOSE_EPS, "Number are not close (i, diff) = (%d, %f)", i, diff);
	}
	printf("\n");
	return true;
}

bool show_gpu_data(float *gpu_data, int n) 
{
	float *data = temp_cpu_allocator.alloc_float(n);
	cudaMemcpy(data, gpu_data, sizeof(float) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		printf("%.4f ", data[i]);
	}
	printf("\n");
	return true;
}

bool show_data(float *data, int n) 
{
	for (int i = 0; i < n; i++) {
		printf("%.4f ", data[i]);
	}
	printf("\n");
	return true;
}

bool test_matmul()
{
	FILE *f = fopen(TEMP_PATH"matmul.bin", "rb");
	TEST_ASSERT(f != NULL, "Could not open test_matmul bin\n");

	int N, K, M;
	LOAD_VAR(N); LOAD_VAR(K); LOAD_VAR(M);

	float x_exp[N * K]; LOAD_ARRAY(x_exp);
	float w_exp[K * M]; LOAD_ARRAY(w_exp);
	float b_exp[M];     LOAD_ARRAY(b_exp);
	float out_exp[N*M]; LOAD_ARRAY(out_exp);
	fclose(f);
	test_model.activation_allocator.clean();

	float *x = (float *) test_model.activation_allocator.alloc(sizeof(x_exp)); 
	float *w = (float *) test_model.activation_allocator.alloc(sizeof(w_exp)); 
	float *b = (float *) test_model.activation_allocator.alloc(sizeof(b_exp)); 
	TEST_COPY_ARRAY(x, x_exp);
	TEST_COPY_ARRAY(w, w_exp);
	TEST_COPY_ARRAY(b, b_exp);
	cudaDeviceSynchronize();

	// show_gpu_data(x, N*K);
	// show_gpu_data(w, K*M);
	// show_gpu_data(b, M);
	// return true;

	matmul_forward(test_model, x, w, b, N, K, M);
	cudaDeviceSynchronize();
	float *out = test_model.activation_allocator.peek_float();
	return assert_close(out, out_exp, N*M);
}

#define TEMP_GPU_BUFFER_CAPACITY 1 << 20
#define TEMP_CPU_BUFFER_CAPACITY 1 << 20
static char *temp_gpu_buffer;
static char temp_cpu_buffer[TEMP_CPU_BUFFER_CAPACITY];

int main()
{
	cudaMalloc(&temp_gpu_buffer, TEMP_GPU_BUFFER_CAPACITY);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}

	temp_cpu_allocator = StaticStackAllocator(temp_cpu_buffer, TEMP_CPU_BUFFER_CAPACITY);
	test_model.activation_allocator = StaticStackAllocator(temp_gpu_buffer, TEMP_GPU_BUFFER_CAPACITY);

	int errors = 0;
	errors += !test_matmul();

	if (errors > 0) {
		fprintf(stderr, "Tests failed with %d errors\n", errors);
		return 1;
	}

	fprintf(stdout, "SUCESS\n");

	return 0;
}
