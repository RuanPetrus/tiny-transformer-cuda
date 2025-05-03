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

	matmul_forward(test_model, x, w, b, N, K, M);
	cudaDeviceSynchronize();
	float *out = test_model.activation_allocator.peek_float();
	return assert_close(out, out_exp, N*M);
}

bool test_gelu_forward()
{
	FILE *f = fopen(TEMP_PATH"gelu_forward.bin", "rb");
	TEST_ASSERT(f != NULL, "Could not open test_gelu forward bin\n");

	int N;
	LOAD_VAR(N);

	float x_exp[N];   LOAD_ARRAY(x_exp);
	float out_exp[N]; LOAD_ARRAY(out_exp);
	fclose(f);
	test_model.activation_allocator.clean();

	float *x = (float *) test_model.activation_allocator.alloc(sizeof(x_exp)); 
	TEST_COPY_ARRAY(x, x_exp);
	cudaDeviceSynchronize();

	gelu_forward(test_model, x, N);
	cudaDeviceSynchronize();
	float *out = test_model.activation_allocator.peek_float();
	return assert_close(out, out_exp, N);
}

bool test_ff_forward()
{
	const char *bin_path = TEMP_PATH"ff_forward.bin";
	FILE *f = fopen(bin_path, "rb");
	TEST_ASSERT(f != NULL, "Could not open bins file %s\n", bin_path);

	int dmodel, dff, B, sequence_len;
	LOAD_VAR(dmodel); LOAD_VAR(dff);
	LOAD_VAR(B); LOAD_VAR(sequence_len);

	float x_exp[B * sequence_len * dmodel];   LOAD_ARRAY(x_exp);
	float w0_exp[dmodel*dff]; LOAD_ARRAY(w0_exp);
	float b0_exp[dff];        LOAD_ARRAY(b0_exp);
	float w1_exp[dmodel*dff]; LOAD_ARRAY(w1_exp);
	float b1_exp[dmodel];     LOAD_ARRAY(b1_exp);
	float out_exp[B * sequence_len * dmodel]; LOAD_ARRAY(out_exp);
	fclose(f);

	test_model.activation_allocator.clean();
	
	test_model.config.dmodel = dmodel;
	test_model.config.dff = dff;

	float *x = (float *) test_model.activation_allocator.alloc(sizeof(x_exp)); 
	TEST_COPY_ARRAY(x, x_exp);


	test_model.params.ffw0 = (float*) test_model.activation_allocator.alloc(sizeof(w0_exp));
	test_model.params.ffb0 = (float*) test_model.activation_allocator.alloc(sizeof(b0_exp));
	test_model.params.ffw1 = (float*) test_model.activation_allocator.alloc(sizeof(w1_exp));
	test_model.params.ffb1 = (float*) test_model.activation_allocator.alloc(sizeof(b1_exp));

	TEST_COPY_ARRAY(test_model.params.ffw0, w0_exp);
	TEST_COPY_ARRAY(test_model.params.ffb0, b0_exp);
	TEST_COPY_ARRAY(test_model.params.ffw1, w1_exp);
	TEST_COPY_ARRAY(test_model.params.ffb1, b1_exp);

	cudaDeviceSynchronize();

	feed_forward_forward(test_model, 
		x, 0, 
		B, sequence_len);
	cudaDeviceSynchronize();

	float *out = test_model.activation_allocator.peek_float();
	return assert_close(out, out_exp, sizeof(out_exp) / sizeof(float));
}

bool test_softmax_forward()
{
	const char *bin_path = TEMP_PATH"softmax_forward.bin";
	FILE *f = fopen(bin_path, "rb");
	TEST_ASSERT(f != NULL, "Could not open bins file %s\n", bin_path);

	int N, M;
	LOAD_VAR(N); LOAD_VAR(M);

	float x_exp[N*M];   LOAD_ARRAY(x_exp);
	float out_exp[N*M]; LOAD_ARRAY(out_exp);
	fclose(f);
	test_model.activation_allocator.clean();

	float *x = (float *) test_model.activation_allocator.alloc(sizeof(x_exp)); 
	TEST_COPY_ARRAY(x, x_exp);
	cudaDeviceSynchronize();

	softmax_forward(test_model, x, N, M);
	cudaDeviceSynchronize();
	float *out = test_model.activation_allocator.peek_float();
	return assert_close(out, out_exp, sizeof(out_exp) / sizeof(float));
}

float cpu_sum(float *gpu_data, int n)
{
	float *data = temp_cpu_allocator.alloc_float(n);
	cudaMemcpy(data, gpu_data, sizeof(float) * n, cudaMemcpyDeviceToHost);
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += data[i];
	}
	return sum;
}

bool test_crossentropy_forward()
{
	const char *bin_path = TEMP_PATH"crossentropy_forward.bin";
	FILE *f = fopen(bin_path, "rb");
	TEST_ASSERT(f != NULL, "Could not open bins file %s\n", bin_path);

	int N, M;
	LOAD_VAR(N); LOAD_VAR(M);

	float x_exp[N*M];   LOAD_ARRAY(x_exp);
	int y_exp[N];       LOAD_ARRAY(y_exp);
	float out_exp[1];   LOAD_ARRAY(out_exp);
	fclose(f);
	test_model.activation_allocator.clean();

	float *x = (float *) test_model.activation_allocator.alloc(sizeof(x_exp)); 
	int   *y = (int *) test_model.activation_allocator.alloc(sizeof(y_exp)); 
	TEST_COPY_ARRAY(x, x_exp);
	TEST_COPY_ARRAY(y, y_exp);
	cudaDeviceSynchronize();

	crossentropy_forward(test_model, x, y, N, M);
	float *out = test_model.activation_allocator.peek_float();
	cudaDeviceSynchronize();

	float mean = cpu_sum(out, N*M) / N;
	float diff = abs(mean - out_exp[0]);

	TEST_ASSERT(diff < CLOSE_EPS, "Values are not close (%.4f, %.4f)\n", mean, out_exp[0]);
	return true;
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
	errors += !test_gelu_forward();
	errors += !test_ff_forward();
	errors += !test_softmax_forward();
	errors += !test_crossentropy_forward();

	if (errors > 0) {
		fprintf(stderr, "Tests failed with %d errors\n", errors);
		return 1;
	}

	fprintf(stdout, "SUCESS\n");

	return 0;
}
