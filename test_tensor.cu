#include <stdio.h>
#include <random>
#include <float.h>

#include "tensor.cu"

#define test_assert(expr) do { if (!(expr)) { fprintf(stderr, "TEST_ASSERT_FAIL: %s:%d\n", __FILE__, __LINE__); return false; } } while(0)
#define ARRAY_SIZE(A) (sizeof(A) / sizeof(A[0]))

float random_norm_float() {
	return (float) rand() / RAND_MAX;
}

float random_float() {
	return random_norm_float() * FLT_MAX;
}


int test_tensor3_add() 
{
	const int MAX_N = 512;
	int N = (rand() % MAX_N) + 1;
	Tensor3 a = tensor3_new(N, 1, 1);
	Tensor3 b = tensor3_new(N, 1, 1);
	for (int i = 0; i < N; i++) {
		a.cpu_data[i] = random_float();
		b.cpu_data[i] = random_float();
	}
	tensor3_copy_cpu_to_gpu(a); 
	tensor3_copy_cpu_to_gpu(b);

	Tensor3 c = tensor3_add(a, b);
	
	tensor3_copy_gpu_to_cpu(a); 
	tensor3_copy_gpu_to_cpu(b);
	tensor3_copy_gpu_to_cpu(c);

	for (int i = 0; i < N; i++) {
		test_assert(c.cpu_data[i] == a.cpu_data[i] + b.cpu_data[i]);
	}

	tensor3_free(a); tensor3_free(b); tensor3_free(c);

	return true;
}

int test_tensor3_broadcast() 
{
	{
		float data[] = {1, 2, 3};
		Tensor3 a = tensor3_new(3, 1, 1, data);
		Tensor3 c = tensor3_broadcast(a, 3, 3, 1);

		float expected[] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
		tensor3_copy_gpu_to_cpu(c);

		test_assert(c.d1 == 3 && c.d2 == 3 && c.d3 == 1);

		for (int i = 0; i < ARRAY_SIZE(expected); i++) {
			test_assert(c.cpu_data[i] == expected[i]);
		}
	}
	{
		float data[] = {1, 2, 3};
		Tensor3 a = tensor3_new(3, 1, 1, data);
		Tensor3 c = tensor3_broadcast(a, 3, 3, 2);

		float expected[] = {1, 2, 3, 1, 2, 3, 1, 2, 3,
							1, 2, 3, 1, 2, 3, 1, 2, 3};
		tensor3_copy_gpu_to_cpu(c);

		test_assert(c.d1 == 3 && c.d2 == 3 && c.d3 == 2);

		for (int i = 0; i < ARRAY_SIZE(expected); i++) {
			test_assert(c.cpu_data[i] == expected[i]);
		}
	}
	{
		float data[] = {1, 2, 3};
		Tensor3 a = tensor3_new(3, 1, 1, data);
		Tensor3 c = tensor3_broadcast(a, 3, 1, 2);

		float expected[] = {1, 2, 3, 1, 2, 3};
		tensor3_copy_gpu_to_cpu(c);

		test_assert(c.d1 == 3 && c.d2 == 1 && c.d3 == 2);

		for (int i = 0; i < ARRAY_SIZE(expected); i++) {
			test_assert(c.cpu_data[i] == expected[i]);
		}
	}
	return true;
}

int test_tensor3_sum() 
{
	{
		float data[] = {1, 2, 3, 1, 2, 3};
		Tensor3 a = tensor3_new(3, 2, 1, data);
		Tensor3 c = tensor3_sum(a, 3, 1, 1);

		float expected[] = {2, 4, 6};
		tensor3_copy_gpu_to_cpu(c);

		test_assert(c.d1 == 3 && c.d2 == 1 && c.d3 == 1);

		for (int i = 0; i < ARRAY_SIZE(expected); i++) {
			test_assert(c.cpu_data[i] == expected[i]);
		}
	}
	{
		float data[] = {1, 2, 3, 1, 2, 3};
		Tensor3 a = tensor3_new(3, 2, 1, data);
		Tensor3 c = tensor3_sum(a, 1, 2, 1);

		float expected[] = {6, 6};
		tensor3_copy_gpu_to_cpu(c);

		test_assert(c.d1 == 1 && c.d2 == 2 && c.d3 == 1);

		for (int i = 0; i < ARRAY_SIZE(expected); i++) {
			test_assert(c.cpu_data[i] == expected[i]);
		}
	}
	{
		float data[] = {1, 2, 3, 1, 2, 3};
		Tensor3 a = tensor3_new(3, 2, 1, data);
		Tensor3 c = tensor3_sum(a, 1, 1, 1);

		float expected[] = {12};
		tensor3_copy_gpu_to_cpu(c);

		test_assert(c.d1 == 1 && c.d2 == 1 && c.d3 == 1);

		for (int i = 0; i < ARRAY_SIZE(expected); i++) {
			test_assert(c.cpu_data[i] == expected[i]);
		}
	}
	return true;
}

int test_tensor3_mul() 
{
	const int MAX_N = 512;
	int N = (rand() % MAX_N) + 1;
	Tensor3 a = tensor3_new(N, 1, 1);
	Tensor3 b = tensor3_new(N, 1, 1);
	for (int i = 0; i < N; i++) {
		a.cpu_data[i] = random_float();
		b.cpu_data[i] = random_float();
	}
	tensor3_copy_cpu_to_gpu(a); 
	tensor3_copy_cpu_to_gpu(b);

	Tensor3 c = tensor3_mul(a, b);
	
	tensor3_copy_gpu_to_cpu(a); 
	tensor3_copy_gpu_to_cpu(b);
	tensor3_copy_gpu_to_cpu(c);

	for (int i = 0; i < N; i++) {
		test_assert(c.cpu_data[i] == a.cpu_data[i] * b.cpu_data[i]);
	}

	tensor3_free(a); tensor3_free(b); tensor3_free(c);

	return true;
}

int test_tensor3_matmul_1_1() 
{
	const int MAX_N = 512;
	int N = (rand() % MAX_N) + 1;
	int M = (rand() % MAX_N) + 1;
	int T = (rand() % MAX_N) + 1;
	Tensor3 a = tensor3_new(T, N, 1);
	Tensor3 b = tensor3_new(M, T, 1);
	for (int i = 0; i < N*T; i++) {
		a.cpu_data[i] = random_float();
	}
	for (int i = 0; i < T*M; i++) {
		b.cpu_data[i] = random_float();
	}

	tensor3_copy_cpu_to_gpu(a); 
	tensor3_copy_cpu_to_gpu(b);

	Tensor3 c = tensor3_matmul(a, b);
	
	tensor3_copy_gpu_to_cpu(a); 
	tensor3_copy_gpu_to_cpu(b);
	tensor3_copy_gpu_to_cpu(c);

	Tensor3 c_cpu = tensor3_cpu_naive_matmul(a, b);

	for (int i = 0; i < N*M; i++) {
		test_assert(c.cpu_data[i] == c_cpu.cpu_data[i]);
	}

	tensor3_free(a); tensor3_free(b); tensor3_free(c); tensor3_free(c_cpu);

	return true;
}

int main() {
	int seed = time(0);
	srand(seed);
	int cnt_error = 0;
	cnt_error += !test_tensor3_add();
	cnt_error += !test_tensor3_mul();
	cnt_error += !test_tensor3_matmul_1_1();
	cnt_error += !test_tensor3_broadcast();
	cnt_error += !test_tensor3_sum();

	if (cnt_error) {
		fprintf(stderr, "%d tests failed, seed: %d\n", cnt_error, seed);
		return 1;
	}

	return 0;
}
