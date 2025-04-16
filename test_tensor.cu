#include <stdio.h>
#include <random>
#include <float.h>

#include "tensor.cu"

#define test_assert(expr) do { if (!(expr)) { fprintf(stderr, "TEST_ASSERT_FAIL: %s:%d\n", __FILE__, __LINE__); return false; } } while(0)

float random_norm_float() {
	return (float) rand() / RAND_MAX;
}

float random_float() {
	return random_norm_float() * FLT_MAX;
}


int test_vector_add() 
{
	const int N = 256;
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

int main() {
	int seed = time(0);
	srand(seed);
	int cnt_error = 0;
	cnt_error += !test_vector_add();

	if (cnt_error) {
		fprintf(stderr, "%d tests failed, seed: %d\n", cnt_error, seed);
		return 1;
	}

	return 0;
}
