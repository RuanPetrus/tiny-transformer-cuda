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
	return random_norm_float();
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

	memset(C, 0, sizeof(float) * c.d1 * c.d2 * c.d3);

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
	tensor3_copy_cpu_to_gpu(c);
	return c;
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

int test_tensor3_softmax()
{
	{
		float x_data[] = {
		-0.9839,  0.2896, -0.6639,  0.3162,  1.1602, -0.2941, -1.8926,  1.0604,
        -0.6406,  1.3370, -0.7871,  1.1252, -1.7588, -0.2872, -0.7927, -0.8530,
        -0.7520,  0.7507, -1.1152,  0.4808,  0.3390, -0.4116,  0.7865, -0.0073,
         0.6995, -0.3520,  0.5569,  0.4294,  0.7227,  0.3441, -0.6944,  0.9668,
        -2.0168,  0.0365, -2.0879,  1.4886,  3.6720,  1.1824, -1.8315, -0.5604,
         0.6067, -0.3952,  0.4251, -1.0766,  1.2391,  0.3002, -1.5661, -1.4361,
        -0.2195, -0.6596, -0.1858,  0.8256,  0.6159,  0.2622, -1.7042,  0.0517,
         0.9444, -1.3036,  0.8708, -0.8818
		};
		float expected_e[] = {
		0.3739,  1.3359,  0.5149,  1.3720,  3.1906,  0.7452,  0.1507,  2.8876,
        0.5270,  3.8076,  0.4552,  3.0808,  0.1722,  0.7504,  0.4526,  0.4261,
        0.4714,  2.1185,  0.3278,  1.6173,  1.4035,  0.6626,  2.1958,  0.9927,
        2.0128,  0.7033,  1.7453,  1.5363,  2.0599,  1.4108,  0.4994,  2.6295,
        0.1331,  1.0371,  0.1240,  4.4307, 39.3291,  3.2622,  0.1602,  0.5710,
        1.8344,  0.6735,  1.5298,  0.3408,  3.4526,  1.3502,  0.2089,  0.2379,
        0.8030,  0.5171,  0.8304,  2.2832,  1.8513,  1.2998,  0.1819,  1.0530,
        2.5713,  0.2715,  2.3887,  0.4140
		};
		float expected_se[] = {
 		6.7872,  8.1181,  4.9112,  4.9613,  7.2674,  7.4556,  4.4230, 47.7532,
        7.8311,  3.1169,  6.4466,  6.6986
		};
		float expected_prob[] = {
		0.0551, 0.1968, 0.0759, 0.2021, 0.4701, 0.0918, 0.0186, 0.3557, 0.0649,
        0.4690, 0.0927, 0.6273, 0.0351, 0.1528, 0.0922, 0.0859, 0.0950, 0.4270,
        0.0661, 0.3260, 0.1931, 0.0912, 0.3021, 0.1366, 0.2770, 0.0943, 0.2341,
        0.2061, 0.2763, 0.1892, 0.1129, 0.5945, 0.0301, 0.2345, 0.0280, 0.0928,
        0.8236, 0.0683, 0.0034, 0.0120, 0.2342, 0.0860, 0.1953, 0.0435, 0.4409,
        0.4332, 0.0670, 0.0763, 0.2576, 0.1659, 0.1288, 0.3542, 0.2872, 0.2016,
        0.0282, 0.1572, 0.3839, 0.0405, 0.3566, 0.0618
		};

		Tensor3 x =    tensor3_new(5, 4, 3, x_data);
		Tensor3 e =    tensor3_exp(x);
		Tensor3 se =   tensor3_sum(e, 1, e.d2, e.d3);
		Tensor3 prob = tensor3_div(e, se);

		test_assert(tensor3_same_shape(x, e));
		test_assert(se.d1 == 1 && se.d2 == e.d2 && se.d3 == e.d3);
		test_assert(tensor3_same_shape(x, prob));

		tensor3_copy_gpu_to_cpu(e);
		tensor3_copy_gpu_to_cpu(se);
		tensor3_copy_gpu_to_cpu(prob);

		{
			float big_diff= 0;
			for (int i = 0; i < e.d1 * e.d2 * e.d3; i++) {
				float diff = abs(expected_e[i] - e.cpu_data[i]);
				big_diff = max(big_diff, diff);
			} 
			if (big_diff >= 1e-2) {
				printf("Diff = %.10f\n", big_diff);
			}
			test_assert(big_diff < 1e-2);
		}
		{
			float big_diff= 0;
			for (int i = 0; i < se.d1 * se.d2 * se.d3; i++) {
				float diff = abs(expected_se[i] - se.cpu_data[i]);
				big_diff = max(big_diff, diff);
			} 
			if (big_diff >= 1e-2) {
				printf("Diff = %.10f\n", big_diff);
			}
			test_assert(big_diff < 1e-2);
		}
		{
			float big_diff= 0;
			for (int i = 0; i < prob.d1 * prob.d2 * prob.d3; i++) {
				float diff = abs(expected_prob[i] - prob.cpu_data[i]);
				big_diff = max(big_diff, diff);
			} 
			if (big_diff >= 1e-2) {
				printf("Diff = %.10f\n", big_diff);
			}
			test_assert(big_diff < 1e-2);
		}


	}
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
		tensor3_free(a); tensor3_free(c);
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
		tensor3_free(a); tensor3_free(c);
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
		tensor3_free(a); tensor3_free(c);
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

		tensor3_free(a); tensor3_free(c);
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
		tensor3_free(a); tensor3_free(c);
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
		tensor3_free(a); tensor3_free(c);
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

	float big_diff = 0;
	for (int i = 0; i < N*M; i++) {
		float diff = abs(c.cpu_data[i] -c_cpu.cpu_data[i]);
		big_diff = max(big_diff, diff);
	}
	if (big_diff >= 1e-4) {
		printf("N %d, M %d, T %d, diff=%.10f\n", N, M, T, big_diff);
	}
	test_assert(big_diff < 1e-4);

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
	cnt_error += !test_tensor3_softmax();

	if (cnt_error) {
		fprintf(stderr, "%d tests failed, seed: %d\n", cnt_error, seed);
		return 1;
	}

	return 0;
}
