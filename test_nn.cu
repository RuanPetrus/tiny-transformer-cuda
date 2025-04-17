#include <stdio.h>

#include "nn.cu"

#define test_assert(expr) do { if (!(expr)) { fprintf(stderr, "TEST_ASSERT_FAIL: %s:%d\n", __FILE__, __LINE__); return false; } } while(0)

int test_linear_shape() 
{
	Config config = {4, 3, 2, 2};

	Linear embw(config.n_alphabet, config.n_embd);
	Tensor3 x = tensor3_new(config.n_alphabet, config.n_context, config.n_batch);

	Tensor3 out = embw.forward(x);

	test_assert(out.d1 == config.n_embd &&
		        out.d2 == config.n_context &&
                out.d3 == config.n_batch);
		  
	return true;
}

int main() {
	int seed = time(0);
	srand(seed);
	int cnt_error = 0;
	cnt_error += !test_linear_shape();

	if (cnt_error) {
		fprintf(stderr, "%d tests failed, seed: %d\n", cnt_error, seed);
		return 1;
	}

	return 0;
}
