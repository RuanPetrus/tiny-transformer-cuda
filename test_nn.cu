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

int test_softmax_forward() 
{
	{
		float data[] = {
			-0.2232, -1.0094,  0.5380,  0.6884,  1.0586,  0.1460,  0.5853,  1.0516,
			 0.8871, -1.4617, -0.8628, -0.0584, -1.5897,  0.7192, -0.4429,  0.8478,
			 0.5169,  0.1614,  1.2202,  2.1393};
		float expected[] = {
			0.1032, 0.0470, 0.2210, 0.2569, 0.3719, 0.1365, 0.2119, 0.3377, 0.2865,
			0.0274, 0.0990, 0.2212, 0.0478, 0.4814, 0.1506, 0.1368, 0.0982, 0.0689,
			0.1985, 0.4976
		};
		Tensor3 x = tensor3_new(5, 4, 1, data);
		Tensor3 out = Softmax::forward(x);

		test_assert(tensor3_same_shape(x, out));

		tensor3_copy_gpu_to_cpu(out);
		float big_diff= 0;
		for (int i = 0; i < x.d1 * x.d2 * x.d3; i++) {
			float diff = abs(expected[i] - out.cpu_data[i]);
			big_diff = max(big_diff, diff);
		}
		test_assert(big_diff < 1e-4);
	}
	{
		float data[] = {
		-0.9839,  0.2896, -0.6639,  0.3162,  1.1602, -0.2941, -1.8926,  1.0604,
        -0.6406,  1.3370, -0.7871,  1.1252, -1.7588, -0.2872, -0.7927, -0.8530,
        -0.7520,  0.7507, -1.1152,  0.4808,  0.3390, -0.4116,  0.7865, -0.0073,
         0.6995, -0.3520,  0.5569,  0.4294,  0.7227,  0.3441, -0.6944,  0.9668,
        -2.0168,  0.0365, -2.0879,  1.4886,  3.6720,  1.1824, -1.8315, -0.5604,
         0.6067, -0.3952,  0.4251, -1.0766,  1.2391,  0.3002, -1.5661, -1.4361,
        -0.2195, -0.6596, -0.1858,  0.8256,  0.6159,  0.2622, -1.7042,  0.0517,
         0.9444, -1.3036,  0.8708, -0.8818
		};
		float expected[] = {
		0.0551, 0.1968, 0.0759, 0.2021, 0.4701, 0.0918, 0.0186, 0.3557, 0.0649,
        0.4690, 0.0927, 0.6273, 0.0351, 0.1528, 0.0922, 0.0859, 0.0950, 0.4270,
        0.0661, 0.3260, 0.1931, 0.0912, 0.3021, 0.1366, 0.2770, 0.0943, 0.2341,
        0.2061, 0.2763, 0.1892, 0.1129, 0.5945, 0.0301, 0.2345, 0.0280, 0.0928,
        0.8236, 0.0683, 0.0034, 0.0120, 0.2342, 0.0860, 0.1953, 0.0435, 0.4409,
        0.4332, 0.0670, 0.0763, 0.2576, 0.1659, 0.1288, 0.3542, 0.2872, 0.2016,
        0.0282, 0.1572, 0.3839, 0.0405, 0.3566, 0.0618
		};
		Tensor3 x = tensor3_new(5, 4, 3, data);
		Tensor3 out = Softmax::forward(x);

		test_assert(tensor3_same_shape(x, out));
		tensor3_copy_gpu_to_cpu(out);
		float big_diff= 0;
		for (int i = 0; i < x.d1 * x.d2 * x.d3; i++) {
			float diff = abs(expected[i] - out.cpu_data[i]);
			big_diff = max(big_diff, diff);
		}
		test_assert(big_diff < 1e-4);
	}
	return true;
}

int test_cross_entropy_forward() 
{
	{
		float x_data[] = {
		-1.0713,  0.4462, -0.8402, -0.0768, -0.2241,  1.7291,  2.0391,  1.7352,
        -0.3897,  0.8211,  0.8207, -0.8696, -0.8852,  0.3269, -1.3966,  1.9687,
         0.8687, -1.0107, -0.7266,  0.3745,  1.3156, -0.3243,  1.6308,  1.3976
		};
		float y_data[] = {
		0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
        1., 0., 1., 0., 0., 0.
		};
		float expected[] = {
			1.6068
		};
		Tensor3 x = tensor3_new(4, 2, 3, x_data);
		Tensor3 y = tensor3_new(4, 2, 3, y_data);
		Tensor3 out = CrossEntropy::forward(x, y);

		test_assert(out.d1 == 1 && out.d2 == 1 && out.d3 == 1);
		tensor3_copy_gpu_to_cpu(out);
		
		{
			float big_diff= 0;
			for (int i = 0; i < out.d1 * out.d2 * out.d3; i++) {
				float diff = abs(expected[i] - out.cpu_data[i]);
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

int main() {
	int seed = time(0);
	srand(seed);
	int cnt_error = 0;
	cnt_error += !test_linear_shape();
	cnt_error += !test_softmax_forward();
	cnt_error += !test_cross_entropy_forward();

	if (cnt_error) {
		fprintf(stderr, "%d tests failed, seed: %d\n", cnt_error, seed);
		return 1;
	}

	return 0;
}
