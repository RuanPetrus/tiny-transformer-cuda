#include "tensor.cu"

struct Config 
{
	int n_embd, n_alphabet, n_context, n_batch;
};

struct Linear {
	Tensor3 w;
	Tensor3 b;
	bool bias;

	Linear(int fan_in, int fan_out, bool _bias = false) 
	{
		bias = _bias;
		if (bias) {
			TODO("Not implemented");
		}
		w = tensor3_randn(fan_out, fan_in, 1);
	}

	Tensor3 forward(const Tensor3 x) 
	{
		if (bias) {
			TODO("Not implemented");
		}
		Tensor3 out = tensor3_matmul(x, w);
		return out;
	}
};

struct Softmax 
{
	static Tensor3 forward(const Tensor3 x) 
	{
		Tensor3 ex = tensor3_exp(x);
		Tensor3 sum = tensor3_sum(ex, 1, ex.d2, ex.d3);
		Tensor3 out = tensor3_div(ex, sum);
		return out;
	}
};

struct CrossEntropy 
{
	static Tensor3 forward(const Tensor3 x, const Tensor3 y) 
	{
		int A = x.d1;
		int T = x.d2;
		int B = x.d3;

		Tensor3 prob = Softmax::forward(x);
		Tensor3 logprob = tensor3_log(prob);
		Tensor3 minuslogprob = tensor3_mul_scalar(logprob, -1);

		Tensor3 xv = tensor3_view(minuslogprob, A*T*B, 1, 1);
		Tensor3 yv = tensor3_view(y, A*T*B, 1, 1);
		Tensor3 yt = tensor3_transpose(yv, 1, 2);
		Tensor3 syt = tensor3_mul_scalar(yt, 1.0f/((float)T*B));
		Tensor3 out = tensor3_matmul(xv, syt);
		return out;
	}
};

