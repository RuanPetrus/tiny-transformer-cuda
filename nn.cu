#include "tensor.cu"

int sample_ix(Tensor3 probs)
{
	static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range [0, 1)
	float pred = dis(e);
	// tensor3_show(probs);
	// printf("%.5f\n\n" ,pred);

	tensor3_copy_gpu_to_cpu(probs);
	int k = 0;
	int t = probs.d2 - 1;

	for (int i = 0; i < probs.d1-1; i++) {
		float v = probs.cpu_data[KEY(k, t, i, probs.d3, probs.d2, probs.d1)];
		// printf("%.5f ", v);
		if (pred <= v) {
			return i;
		}
		else pred -= v;
	}
	// printf("\n");
	return probs.d1 - 1;
}


struct Config 
{
	int n_embd, n_alphabet, n_context, n_batch;
};

struct Linear 
{
	Tensor3 w;
	Tensor3 b;
	bool bias;

	Linear() {}

	Linear(int fan_in, int fan_out, bool _bias = false) 
	{
		bias = _bias;
		if (bias) {
			TODO("Not implemented");
		}
		w = tensor3_randn(fan_out, fan_in, 1);
		w = tensor3_mul_scalar(w, 1.0/(sqrtf(fan_in)));
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

struct TransformerOutput 
{
	Tensor3 logits, loss;
};

// TODO: starting with a simple bigram model
struct Transformer
{
	Linear embw_token;
	int n_alphabet, n_context;

	Transformer(Config config) 
	{
		n_alphabet = config.n_alphabet;
		n_context  = config.n_context;
		embw_token = Linear(n_alphabet, n_alphabet);
	}

	TransformerOutput forward(const Tensor3 x, Tensor3 y) 
	{
		Tensor3 logits = embw_token.forward(x);
		Tensor3 loss = CrossEntropy::forward(logits, y);
		return {logits, loss};
	}

	int* generate(int tokens_to_generate)
	{
		int *generated_tokens = (int *) malloc((tokens_to_generate + 1) * sizeof(int));
		generated_tokens[0] = 0;
		int generated_sz = 1;

		while(tokens_to_generate--) {
			int sz = generated_sz;
			int *xi = generated_tokens;
			if (sz > n_context) {
				xi += (sz-n_context);
				sz = n_context;
			}

			Tensor3 x = tensor3_one_hot_cpu(xi, 1, sz, n_alphabet);
			Tensor3 y = tensor3_one_hot_cpu(xi, 1, sz, n_alphabet); // TODO: add train mode and generate mode

			TransformerOutput out = forward(x, y);
			Tensor3 prob = Softmax::forward(out.logits);
			int ix = sample_ix(prob);
			generated_tokens[generated_sz++] = ix;
		}
		return generated_tokens + 1;
	}
};

