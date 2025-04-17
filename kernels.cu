#define KEY(i3, i2, i1, d3, d2, d1) (d3*0 + i3*d2*d1 + i2*d1 + i1)
#define SWAP(a, b, T) { T temp = a; a = b; b = temp; }

/* === Binary Ops  === */
__global__ void kernel_add(float *c, float * a, float *b,
						   int cd1, int cd2, int cd3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = 
		a[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] + 
		b[KEY(ci3, ci2, ci1, cd3, cd2, cd1)];
	}
}

__global__ void kernel_mul(float *c, float * a, float *b,
						   int cd1, int cd2, int cd3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = 
		a[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] *
		b[KEY(ci3, ci2, ci1, cd3, cd2, cd1)];
	}
}

__global__ void kernel_div(float *c, float * a, float *b,
						   int cd1, int cd2, int cd3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = 
		a[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] /
		b[KEY(ci3, ci2, ci1, cd3, cd2, cd1)];
	}
}

/* === Unary Ops  === */

__global__ void kernel_exp(float *c, float * a,
						   int cd1, int cd2, int cd3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = 
		exp(a[KEY(ci3, ci2, ci1, cd3, cd2, cd1)]);
	}
}

__global__ void kernel_log(float *c, float * a,
						   int cd1, int cd2, int cd3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = 
		log(a[KEY(ci3, ci2, ci1, cd3, cd2, cd1)]);
	}
}

/* === re-shape Ops  === */

__global__ void kernel_sum(float *c, float *a, 
							int cd1, int cd2, int cd3, // c dimensions
							int ad1, int ad2, int ad3) // a dimensions
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	int kn = ad3 / cd3;
	int jn = ad2 / cd2;
	int in = ad1 / cd1;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		float sum = 0;
		for (int k = 0; k < kn; k++)
		for (int j = 0; j < jn; j++)
		for (int i = 0; i < in; i++) {
			int ai3 = ci3 + k;
			int ai2 = ci2 + j;
			int ai1 = ci1 + i;
			sum += a[KEY(ai3, ai2, ai1, ad3, ad2, ad1)];
		}
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = sum;
	}
}

__global__ void kernel_broadcast(float *c, float *a, 
								 int cd1, int cd2, int cd3,
								 int ad1, int ad2, int ad3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	int f1 = cd1 / ad1;
	int f2 = cd2 / ad2;
	int f3 = cd3 / ad3;

	int ai3 = ci3 / f3;
	int ai2 = ci2 / f2;
	int ai1 = ci1 / f1;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		// c[id3][id2][id1] = a[ai3][ai2][ai1];
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = a[KEY(ai3, ai2, ai1, ad3, ad2, ad1)];
	}
}

// op is the dimension that does not change
__global__ void kernel_transpose(float *c, float *a, 
								int cd1, int cd2, int cd3,
								int op) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		int ai1 = ci1, ai2 = ci2, ai3 = ci3;
		int ad1 = cd1, ad2 = cd2, ad3 = cd3;
		if (op == 3) {
			SWAP(ci1, ci2, int);
			SWAP(cd1, cd2, int);
		}
		else if (op == 2) {
			SWAP(ci1, ci3, int);
			SWAP(cd1, cd3, int);
		}
		else if (op == 1) {
			SWAP(ci2, ci3, int);
			SWAP(cd2, cd3, int);
		}
		else {
			// TODO: Error message
		}
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = a[KEY(ai3, ai2, ai1, ad3, ad2, ad1)];
	}
}


/* === Mat mul  === */
__global__ void kernel_matmul(float *c, float *a, float *b, 
							   int cd1, int cd2, int cd3,
                               int ad1)
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	int ad3 = cd3, ad2 = cd2;
	int bd3 = cd3, bd2 = ad1, bd1 = cd1;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		float sum = 0;
		for (int k = 0; k < ad1; k++) {
			int ai2 = ci2, ai1 = k;
			int bi2 = k, bi1 = ci1;
			sum += a[KEY(ci3, ai2, ai1, ad3, ad2, ad1)] * 
				   b[KEY(ci3, bi2, bi1, bd3, bd2, bd1)];
		}
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = sum;
	}
}

/* === Scalar Ops  === */
__global__ void kernel_mul_scalar(float *c, float * a, float b,
						   int cd1, int cd2, int cd3) 
{
	int ci3 = blockIdx.z * blockDim.z + threadIdx.z;
	int ci2 = blockIdx.y * blockDim.y + threadIdx.y;
	int ci1 = blockIdx.x * blockDim.x + threadIdx.x;

	if (ci3 < cd3 && ci2 < cd2 && ci1 < cd1) {
		c[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] = 
		a[KEY(ci3, ci2, ci1, cd3, cd2, cd1)] * b; 
	}
}
