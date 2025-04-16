#include <stdio.h>

__global__ void cuda_hello() {
	printf("hello world from gpu thread: %d\n", threadIdx.x);
}

void hello_world_example() {
	cuda_hello<<<1, 10>>>();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();
	printf("Hello from cpu\n");
	return;
}

__global__ void vector_add(float *a, float * b, float *c) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	c[id] = a[id] + b[id];
}

void vector_add_example() {
	const int N = 256;
	const int BLOCK_SIZE = 32;
	float *a = (float *) malloc(sizeof(float) * N);
	float *b = (float *) malloc(sizeof(float) * N);
	float *c = (float *) malloc(sizeof(float) * N);
	float *d_a, *d_b, *d_c;

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i;
	}

	printf("A value:");
	for (int i = 0; i < N; i++) {
		printf(" %.2f", a[i]);
	}
	printf("\n");

	printf("B value:");
	for (int i = 0; i < N; i++) {
		printf(" %.2f", b[i]);
	}
	printf("\n");

	cudaMalloc(&d_a, sizeof(float) * N);
	cudaMalloc(&d_b, sizeof(float) * N);
	cudaMalloc(&d_c, sizeof(float) * N);
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	vector_add<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);

	printf("C value:");
	for (int i = 0; i < N; i++) {
		printf(" %.2f", c[i]);
	}
	printf("\n");

	free(a); free(b), free(c);
	cudaFree(d_a); cudaFree(d_b), cudaFree(d_c);
}

int main() {
	vector_add_example();
	return 0;
}
