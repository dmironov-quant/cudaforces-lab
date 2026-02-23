// https://cudaforces.com/problem/32

__global__ void kernel(int *d_a, int *d_b) {
    if (threadIdx.x == 0) {
        printf("%d", *d_a % *d_b);
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int *d_a = nullptr;
    int *d_b = nullptr;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_a, d_b);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
}
