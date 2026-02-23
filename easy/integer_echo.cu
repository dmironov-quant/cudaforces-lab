// https://cudaforces.com/problem/126

__global__ void kernel(int *d_a, int *d_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_b) {
        printf("%d ", *d_a);
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

    int block = 256;
    int grid = (b + block - 1) / block;
    kernel<<<grid, block>>>(d_a, d_b);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
}
