// https://cudaforces.com/problem/45

__global__ void kernel(int *dA, int *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(d_out, dA[idx]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hA[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hA[i]);

    int *dA = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(dA, d_out, n);
    cudaDeviceSynchronize();

    int out = 0;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(dA);
}
