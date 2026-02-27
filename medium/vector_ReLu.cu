// https://cudaforces.com/problem/47

__global__ void kernel(int *dA, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dA[idx] = max(dA[idx], 0);
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

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(dA, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%d ", hA[i]);

    cudaFree(dA);
}
