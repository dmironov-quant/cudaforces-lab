// https://cudaforces.com/problem/44


__global__ void kernel(int *dA, int *d_count, int target, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && dA[idx] == target) {
        atomicAdd(d_count, 1);
    }
}

int main() {
    int n, target;
    scanf("%d %d", &n, &target);

    int hA[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hA[i]);

    int *dA = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_count = nullptr;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(dA, d_count, target, n);
    cudaDeviceSynchronize();

    int cnt;
    cudaMemcpy(&cnt, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", cnt);

    cudaFree(dA);
    cudaFree(d_count);
}
