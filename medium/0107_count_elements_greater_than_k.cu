// https://cudaforces.com/problem/107


__global__ void kernel(int *cudaData, int *dout, int n, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && cudaData[idx] > target) {
        atomicAdd(dout, 1);
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int dA[n];
    for (int i = 0; i < n; ++i) scanf("%d", &dA[i]);

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, dA, n * sizeof(int), cudaMemcpyHostToDevice);

    int *dout = nullptr;
    cudaMalloc(&dout, sizeof(int));

    for (int i = 0; i < m; ++i) {
        int target;
        scanf("%d", &target);

        int init = 0;
        cudaMemcpy(dout, &init, sizeof(int), cudaMemcpyHostToDevice);

        int block = 256;
        int grid = (n + block - 1) / block;
        kernel<<<grid, block>>>(cudaData, dout, n, target);
        cudaDeviceSynchronize();

        int out = 0;
        cudaMemcpy(&out, dout, sizeof(int), cudaMemcpyDeviceToHost);
        printf("%d\n", out);
    }

    cudaFree(cudaData);
    cudaFree(dout);
}
