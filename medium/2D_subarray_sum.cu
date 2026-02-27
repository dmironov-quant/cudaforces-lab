// https://cudaforces.com/problem/65

__global__ void kernel(int *cudaData, int *d_out, int r1, int c1, int r2, int c2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + r1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + c1;
    if (i <= r2 && i < n && j <= c2 && j < n) {
        atomicAdd(d_out, cudaData[i * n + j]);
    }
}

int main() {
    int n, r1, c1, r2, c2;
    scanf("%d %d %d %d %d", &n, &r1, &c1, &r2, &c2);

    int hostData[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &hostData[i][j]);
        }
    }

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));


    int rows = r2 - r1 + 1;
    int cols = c2 - c1 + 1;
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    kernel<<<grid, block>>>(cudaData, d_out, r1, c1, r2, c2, n);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(cudaData);
    cudaFree(d_out);
}
