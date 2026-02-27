// https://cudaforces.com/problem/41

__global__ void kernel(int *dA, int *dout, int n, int m) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < m && row < n) {
        dout[col * n + row] = dA[row * m + col];
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int hA[n][m];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            scanf("%d", &hA[i][j]);
        }
    }

    int *dA = nullptr;
    cudaMalloc(&dA, n * m * sizeof(int));
    cudaMemcpy(dA, hA, n * m * sizeof(int), cudaMemcpyHostToDevice);

    int *dout = nullptr;
    cudaMalloc(&dout, n * m * sizeof(int));

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    kernel<<<grid, block>>>(dA, dout, n, m);
    cudaDeviceSynchronize();

    int out[m][n];
    cudaMemcpy(out, dout, m * n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", out[i][j]);
        }
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dout);
}
