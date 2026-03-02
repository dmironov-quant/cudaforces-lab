// https://cudaforces.com/problem/25

__global__ void kernel(const int *dA, const int *dB, int *dC, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        int out = 0;
        for (int k = 0; k < n; ++k) {
            out += dA[i * n + k] * dB[k * n + j];
        }
        dC[i * n + j] = out;
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hA[n][n], hB[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &hA[i][j]);
        }
    } 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &hB[i][j]);
        }
    }

    int *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, n * n * sizeof(int));
    cudaMalloc(&dB, n * n * sizeof(int));
    cudaMemcpy(dA, hA, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *dC = nullptr;
    cudaMalloc(&dC, n * n * sizeof(int));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    kernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    int out[n][n];
    cudaMemcpy(out, dC, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", out[i][j]);
        }
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
