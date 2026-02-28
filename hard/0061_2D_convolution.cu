// https://cudaforces.com/problem/61

__global__ void kernel(int *dA, int *dB, int *dC, int k, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < k && j < k) {
        int out = 0;
        for (int p = 0; p < m; ++p) {
            for (int q = 0; q < m; ++q) {
                if (i + p < n && j + q < n) {
                    out += dA[(i + p) * n + (j + q)] * dB[p * m + q];
                }
            }
        }
        dC[i * k + j] = out;
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int hA[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &hA[i][j]);
        }
    }

    int hB[m][m];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            scanf("%d", &hB[i][j]);
        }
    }

    int *dA = nullptr;
    cudaMalloc(&dA, n * n * sizeof(int));
    cudaMemcpy(dA, hA, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *dB = nullptr;
    cudaMalloc(&dB, m * m * sizeof(int));
    cudaMemcpy(dB, hB, m * m * sizeof(int), cudaMemcpyHostToDevice);

    int k = n - m + 1;

    int *dC = nullptr;
    cudaMalloc(&dC, k * k * sizeof(int));

    dim3 block(32, 32);
    dim3 grid((k + block.x - 1) / block.x, (k + block.y - 1) / block.y);
    kernel<<<grid, block>>>(dA, dB, dC, k, n, m);
    cudaDeviceSynchronize();

    int out[k][k];
    cudaMemcpy(out, dC, k * k * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%d ", out[i][j]);
        }
        printf("\n");
    }


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
