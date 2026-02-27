// https://cudaforces.com/problem/40

__global__ void kernel(int *d_a, int *d_b, int *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        d_out[i * n + j] = d_a[i * n + j] + d_b[i * n + j];
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int Ah[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &Ah[i][j]);
        }
    }

    int Bh[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &Bh[i][j]);
        }
    }

    int *d_a = nullptr;
    cudaMalloc(&d_a, n * n * sizeof(int));
    cudaMemcpy(d_a, Ah, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_b = nullptr;
    cudaMalloc(&d_b, n * n * sizeof(int));
    cudaMemcpy(d_b, Bh, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, n * n * sizeof(int));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    int out[n][n];
    cudaMemcpy(out, d_out, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", out[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
