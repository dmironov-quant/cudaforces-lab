// https://cudaforces.com/problem/42

__global__ void kernel(int *dA, int *dB, int *dout, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        int out = 0;
        for (int i = 0; i < m; ++i) {
            out += dA[idx + i] * dB[i];
        }
        dout[idx] = out;
    }
}


int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int hA[n], hB[m];
    for (int i = 0; i < n; ++i) scanf("%d", &hA[i]);
    for (int i = 0; i < m; ++i) scanf("%d", &hB[i]);

    int *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMalloc(&dB, n * sizeof(int));

    cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(int), cudaMemcpyHostToDevice);

    int k = n - m + 1;

    int *dout = nullptr;
    cudaMalloc(&dout, k * sizeof(int));

    int block = 64;
    int grid = (k + block - 1) / block;
    kernel<<<grid, block>>>(dA, dB, dout, m, k);
    cudaDeviceSynchronize();

    int out[k];
    cudaMemcpy(out, dout, k * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; ++i) printf("%d ", out[i]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dout);
}
