__global__ void kernel(const int* cudaDataA, const int* cudaDataB, int* ans, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ans[i] = cudaDataA[i] + cudaDataB[i];
}

int main() {
    int n;
    scanf("%d", &n);

    int A[n];
    int B[n];
    int C[n];
    for (int i = 0; i < n; ++i) scanf("%d", &A[i]);
    for (int i = 0; i < n; ++i) scanf("%d", &B[i]);

    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMalloc(&dB, n * sizeof(int));
    cudaMalloc(&dC, n * sizeof(int));

    cudaMemcpy(dA, A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;

    kernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%d ", C[i]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
