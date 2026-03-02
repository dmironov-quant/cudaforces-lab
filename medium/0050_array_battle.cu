// https://cudaforces.com/problem/50

__global__ void kernel(int *dA, int *dB, int *score, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (dA[idx] > dB[idx]) {
            atomicAdd(score, 1);
        } 
        else if (dA[idx] < dB[idx]) {
            atomicAdd(score, -1);
        }
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hA[n], hB[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hA[i]);
    for (int i = 0; i < n; ++i) scanf("%d", &hB[i]);

    int *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMalloc(&dB, n * sizeof(int));
    cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(int), cudaMemcpyHostToDevice);

    int *score = nullptr;
    cudaMalloc(&score, sizeof(int));
    cudaMemset(score, 0, sizeof(int));  

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(dA, dB, score, n);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, score, sizeof(int), cudaMemcpyDeviceToHost);

    if (out > 0) {
        printf("%d", 1);
    }
    else {
        printf("%d", 0);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(score);
}
