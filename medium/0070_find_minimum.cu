// https://cudaforces.com/problem/70

__global__ void kernel(const int *cudaData, int *dans, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMin(dans, cudaData[idx]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hostData[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hostData[i]);

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    int *dans = nullptr;
    cudaMalloc(&dans, sizeof(int));
    int init = INT_MAX;
    cudaMemcpy(dans, &init, sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, dans, n);
    cudaDeviceSynchronize();

    int out = 0;
    cudaMemcpy(&out, dans, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(cudaData);
}
