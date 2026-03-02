// https://cudaforces.com/problem/43

__global__ void kernel(int *cudaData, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        int tmp = cudaData[idx];
        cudaData[idx] = cudaData[n - 1 - idx];
        cudaData[n - 1 - idx] = tmp;
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hostData[n];
    for (int i = 0; i < n; ++i) {
        scanf("%d", &hostData[i]);
    }

    int *cudaData;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n / 2 + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hostData, cudaData, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        printf("%d ", hostData[i]);
    }

    cudaFree(cudaData);
}
