// https://cudaforces.com/problem/49

__global__ void kernel(int *cudaData, int *dsum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(dsum, cudaData[idx]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hostData[n];
    for (int i = 0; i < n; ++i) {
        scanf("%d", &hostData[i]);
    }

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    int *dsum = nullptr;
    cudaMalloc(&dsum, sizeof(int));
    int init = 0;
    cudaMemcpy(dsum, &init, sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, dsum, n);
    cudaDeviceSynchronize();

    int sum;
    cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", sum);


    cudaFree(cudaData);
    cudaFree(dsum);
}
