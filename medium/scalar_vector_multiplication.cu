// https://cudaforces.com/problem/39

__global__ void kernel(int *cudaData, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cudaData[idx] *= k;
    }
}

int main() {
    int n, k;
    scanf("%d %d", &n, &k);

    int hostData[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hostData[i]);

    int *cudaData;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, &hostData, n * sizeof(int),cudaMemcpyHostToDevice);
    
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hostData, cudaData, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%d ", hostData[i]);

    cudaFree(cudaData);
}
