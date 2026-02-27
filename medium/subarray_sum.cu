// https://cudaforces.com/problem/64

__global__ void kernel(int *cudaData, int *d_out, int s, int e, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + s;
    if (idx <= e && idx < n) {
        atomicAdd(d_out, cudaData[idx]);
    }
}

int main() {
    int n, s, e;
    scanf("%d %d %d", &n, &s, &e);

    int hostData[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hostData[i]);

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    int block = 256;
    int len = e - s + 1;
    int grid = (len + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, d_out, s, e, n);
    cudaDeviceSynchronize();

    int out = 0;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(cudaData);
    cudaFree(d_out);
}
