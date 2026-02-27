// https://cudaforces.com/problem/74

__global__ void kernel(const int *cudaData, int n, int a, int *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && cudaData[idx] == a) {
            atomicExch(d_out, 1);
    }
}

int main() {
    int n, a;
    scanf("%d %d", &n, &a);

    int hostData[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hostData[i]);

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, n, a, d_out);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(cudaData);
    cudaFree(d_out);
}
