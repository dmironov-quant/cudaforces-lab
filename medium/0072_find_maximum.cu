// https://cudaforces.com/problem/72

__global__ void kernel(int *dA, int *dout, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax(dout, dA[idx]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hA[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hA[i]);

    int *dA = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);

    int *dout = nullptr;
    cudaMalloc(&dout, sizeof(int));
    int init = INT_MIN;
    cudaMemcpy(dout, &init, sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(dA, dout, n);
    cudaDeviceSynchronize();


    int out = 0;
    cudaMemcpy(&out, dout, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d", out);

    cudaFree(dA);
    cudaFree(dout);
}
