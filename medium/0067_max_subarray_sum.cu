// https://cudaforces.com/problem/67

__global__ void kernel_sum(const int *dA, int *dout, int k, int w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        int out = 0;
        for (int j = 0; j < w; ++j) {
            if (i + j < n) {
                out += dA[i + j];
            }
        }
        dout[i] = out;
    }
}

__global__ void kernel_max(const int *dout, int k, int *dans) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        atomicMax(dans, dout[i]);
    }
}

int main() {
    int n, w;
    scanf("%d %d", &n, &w);

    int A[n];
    for (int i = 0; i < n; ++i) scanf("%d", &A[i]);

    int *dA = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMemcpy(dA, A, n * sizeof(int), cudaMemcpyHostToDevice);

    int k = n - w + 1;

    int *dout = nullptr;
    cudaMalloc(&dout, k * sizeof(int));

    int block = 16;
    int grid = (k + block - 1) / block;

    kernel_sum<<<grid, block>>>(dA, dout, k, w, n);
    cudaDeviceSynchronize();

    int int_min = INT_MIN;
    int *dans = nullptr;
    cudaMalloc(&dans, sizeof(int));
    cudaMemcpy(dans, &int_min, sizeof(int), cudaMemcpyHostToDevice);

    kernel_max<<<grid, block>>>(dout, k, dans);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, dans, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(dA);
    cudaFree(dout);
    cudaFree(dans);
}
