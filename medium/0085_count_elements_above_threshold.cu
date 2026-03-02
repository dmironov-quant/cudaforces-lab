// https://cudaforces.com/problem/85

__global__ void kernel(int *deviceA, int *dout, int n, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && deviceA[idx] >= target) {
        atomicAdd(dout, 1);
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int hostA[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hostA[i]);

    int *deviceA = nullptr;
    cudaMalloc(&deviceA, n * sizeof(int));
    cudaMemcpy(deviceA, hostA, n * sizeof(int), cudaMemcpyHostToDevice);

    int *dout = nullptr;
    cudaMalloc(&dout, sizeof(int));

    for (int i = 0; i < m; ++i) {
        int target;
        scanf("%d", &target);

        int init = 0;
        cudaMemcpy(dout, &init, sizeof(int), cudaMemcpyHostToDevice);

        int block = 256;
        int grid = (n + block - 1) / block;
        kernel<<<grid, block>>>(deviceA, dout, n, target);
        cudaDeviceSynchronize();

        int out;
        cudaMemcpy(&out, dout, sizeof(int), cudaMemcpyDeviceToHost);

        printf("%d\n", out);
    }

    cudaFree(deviceA);
    cudaFree(dout);
}
