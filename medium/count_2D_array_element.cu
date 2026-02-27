// https://cudaforces.com/problem/48

__global__ void kernel(int *dA, int n, int *ans, int target) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && dA[i * n + j] == target) {
        atomicAdd(ans, 1);
    }
}

int main() {
    int n, target;
    scanf("%d %d", &n, &target);

    int hA[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &hA[i][j]);
        }
    }

    int *dA = nullptr;
    cudaMalloc(&dA, n * n * sizeof(int));
    cudaMemcpy(dA, hA, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *ans = nullptr;
    cudaMalloc(&ans, sizeof(int));
    cudaMemset(ans, 0, sizeof(int));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    kernel<<<grid, block>>>(dA, n, ans, target);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, ans, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(dA);
    cudaFree(ans);
}
