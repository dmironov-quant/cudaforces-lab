// https://cudaforces.com/problem/68

__global__ void kernel(int *cudaData, int *dmax, int n, int w, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < k && j < k) {
        int sum = 0;
        for (int x = i; x < i + w; ++x) {
            for (int y = j; y < j + w; ++y) {
                sum += cudaData[x * n + y];
            }
        }
        atomicMax(dmax, sum);
    }
}

int main() {
    int n, w;
    scanf("%d %d", &n, &w);

    int hostData[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &hostData[i][j]);
        }
    }

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *dmax = nullptr;
    cudaMalloc(&dmax, sizeof(int));
    int init = -100000;
    cudaMemcpy(dmax, &init, sizeof(int), cudaMemcpyHostToDevice);

    int k = n - w + 1;

    dim3 block(16, 16);
    dim3 grid((k + block.x - 1) / block.x, (k + block.y - 1) / block.y);
    kernel<<<grid, block>>>(cudaData, dmax, n, w, k);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, dmax, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", out);

    cudaFree(cudaData);
    cudaFree(dmax);
}
