// https://cudaforces.com/problem/69

__global__ void kernel(int *cudaData, int *dmaxsum, int w, int p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < p && j < p && k < p) {
        int sum = 0;
        for (int x = i; x < i + w; ++x) {
            for (int y = j; y < j + w; ++y) {
                for (int z = k; z < k + w; ++z) {
                    int idx = (x * n + y) * n + z;
                    sum += cudaData[idx];
                }
            }
        }
        atomicMax(dmaxsum, sum);
    }
}

int main() {
    int n, w;
    scanf("%d %d", &n, &w);

    int hostData[n][n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                scanf("%d", &hostData[i][j][k]);
            }
        }
    }

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * n * n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * n * n * sizeof(int), cudaMemcpyHostToDevice);

    int *dmaxsum = nullptr;
    cudaMalloc(&dmaxsum, sizeof(int));
    int init = INT_MIN;
    cudaMemcpy(dmaxsum, &init, sizeof(int), cudaMemcpyHostToDevice);

    int p = n - w + 1;

    dim3 block(8, 8, 8);
    dim3 grid((p + block.x - 1) / block.x, (p + block.y - 1) / block.y, (p + block.z - 1) / block.z);
    kernel<<<grid, block>>>(cudaData, dmaxsum, w, p, n);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, dmaxsum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", out);

    cudaFree(cudaData);
    cudaFree(dmaxsum);
}
