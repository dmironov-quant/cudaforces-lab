// https://cudaforces.com/problem/46

__global__ void kernel(int *dA, int i, int j, int k, int *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= idx  && idx <= j) {
        atomicAdd(d_out, dA[idx] * k);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hA[n];
    for (int i = 0; i < n; ++i) scanf("%d", &hA[i]);

    int i, j, k;
    scanf("%d %d %d", &i, &j, &k);

    int *dA = nullptr;
    cudaMalloc(&dA, n * sizeof(int));
    cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    int m = j - i + 1;
    int block = 256;
    int grid = (m + block - 1) / block;
    kernel<<<grid, block>>>(dA, i, j, k, d_out);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", out);

    cudaFree(dA);
    cudaFree(d_out);
}
