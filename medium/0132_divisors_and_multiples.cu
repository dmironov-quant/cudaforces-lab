// https://cudaforces.com/problem/132

__global__ void kernel(int n, int m, int *dcnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if ((m % (idx + 1) == 0) || ((idx + 1) % m == 0)) {
            atomicAdd(dcnt, 1);
        }
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int *dcnt = nullptr;
    cudaMalloc(&dcnt, sizeof(int));
    int init = 0;
    cudaMemcpy(dcnt, &init, sizeof(int), cudaMemcpyHostToDevice);

    int block = 128;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(n, m, dcnt);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, dcnt, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", out);

    cudaFree(dcnt);
}
