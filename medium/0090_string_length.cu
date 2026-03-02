// https://cudaforces.com/problem/90

__global__ void kernel(const char *dstr, int *dcnt, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        atomicAdd(dcnt, 1);
    }
}

int main() {
    char hstr[101];
    scanf("%s", hstr);

    size_t len = strlen(hstr);

    char *dstr = nullptr;
    cudaMalloc(&dstr, len);
    cudaMemcpy(dstr, hstr, len, cudaMemcpyHostToDevice);

    int *dcnt = nullptr;
    cudaMalloc(&dcnt, sizeof(int));
    int init = 0;
    cudaMemcpy(dcnt, &init, sizeof(int), cudaMemcpyHostToDevice);

    int block = 128;
    int grid = (len + block - 1) / block;
    kernel<<<grid, block>>>(dstr, dcnt, len);
    cudaDeviceSynchronize();

    int out;
    cudaMemcpy(&out, dcnt, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", out);

    cudaFree(dstr);
    cudaFree(dcnt);
}
