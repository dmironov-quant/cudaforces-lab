// https://cudaforces.com/problem/124

__global__ void kernel(const char* c, int *d_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *d_b) {
        printf("%c ", *c);
    }
}

int main() {
    char c;
    int b;
    scanf(" %c %d", &c, &b);

    char *d_c = nullptr;
    cudaMalloc(&d_c, sizeof(char));

    int *d_b = nullptr;
    cudaMalloc(&d_b, sizeof(int));

    cudaMemcpy(d_c, &c, sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (b + block - 1) / block;
    kernel<<<grid, block>>>(d_c, d_b);
    cudaDeviceSynchronize();

    cudaFree(d_c);
    cudaFree(d_b);
}
