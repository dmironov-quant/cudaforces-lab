// https://cudaforces.com/problem/104

__global__ void kernel(int *d_a, int *d_ans) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (*d_a > 3 && i*i <= *d_a) {
        if (*d_a % i == 0) atomicExch(d_ans, 1);
    }
}

int main() {
    int a;
    scanf("%d", &a);

    int *d_a = nullptr;
    cudaMalloc(&d_a, sizeof(int));
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    int block = 256;
    int limit = (a > 0) ? (int)sqrt((double)a) : 0;
    int grid = (max(0, limit - 1) + block - 1) / block;
    kernel<<<grid, block>>>(d_a, d_out);
    cudaDeviceSynchronize();

    int out = 0;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (a == 1) printf("%d", 0);
    else printf("%d", out ? 0 : 1);

    cudaFree(d_a);
    cudaFree(d_out);
}
