// https://cudaforces.com/problem/21

__global__ void kernel(int *d_a, int *d_b, int *d_out) {
    if (threadIdx.x == 0) {
        *d_out = *d_a + *d_b;
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int *d_a = nullptr;
    int *d_b = nullptr;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));

    kernel<<<1, 1>>>(d_a, d_b, d_out);
    cudaDeviceSynchronize();

    int output;
    cudaMemcpy(&output, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", output);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
