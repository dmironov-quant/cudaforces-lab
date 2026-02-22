// https://cudaforces.com/problem/138

__global__ void kernel(int *d_a, int *d_b, int *d_c, int *d_out) {
    if (threadIdx.x == 0) {
        int first  = *d_a;
        int second = *d_b;
        int third  = *d_c;
        *d_out = first * second * third;
    }
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);

    int *d_a = nullptr;
    int *d_b = nullptr;
    int *d_c = nullptr;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));

    kernel<<<1, 1>>>(d_a, d_b, d_c, d_out);
    cudaDeviceSynchronize();

    int output;
    cudaMemcpy(&output, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d", output);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
}
