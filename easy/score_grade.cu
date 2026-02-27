// https://cudaforces.com/problem/102

__global__ void kernel(int *d_a, int *d_b) {
    if (threadIdx.x == 0) {
        if (*d_a >= 90 && *d_b >= 90) printf("A");
        else if (*d_a >= 90 && *d_b < 90) printf("B");
        else if (*d_a < 90 && *d_b >= 90) printf("C");
        else printf("D");
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

    kernel<<<1, 1>>>(d_a, d_b);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
}
