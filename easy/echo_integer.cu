// https://cudaforces.com/problem/19

__global__ void kernel(const int *d_in) {
    if (threadIdx.x == 0) {
        printf("%d", *d_in);
    }
}

int main() {
    int a;
    scanf("%d", &a);

    int *d_a = nullptr;
    cudaMalloc(&d_a, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_a);
    cudaDeviceSynchronize();

    cudaFree(d_a);
}
