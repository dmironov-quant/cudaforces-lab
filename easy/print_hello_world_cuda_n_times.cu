__global__ void kernel(int* d_dummy, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        printf("Hello World Cuda\n");
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));

    kernel<<<1, n>>>(d_dummy, n);

    cudaDeviceSynchronize();
    cudaFree(d_dummy);
}
