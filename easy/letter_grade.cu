// https://cudaforces.com/solution/681


__global__ void kernel(int *d_p) {
    if (threadIdx.x == 0) {
        if (90 <= *d_p && *d_p <= 100) printf("A");
        else if (80 <= *d_p && *d_p <= 89) printf("B");
        else if (70 <= *d_p && *d_p <= 79) printf("C");
        else if (60 <= *d_p && *d_p <= 69) printf("D");
        else printf("E");
    }
}

int main() {
    int p;
    scanf("%d", &p);

    int *d_p = nullptr;
    cudaMalloc(&d_p, sizeof(int));
    cudaMemcpy(d_p, &p, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_p);
    cudaDeviceSynchronize();

    cudaFree(d_p);
}
