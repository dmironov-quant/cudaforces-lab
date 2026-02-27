// https://cudaforces.com/problem/75

__global__ void kernel(int *d_hh, int *d_mm) {
    if (threadIdx.x == 0) {
        printf("%d", 60 * (*d_hh) + *d_mm);
    }
}

int main() {
    int hh, mm;
    scanf("%d:%d", &hh, &mm);

    int *d_hh = nullptr;
    int *d_mm = nullptr;
    cudaMalloc(&d_hh, sizeof(int));
    cudaMalloc(&d_mm, sizeof(int));

    cudaMemcpy(d_hh, &hh, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mm, &mm, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_hh, d_mm);
    cudaDeviceSynchronize();

    cudaFree(d_hh);
    cudaFree(d_mm);
}
