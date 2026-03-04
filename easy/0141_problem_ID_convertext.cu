// https://cudaforces.com/problem/141

__global__ void kernel(int *m) {
    if (threadIdx.x == 0) {
        char ch = (char)('A' + (*m) - 1);
        printf("%c", ch);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int *m = nullptr;
    cudaMalloc(&m, sizeof(int));
    cudaMemcpy(m, &n, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(m);
    cudaDeviceSynchronize();

    cudaFree(m);
}
