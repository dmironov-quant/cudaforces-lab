// https://cudaforces.com/problem/5

__global__ void kernel(int *d_k, int *d_m, int *d_e, int *d_s, int *d_c, int *d_t) {
    if (threadIdx.x == 0) {
        int total = *d_k + *d_m + *d_e + *d_s + *d_c;
        printf("%d %d", total, total / 5);
    }
}

int main() {
    int k, m, e, s, c, t;
    scanf("%d %d %d %d %d %d", &k, &m, &e, &s, &c, &t);

    int *d_k = nullptr;
    int *d_m = nullptr;
    int *d_e = nullptr;
    int *d_s = nullptr;
    int *d_c = nullptr;
    int *d_t = nullptr;

    cudaMalloc(&d_k, sizeof(int));
    cudaMalloc(&d_m, sizeof(int));
    cudaMalloc(&d_e, sizeof(int));
    cudaMalloc(&d_s, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));
    cudaMalloc(&d_t, sizeof(int));

    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, &e, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, &s, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, &t, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_k, d_m, d_e, d_s, d_c, d_t);
    cudaDeviceSynchronize();

    cudaFree(d_k);
    cudaFree(d_m);
    cudaFree(d_e);
    cudaFree(d_s);
    cudaFree(d_c);
    cudaFree(d_t);
}
