// https://cudaforces.com/problem/168

__global__ void kernel(const float* d_a, float* d_b, float* d_MAX, float* d_SUM, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float e = expf(d_a[idx] - *d_MAX);
        d_b[idx] = e;
        atomicAdd(d_SUM, e);
    } 
}

int main() {
    int n;
    scanf("%d", &n);

    float MAX;
    float a[n];
    for (int i = 0; i < n; ++i) {
        float x;
        scanf("%f", &x);
        if (i == 0) MAX = x;
        if (x > MAX) MAX = x;
        a[i] = x;
    }

    float* d_MAX = nullptr;
    cudaMalloc(&d_MAX, sizeof(float));
    cudaMemcpy(d_MAX, &MAX, sizeof(float), cudaMemcpyHostToDevice);

    float* d_SUM = nullptr;
    cudaMalloc(&d_SUM, sizeof(float));
    cudaMemset(d_SUM, 0.0, sizeof(float));

    float* d_a = nullptr;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    
    float* d_b = nullptr;
    cudaMalloc(&d_b, n * sizeof(float));

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(d_a, d_b, d_MAX, d_SUM, n);
    cudaDeviceSynchronize();

    float ans[n];
    cudaMemcpy(ans, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    float norm;
    cudaMemcpy(&norm, d_SUM, sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%f ", ans[i] / norm);

    cudaFree(d_MAX);
    cudaFree(d_SUM);
    cudaFree(d_a);
    cudaFree(d_b);
}
