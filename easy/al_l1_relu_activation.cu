// https://cudaforces.com/problem/170

__global__ void kernel(const float* X, float* Y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) Y[idx] = fmaxf(0.0f, X[idx]);
}

int main() {
    int n;
    scanf("%d", &n);

    float a[n];
    for (int i = 0; i < n; ++i) scanf("%f", &a[i]);
    
    float* d_a;
    float* d_ans;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_ans, n * sizeof(float));


    cudaMemcpy(d_a, &a, n * sizeof(float), cudaMemcpyHostToDevice);
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(d_a, d_ans, n);
    cudaDeviceSynchronize();

    float ans[n];
    cudaMemcpy(ans, d_ans, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%f ", ans[i]);

    cudaFree(d_a);
    cudaFree(d_ans);
}
