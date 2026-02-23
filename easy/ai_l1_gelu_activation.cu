// https://cudaforces.com/problem/171



__global__ void kernel(float *cudaData, float *d_pi, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = cudaData[idx];
        float gelu = (x / 2) * (1 + tanh(sqrtf(2 / *d_pi) * (x + 0.044715 * x * x * x)));
        cudaData[idx] = gelu;
    } 
}

int main() {
    float pi = 3.14159265358979323846264338327950288;
    float *d_pi = nullptr;
    cudaMalloc(&d_pi, sizeof(float));
    cudaMemcpy(d_pi, &pi, sizeof(float), cudaMemcpyHostToDevice);

    int n;
    scanf("%d", &n);

    float hostData[n];
    for (int i = 0; i < n; ++i) scanf("%f", &hostData[i]);

    float *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(float));
    cudaMemcpy(cudaData, &hostData, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(cudaData, d_pi, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hostData, cudaData, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) printf("%f ", hostData[i]);

    cudaFree(cudaData);
}
