// https://cudaforces.com/problem/166

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

int main() {
    int n;
    scanf("%d", &n);

    float A[n], B[n];
    for (int i = 0; i < n; ++i) scanf("%f", &A[i]);
    for (int i = 0; i < n; ++i) scanf("%f", &B[i]);

    float* d_A, d_B;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    
    cudaMemcpy(d_A, &A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &B, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_C;
    cudaMalloc(&d_C, n * sizeof(float));

    int block = 256;
    int grid = (n + block - 1) / block; 
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    float C[n];
    cudaMemcpy(C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%f ", C[i]);

    cudaFree(d_A);
    cudaFree(d_B);
}
