// https://cudaforces.com/problem/169

__global__ void kernel_sum(float *dX, int n, float *dsum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(dsum, dX[idx]);
    }
}

__global__ void kernel_var(float *dX, int n, float *dsum, float *ddiv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(ddiv, (dX[idx] - *dsum / n) * (dX[idx] - *dsum / n));
    }
}

__global__ void kernel_layer_normalization(float *dX, float *dgamma, float *dbeta, float *dsum, float *ddiv, float *dnorm, float eps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dnorm[idx] = (dX[idx] - *dsum / n) / sqrtf(eps + *ddiv / n) * dgamma[idx] + dbeta[idx];
    }
}

int main() {
    int n;
    scanf("%d", &n);

    float hX[n], hgamma[n], hbeta[n];
    for (int i = 0; i < n; ++i) scanf("%f", &hX[i]);
    for (int i = 0; i < n; ++i) scanf("%f", &hgamma[i]);
    for (int i = 0; i < n; ++i) scanf("%f", &hbeta[i]);

    float *dX = nullptr, *dgamma = nullptr, *dbeta = nullptr;
    cudaMalloc(&dX, n * sizeof(float));
    cudaMalloc(&dgamma, n * sizeof(float));
    cudaMalloc(&dbeta, n * sizeof(float));

    cudaMemcpy(dX, hX, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dgamma, hgamma, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, hbeta, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    
    float *dsum = nullptr;
    cudaMalloc(&dsum, sizeof(float));
    cudaMemset(dsum, 0.0, sizeof(float));

    kernel_sum<<<grid, block>>>(dX, n, dsum);
    cudaDeviceSynchronize();

    float *ddiv = nullptr;
    cudaMalloc(&ddiv, sizeof(float));
    cudaMemset(ddiv, 0.0, sizeof(float));

    kernel_var<<<grid, block>>>(dX, n, dsum, ddiv);
    cudaDeviceSynchronize();

    float *dnorm = nullptr;
    cudaMalloc(&dnorm, n * sizeof(float));

    float eps = 1e-5;

    kernel_layer_normalization<<<grid, block>>>(dX, dgamma, dbeta, dsum, ddiv, dnorm, eps, n);
    cudaDeviceSynchronize();

    float ans[n];
    cudaMemcpy(ans, dnorm, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%f ", ans[i]);

    cudaFree(dX);
    cudaFree(dgamma);
    cudaFree(dbeta);
    cudaFree(dsum);
    cudaFree(ddiv);
    cudaFree(dnorm);
}
