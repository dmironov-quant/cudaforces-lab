// https://cudaforces.com/problem/36

__global__ void kernel(int* cudaData, int* ans) {
    if (threadIdx.x == 0) {
        *ans = (cudaData[0] <= cudaData[1]) ? 1 : 0;
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int hostData[2] = {a, b};

    int* cudaData = nullptr;
    cudaMalloc(&cudaData, 2 * sizeof(int));

    int* ans = nullptr;
    cudaMalloc(&ans, sizeof(int));

    cudaMemcpy(cudaData, hostData, 2 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 1>>>(cudaData, ans);
    cudaDeviceSynchronize();
    cudaMemcpy(&hostData, cudaData, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    int res;
    cudaMemcpy(&res, ans, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", res);

    cudaFree(cudaData);
    cudaFree(ans);
}
