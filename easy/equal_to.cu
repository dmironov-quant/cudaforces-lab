// https://cudaforces.com/problem/37

__global__ void kernel(const int* cudaData, int* ans) {
    if (threadIdx.x == 0) {
        *ans = (cudaData[0] == cudaData[1]) ? 1 : 0;
    }
} 

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int hostData[2] = {a, b};

    int* cudaData = nullptr;
    int* ans = nullptr;

    cudaMalloc(&cudaData, 2 * sizeof(int));
    cudaMalloc(&ans, sizeof(int));

    cudaMemcpy(cudaData, hostData, 2 * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(cudaData, ans);
    cudaDeviceSynchronize();

    int res = 0;
    cudaMemcpy(&res, ans, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", res);

    cudaFree(cudaData);
    cudaFree(ans);
}
