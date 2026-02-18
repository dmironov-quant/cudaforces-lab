__global__ void kernel(int* cudaData) {
    if (threadIdx.x == 0) {
        int tmp = cudaData[0];
        cudaData[0] = cudaData[1];
        cudaData[1] = tmp;
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int hostData[2] = {a, b};
    int* cudaData;
    cudaMalloc(cudaData, 2 * sizeof(int));


    cudaMemcpy(cudaData, hostData, 2 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 1>>>(cudaData);
    cudaMemcpy(hostData, cudaData, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d %d", hostData[0], hostData[1]);

    cudaDeviceSynchronize();
    cudaFree(cudaData);

}
