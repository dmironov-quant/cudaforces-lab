__global__ void kernel(int* cudaData) {
    if (threadIdx.x == 0) {
        int idx1 = cudaData[3];
        int idx2 = cudaData[4];
        int tmp = cudaData[idx1];
        cudaData[idx1] = cudaData[idx2];
        cudaData[idx2] = tmp;
    }
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    int idx1, idx2;
    scanf("%d %d", &idx1, &idx2);
    --idx1;
    --idx2;

    int hostData[5] = {a, b, c, idx1, idx2};
    int* cudaData = nullptr;
    cudaMalloc(&cudaData, 5 * sizeof(int));

    cudaMemcpy(cudaData, hostData, 5 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 1>>>(cudaData);
    cudaDeviceSynchronize();
    cudaMemcpy(hostData, cudaData, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d %d %d", hostData[0], hostData[1], hostData[2]);

    cudaFree(cudaData);
}
