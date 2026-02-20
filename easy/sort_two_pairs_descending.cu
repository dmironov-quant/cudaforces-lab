// https://cudaforces.com/problem/16

__global__ void kernel(int* cudaData) {
    if (threadIdx.x == 0) {
        int a1 = cudaData[0], b1 = cudaData[1];
        int a2 = cudaData[2], b2 = cudaData[3];
        if (a2 > a1 || (a2 == a1 && b2 > b1)) {
            cudaData[0] = a2;
            cudaData[1] = b2;
            cudaData[2] = a1;
            cudaData[3] = b1;
        }
    }
}

int main() {
    int a1, b1, a2, b2;
    scanf("%d %d", &a1, &b1);
    scanf("%d %d", &a2, &b2);

    int hostData[4] = {a1, b1, a2, b2};
    
    int* cudaData = nullptr;
    cudaMalloc(&cudaData, 4 * sizeof(int));

    cudaMemcpy(cudaData, hostData, 4 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 1>>>(cudaData);
    cudaDeviceSynchronize();
    cudaMemcpy(hostData, cudaData, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d %d\n", hostData[0], hostData[1]);
    printf("%d %d\n", hostData[2], hostData[3]);

    cudaFree(cudaData);
}
