// https://cudaforces.com/problem/12

__global__ void kernel(int* cudaData) {
    if (threadIdx.x == 0) {
        if (cudaData[0] > cudaData[1]) {
            int tmp = cudaData[0];
            cudaData[0] = cudaData[1];
            cudaData[1] = tmp;
        }
        if (cudaData[1] > cudaData[2]) {
            int tmp = cudaData[1];
            cudaData[1] = cudaData[2];
            cudaData[2] = tmp;
        }
        if (cudaData[0] > cudaData[1]) {
            int tmp = cudaData[0];
            cudaData[0] = cudaData[1];
            cudaData[1] = tmp;
        }
    }
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);

    int hostData[3] = {a, b, c};

    int* cudaData;
    cudaMalloc(cudaData, 3 * sizeof(int));

    cudaMemcpy(cudaData, hostData, 3 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 1>>>(cudaData);
    cudaMemcpy(hostData, cudaData, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d %d %d", hostData[0], hostData[1], hostData[2]);

    cudaDeviceSynchronize();
    cudaFree(cudaData);
}
