// https://cudaforces.com/problem/15

__global__ void kernel(int *cudaData) {
    if (threadIdx.x == 0) {
        int a1 = cudaData[0], b1 = cudaData[1];
        int a2 = cudaData[2], b2 = cudaData[3];
        int a3 = cudaData[4], b3 = cudaData[5];

        if ((a2 < a1) || (a1 == a2 && b2 < b1)) {
            
            int tmp1 = a1;
            a1 = a2;
            a2 = tmp1;

            int tmp2 = b1;
            b1 = b2;
            b2 = tmp2;
        }
        if ((a2 > a3) || (a2 == a3 && b2 > b3)) {
            int tmp1 = a2;
            a2 = a3;
            a3 = tmp1;

            int tmp2 = b2;
            b2 = b3;
            b3 = tmp2; 
        }
        if ((a2 < a1) || (a1 == a2 && b2 < b1)) {
            int tmp1 = a1;
            a1 = a2;
            a2 = tmp1;

            int tmp2 = b1;
            b1 = b2;
            b2 = tmp2;
        }
        
        cudaData[0] = a1;
        cudaData[1] = b1;
        cudaData[2] = a2;
        cudaData[3] = b2;
        cudaData[4] = a3;
        cudaData[5] = b3;
    }
}

int main() {
    int a1, b1;
    int a2, b2;
    int a3, b3;

    scanf("%d %d", &a1, &b1);
    scanf("%d %d", &a2, &b2);
    scanf("%d %d", &a3, &b3);

    int hostData[6] = {a1, b1, a2, b2, a3, b3};

    int *cudaData;
    cudaMalloc(&cudaData, 6 * sizeof(int));

    cudaMemcpy(cudaData, hostData, 6 * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(cudaData);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&hostData, cudaData, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d %d\n", hostData[0], hostData[1]);
    printf("%d %d\n", hostData[2], hostData[3]);
    printf("%d %d\n", hostData[4], hostData[5]);

    cudaFree(cudaData);
}
