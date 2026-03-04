// https://cudaforces.com/problem/29

__global__ void kernel(int *cudaData, int n) {
    extern __shared__ int s[];

    int t = threadIdx.x;

    for (int i = t; i < n; i += blockDim.x) {
        s[i] = cudaData[i];
    }
    __syncthreads();

    for (int phase = 0; phase < n; ++phase) {
        int i = 2 * t + (phase & 1);
        if (i + 1 < n) {
            if (s[i] < s[i + 1]) {
                int tmp = s[i];
                s[i] = s[i + 1];
                s[i + 1] = tmp;
            }
        }
        __syncthreads();
    }

    for (int i = t; i < n; i += blockDim.x) {
        cudaData[i] = s[i];
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int hostData[n];
    for (int i = 0; i < n; ++i) {
        scanf("%d", &hostData[i]);
    }

    int *cudaData = nullptr;
    cudaMalloc(&cudaData, n * sizeof(int));
    cudaMemcpy(cudaData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 256, n * sizeof(int)>>>(cudaData, n);

    cudaMemcpy(hostData, cudaData, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        printf("%d ", hostData[i]);
    }

    cudaFree(cudaData);
}
