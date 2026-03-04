// https://cudaforces.com/problem/143

__global__ void kernel(char *dch) {
    if (threadIdx.x == 0) {
        int x = *dch - 'A' + 1;
        printf("%d", x);
    }
}

int main() {
    char ch;
    scanf("%c", &ch);

    char *dch = nullptr;
    cudaMalloc(&dch, sizeof(char));
    cudaMemcpy(dch, &ch, sizeof(char), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(dch);
    cudaDeviceSynchronize();

    cudaFree(dch);
}
