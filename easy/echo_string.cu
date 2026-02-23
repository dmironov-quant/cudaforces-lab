// https://cudaforces.com/problem/18

__global__ void kernel(const char *d_in) {
    if (threadIdx.x == 0) {
        printf("%s", d_in);
    }
}

int main() {
    char s[100];
    scanf("%s", s);

    char *d_s = nullptr;
    cudaMalloc(&d_s, 100 * sizeof(char));

    size_t bytes = strlen(s) + 1;
    cudaMemcpy(d_s, s, bytes, cudaMemcpyHostToDevice);
    
    kernel<<<1, 1>>>(d_s);
    cudaDeviceSynchronize();

    cudaFree(d_s);
}
