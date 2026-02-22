// https://cudaforces.com/problem/9

__global__ void kernel(char* d_c, char* d_out) {
    if (threadIdx.x == 0) {
        *d_out = *d_c;
    }
}

int main() {
    char c;
    scanf("%c", &c);

    char* d_c = nullptr;
    cudaMalloc(&d_c, sizeof(char));
    cudaMemcpy(d_c, &c, sizeof(char), cudaMemcpyHostToDevice);
    
    char* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(char));
    
    kernel<<<1, 1>>>(d_c, d_out);
    cudaDeviceSynchronize();
    
    char ans;
    cudaMemcpy(&ans, d_out, sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("%c", ans);
    
    cudaFree(d_c);
    cudaFree(d_out);
}
