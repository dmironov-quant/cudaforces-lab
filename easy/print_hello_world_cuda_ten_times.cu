__global__ void kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World Cuda\n");
}

int main() {
    int *h_a, *d_a;
    h_a = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_a, sizeof(int));
    
    for (int i = 0; i < 10; ++i) {
        kernel<<<1, 1>>>();
    }
}
