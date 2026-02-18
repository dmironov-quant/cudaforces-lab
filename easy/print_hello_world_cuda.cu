__global__ void kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World Cuda");
}

int main() {
    int *h_a, *d_a;
    h_a = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_a, sizeof(int));
    
    kernel<<<1, 1>>>();
}
