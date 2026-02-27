// https://cudaforces.com/problem/134


__global__ void kernel(int *d_a, int *d_b, int *d_r, int *d_c, int *d_d) {
    if (threadIdx.x == 0) {
        int dist = (*d_c - *d_a)*(*d_c - *d_a) + (*d_d - *d_b)*(*d_d - *d_b);
        int radius = (*d_r) * (*d_r);        
        if (dist < radius) {
            printf("IN");
        }
        else if (dist > radius) {
            printf("OUT");
        } 
        else {
            printf("ON");
        }
    }
}

int main() {
    int a, b, r;
    scanf("%d %d %d", &a, &b, &r);

    int c, d;
    scanf("%d %d", &c, &d);

    int *d_a = nullptr;
    int *d_b = nullptr;
    int *d_r = nullptr;
    int *d_c = nullptr;
    int *d_d = nullptr;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_r, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));
    cudaMalloc(&d_d, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, &r, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, &d, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_a, d_b, d_r, d_c, d_d);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_c);
    cudaFree(d_d);
}
