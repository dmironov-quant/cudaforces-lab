// https://cudaforces.com/problem/35

__global__ void kernel(int *first, int *second, int *ans) {
    if (threadIdx.x == 0) {
        *ans = (*first < *second) ? 1 : 0;
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int *first, *second, *ans;
    cudaMalloc(&first, sizeof(int));
    cudaMalloc(&second, sizeof(int));
    cudaMalloc(&ans, sizeof(int));

    cudaMemcpy(first, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(second, &b, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(first, second, ans);
    cudaDeviceSynchronize();

    int res;
    cudaMemcpy(&res, ans, cudaMemcpyDeviceToHost);
    printf("%d", res);

    cudaFree(first);
    cudaFree(second);
    cudaFree(ans);
}
