// https://cudaforces.com/problem/135

__global__ void kernel(int *dn, int *da, int *db, int *dc, int *dans) {
    if (threadIdx.x == 0) {
        int x = (*dn) / 3;
        x *= (*da);
        x /= 5;
        x *= (*db);
        x /= 3;
        x *= (*dc);
        *dans = x;  
    }
}

int main() {
    int n, a, b, c;
    scanf("%d %d %d %d", &n, &a, &b, &c);

    int *dn = nullptr, *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaMalloc(&dn, sizeof(int));
    cudaMalloc(&da, sizeof(int));
    cudaMalloc(&db, sizeof(int));
    cudaMalloc(&dc, sizeof(int));

    cudaMemcpy(dn, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, &c, sizeof(int), cudaMemcpyHostToDevice);

    int *dans = nullptr;
    cudaMalloc(&dans, sizeof(int));
    int init = 0;
    cudaMemcpy(dans, &init, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(dn, da, db, dc, dans);
    cudaDeviceSynchronize();

    int ans;
    cudaMemcpy(&ans, dans, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", ans);

    cudaFree(dn);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
