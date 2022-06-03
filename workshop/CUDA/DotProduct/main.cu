#include <stdio.h>
#include <stdint.h>
#include <cuda.h>

#define MAXN 16777216
#define LOCALSIZE 1024

__host__ __device__ uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}

__host__ __device__ uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}

__global__ void vecdot(uint32_t key1, uint32_t key2, uint32_t vec[]) {
    __shared__ uint32_t buf[LOCALSIZE];

    int globalID = blockIdx.x * blockDim.x + threadIdx.x,
        groupID = blockIdx.x,
        localID = threadIdx.x,
        localSize = blockDim.x;

    buf[localID] = encrypt(globalID, key1) * encrypt(globalID, key2);
    __syncthreads();

    for (int i = localSize / 2; i > 0; i /= 2) {
        if (localID < i)
            buf[localID] += buf[localID+i];
        __syncthreads();
    }

    if (localID == 0)
        vec[groupID] = buf[0];
}

uint32_t vec[MAXN / LOCALSIZE];

int main() {
    int N;
    uint32_t key1, key2;
    uint32_t *vecCuda;
    cudaMalloc((void **)&vecCuda, sizeof(uint32_t) * MAXN / LOCALSIZE);

    while (scanf("%d %u %u", &N, &key1, &key2) == 3) {
        uint32_t padding = 0;
        while (N % LOCALSIZE != 0) {
            padding += encrypt(N, key1) * encrypt(N, key2);
            N ++;
        }

        vecdot<<<N / LOCALSIZE, LOCALSIZE>>>(key1, key2, vecCuda);
        cudaDeviceSynchronize();

        cudaMemcpy(vec, vecCuda, sizeof(uint32_t) * N / LOCALSIZE, cudaMemcpyDeviceToHost);

        uint32_t sum = 0;
        for (int i = 0; i < N / LOCALSIZE; i ++)
            sum += vec[i];
        printf("%u\n", sum - padding);
    }
    cudaFree(vecCuda);
    return 0;
}