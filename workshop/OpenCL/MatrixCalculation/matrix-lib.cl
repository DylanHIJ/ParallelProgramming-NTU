#define uint32_t unsigned int
#define MAXN 1024
#define BLOCKSIZE 16

__kernel void multiply(__global uint32_t A[MAXN][MAXN], __global uint32_t B[MAXN][MAXN], __global uint32_t C[MAXN][MAXN])
{
    __local int localA[BLOCKSIZE][BLOCKSIZE];
    __local int localB[BLOCKSIZE][BLOCKSIZE];
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    int sum = 0;
    for (int block = 0; block < MAXN / BLOCKSIZE; block++) {
        localA[localRow][localCol] =
            A[globalRow][block * BLOCKSIZE + localCol];
        localB[localRow][localCol] =
            B[block * BLOCKSIZE + localRow][globalCol];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BLOCKSIZE; k++)
            sum += localA[localRow][k] * localB[k][localCol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalRow][globalCol] = sum;
}

__kernel void add(__global uint32_t A[MAXN][MAXN], __global uint32_t B[MAXN][MAXN], __global uint32_t C[MAXN][MAXN]) {
    int row = get_global_id(0), col = get_global_id(1);
    C[row][col] = A[row][col] + B[row][col];
}