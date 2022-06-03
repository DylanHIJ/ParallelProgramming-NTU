#include <stdio.h>
#include <cuda.h>

#define MAXN 2048
#define MAXM 5000
#define BOARDSIZE (MAXN * MAXN * 2)
#define LOCALSIZE 16

char board[2][MAXN][MAXN] = {0};

__global__ void gameOfLife(char board[2][MAXN][MAXN], int round, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i > n || j < 1 || j > n)
        return;

    int cur = round & 1;
    int aliveNeighbors =
        board[cur][i-1][j-1] + board[cur][i-1][j] +
        board[cur][i-1][j+1] + board[cur][i][j-1] +
        board[cur][i][j+1] + board[cur][i+1][j-1] +
        board[cur][i+1][j] + board[cur][i+1][j+1];
    board[1-cur][i][j] = (aliveNeighbors == 3) || (board[cur][i][j] && aliveNeighbors == 2);
}

int main() {
    // input
    int n, m;
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++) {
        scanf("%s", &board[0][i][1]);
        for (int j = 1; j <= n; j ++)
            board[0][i][j] -= '0';
    }

    char (*boardCuda)[MAXN][MAXN];
    cudaMalloc((void **)&boardCuda, sizeof(char) * BOARDSIZE);
    cudaMemcpy(boardCuda, board, sizeof(char) * BOARDSIZE, cudaMemcpyHostToDevice);

    dim3 gridDims(MAXN / LOCALSIZE, MAXN / LOCALSIZE);
    dim3 blockDims(LOCALSIZE, LOCALSIZE);
    for (int i = 0; i < m; i ++) {
        gameOfLife<<<gridDims, blockDims>>>(boardCuda, i, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(board, boardCuda, sizeof(char) * BOARDSIZE, cudaMemcpyDeviceToHost);

    // output
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= n; j ++)
            board[m&1][i][j] += '0';
        board[m&1][i][n+1] = '\0';
        printf("%s\n", &board[m&1][i][1]);
    }

    return 0;
}