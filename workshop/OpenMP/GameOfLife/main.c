#include <stdio.h>
#include <string.h>
#include <omp.h>
#define MAXN 2048

int N, M;
char board[2][MAXN][MAXN] = {0};

int main()
{
    scanf("%d%d", &N, &M);
    for (int i = 1; i <= N; i ++) {
        scanf("%s", &board[0][i][1]);
        for (int j = 1; j <= N; j ++)
            board[0][i][j] -= '0';
    }

    omp_set_num_threads(32);

    int cur = 0;
#pragma omp parallel firstprivate(cur)
    for (int t = 0; t < M; t++) {
#pragma omp for schedule(static, 64)
        for (int i = 1; i <= N; i ++) {
            for (int j = 1; j <= N; j ++) {
                int aliveNeighbors =
                    board[cur][i-1][j-1] + board[cur][i-1][j] +
                    board[cur][i-1][j+1] + board[cur][i][j-1] +
                    board[cur][i][j+1] + board[cur][i+1][j-1] +
                    board[cur][i+1][j] + board[cur][i+1][j+1];
                board[!cur][i][j] = (aliveNeighbors == 3) || (board[cur][i][j] && aliveNeighbors == 2);
            }
        }
        cur = !cur;
    }

    for (int i = 1; i <= N; i ++) {
        for (int j = 1; j <= N; j ++)
            board[N&1][i][j] += '0';
        board[N&1][i][N+1] = '\0';
        printf("%s\n", &board[N&1][i][1]);
    }

    return 0;
}