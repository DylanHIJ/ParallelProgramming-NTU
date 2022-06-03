#define MAXN 2048

__kernel void gameOfLife(__global char board[2][MAXN][MAXN], int round, int n) {
    int i = get_global_id(0), j = get_global_id(1);
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
