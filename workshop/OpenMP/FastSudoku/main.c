#include <stdio.h>
#include <omp.h>


int zeros[81];
int n = 0;

int rowColConflict(int try, int row, int col, int sudoku[9][9])
{
  int conflict = 0;
  for (int i = 0; i < 9 && !conflict; i++)
    if (((col != i) && (sudoku[row][i] == try)) ||
	((row != i) && (sudoku[i][col] == try)))
      conflict = 1;
  return conflict;
}


int blockConflict(int try, int row, int col, int sudoku[9][9])
{
  int blockRow = row / 3;
  int blockCol = col / 3;

  int conflict = 0;
  for (int i = 0; i < 3 && !conflict; i ++)
    for (int j = 0; j < 3 && !conflict; j ++)
      if (sudoku[3 * blockRow + i][3 * blockCol + j] == try)
	conflict = 1;
  return conflict;
}

int conflict(int try, int row, int col, int board[9][9])
{
  return (rowColConflict(try, row, col, board) ||
	  blockConflict(try, row, col, board));
}

int solve(int i, int board[9][9]) {
    if (i >= n) return 1;
    int row = zeros[i] / 9, col = zeros[i] % 9;

    int numSolution = 0;
    for (int try = 1; try <= 9; try ++) {
        if (!conflict(try, row, col, board)) {
            board[row][col] = try;
            numSolution += solve(i + 1, board);
        }
    }
    board[row][col] = 0;
    return numSolution;
}

int main() {
    int board[9][9];
    for (int i = 0; i < 9; i ++)
        for (int j = 0; j < 9; j ++) {
            scanf("%d", &board[i][j]);
            if (board[i][j] == 0)
                zeros[n++] = i * 9 + j;
        }

    omp_set_num_threads(32);

    int numSolution = 0;
#pragma parallel for reduction(+: numSolution) firstprivate(board) collapse(4)
    for (int i = 1; i <= 9; i ++)
        for (int j = 1; j <= 9; j ++)
            for (int k = 1; k <= 9; k ++)
                for (int l = 1; l <= 9; l ++) {
                    if (conflict(i, zeros[0]/9, zeros[0]%9, board))
                        continue;
                    board[zeros[0]/9][zeros[0]%9] = i;
                    if (n == 1) {
                        numSolution += solve(1, board);
                        continue;
                    }


                    if (conflict(j, zeros[1]/9, zeros[1]%9, board)) {
                        board[zeros[0]/9][zeros[0]%9] = 0;
                        continue;
                    }
                    board[zeros[1]/9][zeros[1]%9] = j;
                    if (n == 2) {
                        numSolution += solve(2, board);
                        continue;
                    }

                    if (conflict(k, zeros[2]/9, zeros[2]%9, board)) {
                        board[zeros[0]/9][zeros[0]%9] = 0;
                        board[zeros[1]/9][zeros[1]%9] = 0;
                        continue;
                    }
                    board[zeros[2]/9][zeros[2]%9] = k;
                    if (n == 3) {
                        numSolution += solve(3, board);
                        continue;
                    }

                    if (conflict(l, zeros[3]/9, zeros[3]%9, board)) {
                        board[zeros[0]/9][zeros[0]%9] = 0;
                        board[zeros[1]/9][zeros[1]%9] = 0;
                        board[zeros[2]/9][zeros[2]%9] = 0;
                        continue;
                    }
                    board[zeros[3]/9][zeros[3]%9] = l;
                    numSolution += solve(4, board);

                    board[zeros[0]/9][zeros[0]%9] = 0;
                    board[zeros[1]/9][zeros[1]%9] = 0;
                    board[zeros[2]/9][zeros[2]%9] = 0;
                    board[zeros[3]/9][zeros[3]%9] = 0;
                }
    printf("%d\n", numSolution);
    return 0;
}