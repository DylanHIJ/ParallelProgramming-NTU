#include <stdio.h>
#include <omp.h>

#define BOX_IDX(r, c) (((r) / 3) * 3 + (c) / 3);

int n = 0;
int zeros[81];

int getBit(int rows[9], int cols[9], int boxes[9], int r, int c, int b, int bit) {
    return ((rows[r] >> bit) & 1) || ((cols[c] >> bit) & 1) || ((boxes[b] >> bit) & 1);
}

void setBit(int rows[9], int cols[9], int boxes[9], int r, int c, int bit) {
    int boxIdx = BOX_IDX(r, c);
    rows[r] |= (1 << bit);
    cols[c] |= (1 << bit);
    boxes[boxIdx] |= (1 << bit);
}

void reverseBit(int rows[9], int cols[9], int boxes[9], int r, int c, int bit) {
    int boxIdx = BOX_IDX(r, c);
    rows[r] ^= (1 << bit);
    cols[c] ^= (1 << bit);
    boxes[boxIdx] ^= (1 << bit);
}

int solve(int i, int rows[9], int cols[9], int boxes[9]) {
    if (i == -1) return 1;
    int r = zeros[i] / 9, c = zeros[i] % 9, b = BOX_IDX(r, c);
    int numSolution = 0;
    for (int try = 1; try <= 9; try ++) {
        if (getBit(rows, cols, boxes, r, c, b, try))
            continue;
        setBit(rows, cols, boxes, r, c, try);
        numSolution += solve(i - 1, rows, cols, boxes);
        reverseBit(rows, cols, boxes, r, c, try);
    }
    return numSolution;
}

int main() {
    int board[9][9], rows[9] = {0}, cols[9] = {0}, boxes[9] = {0};
    for (int i = 0; i < 9; i ++)
        for (int j = 0; j < 9; j ++) {
            scanf("%d", &board[i][j]);
            if (board[i][j] == 0)
                zeros[n++] = i * 9 + j;
            else
                setBit(rows, cols, boxes, i, j, board[i][j]);
        }
    if (n == 0) return 1;

    omp_set_num_threads(64);

    int numSolution = 0;
    #pragma omp parallel for reduction(+: numSolution) firstprivate(rows, cols, boxes) collapse(4)
    for (int i = 1; i <= 9; i ++)
        for (int j = 1; j <= 9; j ++)
            for (int k = 1; k <= 9; k ++)
                for (int l = 1; l <= 9; l ++) {
                    // try first zero
                    int r = zeros[n-1] / 9, c = zeros[n-1] % 9, b = BOX_IDX(r, c);
                    if (getBit(rows, cols, boxes, r, c, b, i))
                        continue;
                    if (n == 1) {
                        numSolution += 1;
                        continue;
                    }
                    setBit(rows, cols, boxes, r, c, i);

                    // try second zero
                    r = zeros[n-2] / 9, c = zeros[n-2] % 9, b = BOX_IDX(r, c);
                    if (getBit(rows, cols, boxes, r, c, b, j)) {
                        reverseBit(rows, cols, boxes, zeros[n-1] / 9, zeros[n-1] % 9, i);
                        continue;
                    }
                    if (n == 2) {
                        numSolution += 1;
                        continue;
                    }
                    setBit(rows, cols, boxes, r, c, j);


                    // try third zero
                    r = zeros[n-3] / 9, c = zeros[n-3] % 9, b = BOX_IDX(r, c);
                    if (getBit(rows, cols, boxes, r, c, b, k)) {
                        reverseBit(rows, cols, boxes, zeros[n-1] / 9, zeros[n-1] % 9, i);
                        reverseBit(rows, cols, boxes, zeros[n-2] / 9, zeros[n-2] % 9, j);
                        continue;
                    }
                    if (n == 3) {
                        numSolution += 1;
                        continue;
                    }
                    setBit(rows, cols, boxes, r, c, k);


                    r = zeros[n-4] / 9, c = zeros[n-4] % 9, b = BOX_IDX(r, c);
                    if (getBit(rows, cols, boxes, r, c, b, l)) {
                        reverseBit(rows, cols, boxes, zeros[n-1] / 9, zeros[n-1] % 9, i);
                        reverseBit(rows, cols, boxes, zeros[n-2] / 9, zeros[n-2] % 9, j);
                        reverseBit(rows, cols, boxes, zeros[n-3] / 9, zeros[n-3] % 9, k);
                        continue;
                    }
                    setBit(rows, cols, boxes, r, c, l);

                    numSolution += solve(n - 5, rows, cols, boxes);

                    reverseBit(rows, cols, boxes, zeros[n-1] / 9, zeros[n-1] % 9, i);
                    reverseBit(rows, cols, boxes, zeros[n-2] / 9, zeros[n-2] % 9, j);
                    reverseBit(rows, cols, boxes, zeros[n-3] / 9, zeros[n-3] % 9, k);
                    reverseBit(rows, cols, boxes, zeros[n-4] / 9, zeros[n-4] % 9, l);
                }

    printf("%d\n", numSolution);
    return 0;
}