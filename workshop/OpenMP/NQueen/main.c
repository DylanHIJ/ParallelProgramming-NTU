#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAXN 20

char board[MAXN][MAXN];

int isValid(int r, int c, int *positions) {
    if (board[r][c] == '*') return 0;
    for (int i = 0; i < r; i ++)
        if (positions[i] == c || (abs(c - positions[i])) == r - i)
            return 0;
    return 1;
}

int dfsQueen(int r, int N, int *positions) {
    if (r == N) return 1;

    int ans = 0;
    for (int i = 0; i < N; i ++)
        if (isValid(r, i, positions)) {
            positions[r] = i;
            ans += dfsQueen(r + 1, N, positions);
        }
    return ans;
}
int main() {
    int count = 0, N;
    while (scanf("%d", &N) == 1) {
        for (int i = 0; i < N; i ++)
            scanf("%s", &board[i]);

        int positions[N];
        int ans = 0;
#pragma omp parallel
        {
#pragma omp for
        for (int i = 0; i < N; i ++)
            positions[i] = 0;


#pragma omp for private(positions) schedule(dynamic, 16) reduction(+:ans) collapse(4)
        for (int i = 0; i < N; i ++)
            for (int j = 0; j < N; j ++)
                for (int k = 0; k < N; k ++)
                    for (int l = 0; l < N; l ++) {
                        positions[0] = i;
                        positions[1] = j;
                        positions[2] = k;
                        positions[3] = l;
                        if (isValid(0, i, positions) && isValid(1, j, positions) &&
                            isValid(2, k, positions) && isValid(3, l, positions))
                            ans += dfsQueen(4, N, positions);
                    }
        }
        count ++;
        printf("Case %d: %d\n", count, ans);
    }
}