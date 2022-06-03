#include <stdio.h>
#include <limits.h>
#include <omp.h>
#define MAXLEN 500

int square(int x) {
    return x * x;
}

int main() {
    int aHeight, aWidth, bHeight, bWidth;
    int A[MAXLEN][MAXLEN], B[MAXLEN][MAXLEN], diffs[MAXLEN][MAXLEN];
    while (scanf("%d%d%d%d", &aHeight, &aWidth, &bHeight, &bWidth) == 4) {
        for (int i = 0; i < aHeight; i ++)
            for (int j = 0; j < aWidth; j ++)
                scanf("%d", &A[i][j]);
        for (int i = 0; i < bHeight; i ++)
            for (int j = 0; j < bWidth; j ++)
                scanf("%d", &B[i][j]);

        #pragma omp parallel for
        for (int x = 0; x <= aHeight - bHeight; x ++) {
            for (int y = 0; y <= aWidth - bWidth; y ++) {
                int diff = 0;
                for (int i = 0; i < bHeight; i ++)
                    for (int j = 0; j < bWidth; j ++)
                        diff += square(A[x+i][y+j] - B[i][j]);
                diffs[x][y] = diff;
            }
        }

        int minDiff = INT_MAX, minX = -1, minY = -1;
        for (int x = 0; x <= aHeight - bHeight; x ++)
            for (int y = 0; y <= aWidth - bWidth; y ++)
                if (diffs[x][y] < minDiff) {
                    minDiff = diffs[x][y];
                    minX = x + 1;
                    minY = y + 1;
                }

        printf("%d %d\n", minX, minY);
    }

    return 0;
}