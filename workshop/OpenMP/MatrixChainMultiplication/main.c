#include <stdio.h>
#include <omp.h>
#define MAXN 2048
#define INF (1LL<<60)
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

int n;
long long dp[MAXN*MAXN], sizes[MAXN+1];

int main() {
    while (scanf("%d", &n) == 1) {
        for (int i = 0; i <= n; i++)
            scanf("%lld", &sizes[i]);
        for (int i = 0; i < n; i++)
            dp[i*n+i] = 0;

        omp_set_num_threads(16);

#pragma omp parallel
        for (int len = 1; len < n; len ++) {
#pragma omp for
            for (int left = 0; left < n - len; left ++) {
                int right = left + len;
                long long localMin = INF, product = sizes[left] * sizes[right+1];
                for (int cut = left; cut < right; cut ++)
                    localMin = MIN(localMin, dp[left*n+cut] + dp[right*n+(cut+1)] + product * sizes[cut+1]);
                dp[left*n+right] = dp[right*n+left]= localMin;
            }
        }
        printf("%lld\n", dp[0*n+n-1]);
    }
    return 0;
}