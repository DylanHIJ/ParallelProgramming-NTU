#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define MAXN 1000000

int nums[MAXN];
int res[49];
int N; 

typedef struct range {
    int *res;
    int *start, *end;
} Range;

void *count(void *params) {
    Range *r = (Range *)params;
    int *start = r->start, *end = r->end;
    int *res = r->res;
    while (start != end) {
        *res += *start;
        start ++;
    }
    pthread_exit(NULL);
}

int count_parallel(int *p1, int *p2, int *p3, int *p4) {
    pthread_t threads[4];

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    Range *r;
    r = (Range *)calloc(1, sizeof(Range));
    r->res = p1;
    r->start = &nums[0];
    r->end = &nums[N/4];
    pthread_create(&threads[0], &attr, count, (void *)r);


    r = (Range *)calloc(1, sizeof(Range));
    r->res = p2;
    r->start = &nums[N/4];
    r->end = &nums[N/2];
    pthread_create(&threads[1], &attr, count, (void *)r);


    r = (Range *)calloc(1, sizeof(Range));
    r->res = p3;
    r->start = &nums[N/2];
    r->end = &nums[N/4*3];
    pthread_create(&threads[2], &attr, count, (void *)r);

    r = (Range *)calloc(1, sizeof(Range));
    r->res = p4;
    r->start = &nums[N/4*3];
    r->end = &nums[N]; 
    pthread_create(&threads[3], &attr, count, (void *)r);


    for (int i = 0; i < 4; i ++) 
        pthread_join(threads[i], NULL);
    return *p1 + *p2 + *p3 + *p4;
}
int main(int argc, char *argv[])
{
    N = atoi(argv[1]);
    for (int i = 0; i < N; i ++)
        nums[i] = 1;

    int result = count_parallel(&res[0], &res[16], &res[32], &res[48]);
    printf("%d\n", result);
    return 0;
}
