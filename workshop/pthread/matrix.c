#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

 
typedef struct matmul {
    unsigned long (*A)[2048], (*B)[2048], (*C)[2048];
    int N;
    int rowIdx;
} MatMul;

void *matrix_multiply(void *p) {
    MatMul *params = (MatMul *)p;
    int N = params->N, i = params->rowIdx;
    for (int j = 0; j < N; j++) {
        unsigned long sum = 0; // overflow, let it go.
        for (int k = 0; k < N; k++)
            sum += params->A[i][k] * params->B[j][k];
        params->C[i][j] = sum;
    }
    // free(params);
    pthread_exit(NULL);
}

void multiply(int N, unsigned long A[][2048], unsigned long B[][2048], unsigned long C[][2048])
{
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
        {
            unsigned long tmp = B[i][j];
            B[i][j] = B[j][i];
            B[j][i] = tmp;
        }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t threads[6];
    MatMul *params;
    for (int i = 0; i < N; i ++)
    {
        if (i >= 6)
            pthread_join(threads[i%6], NULL);
        
        params = (MatMul *)calloc(1, sizeof(MatMul));

        params->N = N;
        params->rowIdx = i;
        params->A = A;
        params->B = B;
        params->C = C;
        pthread_create(&threads[i%6], &attr, matrix_multiply, (void *)params);
    }

    pthread_attr_destroy(&attr);
    for (int i = 0; i < (N < 6 ? N : 6); i ++)
        pthread_join(threads[i], NULL);
}