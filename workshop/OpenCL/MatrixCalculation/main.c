#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <CL/cl.h>

#define UINT uint32_t
#define MAXK (4096)
#define MAXN 1024
#define MAXLOG (1024 * 1024)
#define BLOCKSIZE 16

cl_int status;
cl_platform_id platform_id;
cl_device_id GPU;
cl_context context;
cl_command_queue cmdQueue;
cl_program program;
cl_kernel mulKernel, addKernel;
cl_mem arrBuf[6], resultBuf[6];

UINT arr[6][MAXN][MAXN] = {0}, result[6][MAXN][MAXN] = {0};

void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i][j] = x;
        }
    }
}

UINT signature(int N, UINT A[][MAXN]) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i][j]) * 2654435761LU;
    }
    return h;
}

void initCL() {
    /* platform */
    status = clGetPlatformIDs(1, &platform_id, NULL);
    assert(status == CL_SUCCESS);

    /* devices */
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &GPU, NULL);
    assert(status == CL_SUCCESS);

    /* context */
    context = clCreateContext(NULL, 1, &GPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    /* command queue */
    cmdQueue = clCreateCommandQueueWithProperties(context, GPU, NULL, &status);
    assert(status == CL_SUCCESS);

    /* kernel source */
    FILE *kernelfp = fopen("matrix-lib.cl", "r");
    assert(kernelfp != NULL);
    char kernelBuffer[MAXK];
    const char *bufPtr = kernelBuffer;
    size_t kernelLen= fread(kernelBuffer, 1, MAXK, kernelfp);
    program = clCreateProgramWithSource(context, 1, &bufPtr, &kernelLen, &status);
    assert(status == CL_SUCCESS);
    fclose(kernelfp);

    /* build program */
    status = clBuildProgram(program, 1, &GPU, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        char buildLog[MAXLOG];
        status = clGetProgramBuildInfo(program, GPU, CL_PROGRAM_BUILD_LOG, MAXLOG, (void *)buildLog, NULL);
        printf("%s\n", buildLog);
        exit(0);
    }

    /* create kernel */
    mulKernel = clCreateKernel(program, "multiply", &status);
    assert(status == CL_SUCCESS);
    addKernel = clCreateKernel(program, "add", &status);
    assert(status == CL_SUCCESS);
}

void cleanUpCL() {
    if (context) clReleaseContext(context);
    if (cmdQueue) clReleaseCommandQueue(cmdQueue);
    if (program) clReleaseProgram(program);
    if (mulKernel) clReleaseKernel(mulKernel);
    if (addKernel) clReleaseKernel(addKernel);
}

void multiplyCL(int N, UINT A[][MAXN], UINT B[][MAXN], UINT C[][MAXN]) {
    size_t globalWorkSizes[] = {MAXN, MAXN};
    size_t localWorkSizes[] = {BLOCKSIZE, BLOCKSIZE};

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(UINT) * MAXN * MAXN, A, &status);
    assert(status == CL_SUCCESS);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(UINT) * MAXN * MAXN, B, &status);
    assert(status == CL_SUCCESS);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, C, &status);
    assert(status == CL_SUCCESS);

    status = clSetKernelArg(mulKernel, 0, sizeof(cl_mem), (void *)&bufferA);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(mulKernel, 1, sizeof(cl_mem), (void *)&bufferB);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(mulKernel, 2, sizeof(cl_mem), (void *)&bufferC);
    assert(status == CL_SUCCESS);

    status = clEnqueueNDRangeKernel(
        cmdQueue, mulKernel, 2, NULL, globalWorkSizes, localWorkSizes, 0, NULL, NULL
    );
    assert(status == CL_SUCCESS);
    clFinish(cmdQueue);

    status = clEnqueueReadBuffer(
        cmdQueue, bufferC, CL_TRUE, 0,
        sizeof(UINT) * MAXN * MAXN, C, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
}

void addCL(int N, UINT A[][MAXN], UINT B[][MAXN], UINT C[][MAXN]) {
    size_t globalWorkSizes[] = {MAXN, MAXN};
    size_t localWorkSizes[] = {BLOCKSIZE, BLOCKSIZE};

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(UINT) * MAXN * MAXN, A, &status);
    assert(status == CL_SUCCESS);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(UINT) * MAXN * MAXN, B, &status);
    assert(status == CL_SUCCESS);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(UINT) * MAXN * MAXN, C, &status);
    assert(status == CL_SUCCESS);

    status = clSetKernelArg(addKernel, 0, sizeof(cl_mem), (void *)&bufferA);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(addKernel, 1, sizeof(cl_mem), (void *)&bufferB);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(addKernel, 2, sizeof(cl_mem), (void *)&bufferC);
    assert(status == CL_SUCCESS);

    status = clEnqueueNDRangeKernel(
        cmdQueue, addKernel, 2, NULL, globalWorkSizes, localWorkSizes, 0, NULL, NULL
    );
    assert(status == CL_SUCCESS);
    clFinish(cmdQueue);

    status = clEnqueueReadBuffer(
        cmdQueue, bufferC, CL_TRUE, 0,
        sizeof(UINT) * MAXN * MAXN, C, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
}

void print_matrix(int N, UINT A[][MAXN]) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i][j]);
        fprintf(stderr, " ]\n");
    }
}

int main() {
    int N, seed;
    scanf("%d", &N);
    for (int i = 0; i < 6; i++) {
        scanf("%d", &seed);
        rand_gen(seed, N, arr[i]);
    }
    initCL();

    multiplyCL(N, arr[0], arr[1], result[0]);
    multiplyCL(N, arr[2], arr[3], result[1]);
    addCL(N, result[0], result[1], result[2]);
    printf("%u\n", signature(N, result[2]));

    multiplyCL(N, result[0], arr[4], result[3]);
    multiplyCL(N, result[1], arr[5], result[4]);
    addCL(N, result[3], result[4], result[5]);
    printf("%u\n", signature(N, result[5]));

    cleanUpCL();

    return 0;
}