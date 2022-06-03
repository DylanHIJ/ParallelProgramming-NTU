#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <CL/cl.h>

#define UINT uint32_t
#define MAXK 4096
#define MAXN 2048
#define MAXM 5000
#define MAXLOG (1024 * 1024)
#define LOCAL_SIZE 16

cl_int status;
cl_device_id GPU;
cl_context context;
cl_command_queue cmdQueue;
cl_program program;
cl_kernel kernel;
cl_mem bufferBoard;

char board[2][MAXN][MAXN] = {0};

void initCL() {
    /* platform */
    cl_platform_id platform_id;
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
    FILE *kernelfp = fopen("game-of-life.cl", "r");
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
    kernel = clCreateKernel(program, "gameOfLife", &status);
    assert(status == CL_SUCCESS);
}


void cleanUpCL() {
    if (context) clReleaseContext(context);
    if (cmdQueue) clReleaseCommandQueue(cmdQueue);
    if (program) clReleaseProgram(program);
}

int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++) {
        scanf("%s", &board[0][i][1]);
        for (int j = 1; j <= n; j ++)
            board[0][i][j] -= '0';
    }

    initCL();

    size_t globalWorkSizes[] = {MAXN, MAXN};
    size_t localWorkSizes[] = {LOCAL_SIZE, LOCAL_SIZE};

    bufferBoard = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(char) * MAXN * MAXN * 2, board, &status
    );
    assert(status == CL_SUCCESS);

    for (int i = 0; i < m; i ++) {
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferBoard);
        assert(status == CL_SUCCESS);
        status = clSetKernelArg(kernel, 1, sizeof(int), (void *)&i);
        assert(status == CL_SUCCESS);
        status = clSetKernelArg(kernel, 2, sizeof(int), (void *)&n);
        assert(status == CL_SUCCESS);

        status = clEnqueueNDRangeKernel(
            cmdQueue, kernel, 2, NULL, globalWorkSizes, localWorkSizes, 0, NULL, NULL);
        assert(status == CL_SUCCESS);
        clFinish(cmdQueue);
    }

    status = clEnqueueReadBuffer(
        cmdQueue, bufferBoard, CL_TRUE, 0,
        sizeof(char) * MAXN * MAXN * 2, board, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= n; j ++)
            board[m&1][i][j] += '0';
        board[m&1][i][n+1] = '\0';
        printf("%s\n", &board[m&1][i][1]);
    }
    cleanUpCL();
    return 0;
}