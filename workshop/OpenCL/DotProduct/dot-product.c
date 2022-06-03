#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <CL/cl.h>
#include "utils.h"

#define MAXK 1024
#define MAXN 16777216
#define MAXLOG 1024
#define LOCALSIZE 1024

int main(int argc, char *argv[]) {
    cl_int status;

    /* platform */
    cl_platform_id platform_id;
    status = clGetPlatformIDs(1, &platform_id, NULL);
    assert(status == CL_SUCCESS);

    /* devices */
    cl_device_id GPU;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &GPU, NULL);
    assert(status == CL_SUCCESS);

    /* context */
    cl_context context = clCreateContext(NULL, 1, &GPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    /* command queue */
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
        context, GPU, NULL, &status
    );
    assert(status == CL_SUCCESS);

    /* kernel source */
    FILE *kernelfp = fopen("vecdot.cl", "r");
    assert(kernelfp != NULL);
    char kernelBuffer[MAXK];
    const char *constKernelSource = kernelBuffer;
    size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
    cl_program program = clCreateProgramWithSource(
        context, 1, &constKernelSource, &kernelLength, &status
    );
    assert(status == CL_SUCCESS);
    fclose(kernelfp);

    /* build program */
    status = clBuildProgram(program, 1, GPU, NULL, NULL, NULL);
    assert(status == CL_SUCCESS);


    /* create kernel */
    cl_kernel kernel = clCreateKernel(program, "vecDot", &status);
    assert(status == CL_SUCCESS);

    /* create vector */
    uint32_t *vec = (uint32_t *)malloc(MAXN * sizeof(uint32_t) / LOCALSIZE);
    assert(vec != NULL);

    // /* create buffer */
    cl_mem bufferVec = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, MAXN * sizeof(uint32_t) / LOCALSIZE, vec, &status);
    assert(status == CL_SUCCESS);


    int N;
    uint32_t key1, key2;
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        uint32_t padding = 0;
        while (N % LOCALSIZE != 0) {
            padding += encrypt(N, key1) * encrypt(N, key2);
            N ++;
        }

        /* set arguments */
        status = clSetKernelArg(kernel, 0, sizeof(uint32_t), (void *)&key1);
        assert(status == CL_SUCCESS);
        status = clSetKernelArg(kernel, 1, sizeof(uint32_t), (void *)&key2);
        assert(status == CL_SUCCESS);
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferVec);
        assert(status == CL_SUCCESS);

        /* kernel calculation */
        size_t globalOffsets[] = {0};
        size_t globalWorkSizes[] = {N};
        size_t localWorkSizes[] = {LOCALSIZE};
        status = clEnqueueNDRangeKernel(
            commandQueue, kernel, 1, globalOffsets, globalWorkSizes, localWorkSizes, 0, NULL, NULL
        );
        assert(status == CL_SUCCESS);

        /* read result from device */
        status = clEnqueueReadBuffer(commandQueue, bufferVec, CL_TRUE, 0, N * sizeof(cl_uint) / LOCALSIZE, vec, 0, NULL, NULL);
        assert(status == CL_SUCCESS);

        uint32_t sum = 0;
        for (int i = 0; i < N / LOCALSIZE; i ++)
            sum += vec[i];

        printf("%" PRIu32 "\n", sum - padding);
    }

    /* free and release */
    clReleaseContext(context);
    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(bufferVec);
    free(vec);

    return 0;
}d n