#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#define MAXGPU 10
#define MAXK 1024
#define MAXLOG (1024 * 1024)
#define MAXFILENAME 128

int main(int argc, char *argv[]) {
    char filename[MAXFILENAME];
    scanf("%s", filename);

    cl_int status;
    cl_platform_id platform_id;
    cl_uint platform_id_got;
    status = clGetPlatformIDs(1, &platform_id, &platform_id_got);
    assert(status == CL_SUCCESS && platform_id_got == 1);

    cl_device_id GPU[MAXGPU];
    cl_uint GPU_id_got;
    status = clGetDeviceIDs(
        platform_id, CL_DEVICE_TYPE_GPU, MAXGPU, GPU, &GPU_id_got
    );
    assert(status == CL_SUCCESS);

    cl_context context = clCreateContext(
        NULL, GPU_id_got, GPU, NULL, NULL, &status
    );
    assert(status == CL_SUCCESS);

    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
        context, GPU[0], NULL, &status
    );
    assert(status == CL_SUCCESS);

    FILE *kernelfp = fopen(filename, "r");
    assert(kernelfp != NULL);
    char kernelBuffer[MAXK];
    const char *constKernelSource = kernelBuffer;
    size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);

    cl_program program = clCreateProgramWithSource(
        context, 1, &constKernelSource, &kernelLength, &status
    );
    assert(status == CL_SUCCESS);

    status = clBuildProgram(program, GPU_id_got, GPU, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        size_t logLength;
        char buildLog[MAXLOG];
        status = clGetProgramBuildInfo(program, GPU[0], CL_PROGRAM_BUILD_LOG, MAXLOG, (void *)buildLog, &logLength);
        assert(status == CL_SUCCESS);

        buildLog[logLength] = '\0';
        printf("%s", buildLog);
    }

    clReleaseContext(context);
    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    return 0;
}