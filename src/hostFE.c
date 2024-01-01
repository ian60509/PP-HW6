#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#include <cuda.h>

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, 
            cl_device_id *device, cl_context *context, cl_program *program) {

    int filt_size = filterWidth * filterWidth * sizeof(float);
    int data_size = imageHeight *  imageWidth * sizeof(float);
    int half_filter = filterWidth / 2;

    cl_int status;
    cl_command_queue first_queue = clCreateCommandQueue(*context, *device, 0, &status);;
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);;
    cl_mem input_mem = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, data_size, inputImage, &status);
    cl_mem ouput_mem = clCreateBuffer(*context,   CL_MEM_WRITE_ONLY, data_size, NULL, &status);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filt_size,  filter, &status);




    // -------------------set Argument-----------------------------------
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ouput_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_mem);
    clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 5, sizeof(int), &half_filter);


    //---------------------------- set workgroup sizes ----------------------
    // size_t global_size = data_size / (4*sizeof(float));
    size_t global_size = imageHeight * imageWidth / 4;
    size_t local_size = 64;
    size_t global_size_2D[2] = { global_size, 1 };
    clEnqueueNDRangeKernel(first_queue, kernel, 2, NULL,  global_size_2D, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(first_queue, ouput_mem, CL_TRUE, 0, data_size, outputImage, 0, NULL, NULL);
}