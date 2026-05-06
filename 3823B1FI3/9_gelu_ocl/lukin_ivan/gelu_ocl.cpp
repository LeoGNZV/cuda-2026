#define CL_TARGET_OPENCL_VERSION 300
#include <vector>
#include <CL/cl.h>
#include "gelu_ocl.h"

const char* gelu_source_code = R"(
__kernel void gelu_kernel(__global const float* data_in, __global float* data_out, int count) {
    int idx = get_global_id(0);
    if (idx < count) {
        float val = data_in[idx];
        float poly = 1.595769f * (val + 0.044715f * val * val * val);
        data_out[idx] = val - val / (exp(poly) + 1.0f);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) 
{
    static cl_device_id target_device;
    static cl_context ocl_context;
    static cl_command_queue cmd_queue;
    static cl_program gelu_prog;
    static cl_kernel processing_kernel;
    static bool initialized = false;

    if (!initialized) 
    {
        cl_uint plat_count;
        clGetPlatformIDs(0, nullptr, &plat_count);
        std::vector<cl_platform_id> plat_list(plat_count);
        clGetPlatformIDs(plat_count, plat_list.data(), nullptr);

        cl_platform_id selected_plat = plat_list[platform];
        clGetDeviceIDs(selected_plat, CL_DEVICE_TYPE_GPU, 1, &target_device, nullptr);
        
        ocl_context = clCreateContext(nullptr, 1, &target_device, nullptr, nullptr, nullptr);
        
        cl_queue_properties queue_settings[] = {0}; 
        cmd_queue = clCreateCommandQueueWithProperties(ocl_context, target_device, queue_settings, nullptr);

        gelu_prog = clCreateProgramWithSource(ocl_context, 1, &gelu_source_code, nullptr, nullptr);
        clBuildProgram(gelu_prog, 1, &target_device, nullptr, nullptr, nullptr);
        processing_kernel = clCreateKernel(gelu_prog, "gelu_kernel", nullptr);

        initialized = true;
    }

    const size_t element_count = input.size();
    const size_t buffer_size = element_count * sizeof(float);
    const int kernel_limit = static_cast<int>(element_count);
    const size_t global_work_offset = 0; 

    cl_mem mem_in = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, const_cast<float*>(input.data()), nullptr);
    cl_mem mem_out = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, buffer_size, nullptr, nullptr);

    clSetKernelArg(processing_kernel, 2, sizeof(int), &kernel_limit);
    clSetKernelArg(processing_kernel, 0, sizeof(cl_mem), &mem_in);
    clSetKernelArg(processing_kernel, 1, sizeof(cl_mem), &mem_out);

    size_t global_dims = element_count;
    clEnqueueNDRangeKernel(cmd_queue, processing_kernel, 1, nullptr, &global_dims, nullptr, 0, nullptr, nullptr);

    std::vector<float> output_data(element_count);
    clEnqueueReadBuffer(cmd_queue, mem_out, CL_TRUE, 0, buffer_size, output_data.data(), 0, nullptr, nullptr);

    clReleaseMemObject(mem_out);
    clReleaseMemObject(mem_in);

    return output_data;
}