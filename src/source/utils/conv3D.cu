
#include "define.h"
#include "cuda.cuh"


__global__ void CudaConv3D(float* input, float* output, 
                           float* kernel, Conv3DParam param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < param.img_shape[0] && y < param.img_shape[1] && z < param.img_shape[2]){
        // padded_size: 输入图像（padding之后的图像）的尺寸
        // int padded_size_0 = param.img_shape[0] + param.kernel_size - 1;
        int padded_size_1 = param.img_shape[1] + param.kernel_size - 1;
        int padded_size_2 = param.img_shape[2] + param.kernel_size - 1;
        float sum = 0.0;
        int input_x, input_y, input_z;
        for (int i = 0; i < param.kernel_size; i++) {
            for (int j = 0; j < param.kernel_size; j++) {
                for (int k = 0; k < param.kernel_size; k++) {
                    input_x = x + i;
                    input_y = y + j;
                    input_z = z + k;
                    sum += input[input_x * padded_size_1 * padded_size_2
                                + input_y * padded_size_2 + input_z] 
                        * kernel[i * param.kernel_size * param.kernel_size 
                                + j * param.kernel_size + k ];                
                }
            }
        }
        output[x * param.img_shape[1] * param.img_shape[2] + y * param.img_shape[2] + z] = sum;
    }
}







