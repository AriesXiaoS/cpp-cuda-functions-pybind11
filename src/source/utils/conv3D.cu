
#include "cuda.cuh"



__global__ void CudaConv3D(float* input, float* output, 
                           float* kernel, int kernel_size,
                           int img_size_0, int img_size_1, int img_size_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < img_size_0 && y < img_size_1 && z < img_size_2){
        // padded_size: 输入图像（padding之后的图像）的尺寸
        // int padded_size_0 = img_size_0 + kernel_size - 1;
        int padded_size_1 = img_size_1 + kernel_size - 1;
        int padded_size_2 = img_size_2 + kernel_size - 1;
        float sum = 0.0;
        int input_x, input_y, input_z;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                for (int k = 0; k < kernel_size; k++) {
                    input_x = x + i;
                    input_y = y + j;
                    input_z = z + k;
                    sum += input[input_x * padded_size_1 * padded_size_2
                                + input_y * padded_size_2 + input_z] 
                        * kernel[i * kernel_size * kernel_size 
                                + j * kernel_size + k ];                
                }
            }
        }
        output[x * img_size_1 * img_size_2 + y * img_size_2 + z] = sum;
    }
}







