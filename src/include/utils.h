#ifndef _UTILS_H
#define _UTILS_H

#include "define.h"

/**
* @brief Padding 三维数组 但在一维层面
*/
void PaddingFlattenedArr_3D(float* arr, float* result, 
        int size_0, int size_1, int size_2,
        float pad_value, int pad_size_0, int pad_size_1, int pad_size_2);

/**
* @brief 高斯核二阶导
*/
template <typename T>
SDM3D* GetGaussianKernels(T sigma, int kernelSize);


void FreeSDM3D(SDM3D* x);
// 只是把hessian3D的子元素分配显存
void CudaMallocSDM3D(SDM3D* item, int imageSize);
void CudaFreeSDM3D(SDM3D* item);



#endif // _UTILS_H