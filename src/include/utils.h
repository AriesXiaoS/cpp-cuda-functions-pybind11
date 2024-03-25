#ifndef _UTILS_H
#define _UTILS_H

#include "define.h"

/**
* @brief Padding 三维数组 但在一维层面
*/
template <typename T>
void PaddingFlattenedArr_3D(T* arr, T* result, 
        int size_0, int size_1, int size_2,
        T pad_value, int pad_size_0, int pad_size_1, int pad_size_2);

/**
* @brief 高斯核二阶导
*/
template <typename T>
SDM3D<T>* GetGaussianKernels(T sigma, int kernelSize);


template <typename T>
void FreeSDM3D(SDM3D<T>* x)
;
// 只是把hessian3D的子元素分配显存
template <typename T>
void CudaMallocSDM3D(SDM3D<T>* item, int imageSize);

template <typename T>
void CudaFreeSDM3D(SDM3D<T>* item);



#endif // _UTILS_H