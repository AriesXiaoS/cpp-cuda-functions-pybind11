#ifndef _UTILS_H
#define _UTILS_H


/**
* @brief Padding 三维数组 但在一维层面
*/
void PaddingFlattenedArr_3D(float* arr, float* result, 
        int size_0, int size_1, int size_2,
        float pad_value, int pad_size_0, int pad_size_1, int pad_size_2);

/**
* @brief 高斯核二阶导
*/
struct GaussianPartialDerivativeKernel{
    int size;
    float sigma;
    float* xx;
    float* yy;
    float* zz;
    float* xy;
    float* xz;
    float* yz;
};









#endif // _UTILS_H