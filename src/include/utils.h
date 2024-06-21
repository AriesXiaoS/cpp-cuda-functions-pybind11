#ifndef _UTILS_H
#define _UTILS_H

#include <vector>
#include <array>
#include <queue>
#include <cmath>

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


//
struct Voxel{
    bool valid=true;
    int z;
    int y;
    int x;
    int idx=-1;

    // 重载 == 运算符
    bool operator==(const Voxel& q) const {
        return z==q.z && y==q.y && x==q.x;
    }

    Voxel(int z, int y, int x): z(z), y(y), x(x){}
    Voxel(int z, int y, int x, int idx): z(z), y(y), x(x), idx(idx){}
    Voxel(int z, int y, int x, std::array<int, 3> shape): z(z), y(y), x(x){
        idx = z*shape[1]*shape[2] + y*shape[2] + x;
    }
    Voxel(bool valid): valid(valid){}
    Voxel(){}
    Voxel(int idx, std::array<int, 3> shape){
        this->idx = idx;
        z = idx/(shape[1]*shape[2]);
        y = (idx%(shape[1]*shape[2]))/shape[2];
        x = (idx%(shape[1]*shape[2]))%shape[2];
    }


    int getIdx(std::array<int, 3> shape){
        if(idx==-1){
            return z*shape[1]*shape[2] + y*shape[2] + x;
        }else{
            return idx;
        }
    }
};


#endif // _UTILS_H