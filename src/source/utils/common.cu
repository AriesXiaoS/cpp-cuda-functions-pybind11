#include "utils.h"
#include "define.h"
#include "cuda.cuh"
#include "check.cuh"

template <typename T>
void FreeSDM3D(SDM3D<T>* x)
{
    delete[] x->xx;
    delete[] x->yy;
    delete[] x->zz;
    delete[] x->xy;
    delete[] x->xz;
    delete[] x->yz;
    delete x;
}
template void FreeSDM3D<float>(SDM3D<float>* x);
template void FreeSDM3D<double>(SDM3D<double>* x);

// 只是把hessian3D的子元素分配显存
template <typename T>
void CudaMallocSDM3D(SDM3D<T>* item, int imageSize)
{
    CUDA_CHECK(cudaMalloc((void**)&item->xx, sizeof(T) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->yy, sizeof(T) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->zz, sizeof(T) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->xy, sizeof(T) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->xz, sizeof(T) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->yz, sizeof(T) * imageSize));
}
template void CudaMallocSDM3D<float>(SDM3D<float>* item, int imageSize);
template void CudaMallocSDM3D<double>(SDM3D<double>* item, int imageSize);

template <typename T>
void CudaFreeSDM3D(SDM3D<T>* item)
{
    CUDA_CHECK(cudaFree(item->xx));
    CUDA_CHECK(cudaFree(item->yy));
    CUDA_CHECK(cudaFree(item->zz));
    CUDA_CHECK(cudaFree(item->xy));
    CUDA_CHECK(cudaFree(item->xz));
    CUDA_CHECK(cudaFree(item->yz));
}
template void CudaFreeSDM3D<float>(SDM3D<float>* item);
template void CudaFreeSDM3D<double>(SDM3D<double>* item);


__device__ __host__ int getVecSize(int vecType){
    if(vecType == VEC_TYPE_CARTESIAN){
        return VEC_TYPE_CARTESIAN_SIZE;
    }else if(vecType == VEC_TYPE_SPHERE){
        return VEC_TYPE_SPHERE_SIZE;
    }


    else{
        return 0;
    }

}









