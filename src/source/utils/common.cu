#include "utils.h"
#include "cuda.cuh"
#include "check.cuh"


void FreeSDM3D(SDM3D* x)
{
    delete[] x->xx;
    delete[] x->yy;
    delete[] x->zz;
    delete[] x->xy;
    delete[] x->xz;
    delete[] x->yz;
    delete x;
}

// 只是把hessian3D的子元素分配显存
void CudaMallocSDM3D(SDM3D* item, int imageSize)
{
    CUDA_CHECK(cudaMalloc((void**)&item->xx, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->yy, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->zz, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->xy, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->xz, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&item->yz, sizeof(float) * imageSize));
}
void CudaFreeSDM3D(SDM3D* item)
{
    CUDA_CHECK(cudaFree(item->xx));
    CUDA_CHECK(cudaFree(item->yy));
    CUDA_CHECK(cudaFree(item->zz));
    CUDA_CHECK(cudaFree(item->xy));
    CUDA_CHECK(cudaFree(item->xz));
    CUDA_CHECK(cudaFree(item->yz));
    delete item;
}












