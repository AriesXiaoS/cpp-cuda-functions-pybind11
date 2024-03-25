#include "common.h"
#include "define.h"
#include "utils.h"
#include "cuda.cuh"
#include "check.cuh"


// 卷积高斯二阶导 得到 Hessian矩阵
template <typename T>
void ConvHessian(T* paddedImage_d, SDM3D<T>* kernels, SDM3D<T>* hessian,
                 int* imgShape, int kernelSize, dim3 dimBlock, dim3 dimGrid)
{
    int kernelSizeFlattened = kernelSize * kernelSize * kernelSize;
    // kernel on device
    SDM3D<T>* kernels_d = new SDM3D<T>();
    CudaMallocSDM3D(kernels_d, kernelSizeFlattened);
    Conv3DParam convParams = {int(kernelSize), 
    { int(imgShape[0]), int(imgShape[1]), int(imgShape[2]), } };
    // xx
    CUDA_CHECK(cudaMemcpy(kernels_d->xx, kernels->xx, sizeof(T) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->xx, 
                                        kernels_d->xx, convParams);
    // yy
    CUDA_CHECK(cudaMemcpy(kernels_d->yy, kernels->yy, sizeof(T) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->yy, 
                                        kernels_d->yy, convParams);
    // zz
    CUDA_CHECK(cudaMemcpy(kernels_d->zz, kernels->zz, sizeof(T) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->zz, 
                                        kernels_d->zz, convParams);
    // xy
    CUDA_CHECK(cudaMemcpy(kernels_d->xy, kernels->xy, sizeof(T) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->xy, 
                                        kernels_d->xy, convParams);
    // xz
    CUDA_CHECK(cudaMemcpy(kernels_d->xz, kernels->xz, sizeof(T) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->xz, 
                                        kernels_d->xz, convParams);
    // yz
    CUDA_CHECK(cudaMemcpy(kernels_d->yz, kernels->yz, sizeof(T) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->yz, 
                                        kernels_d->yz, convParams);
    //
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaFreeSDM3D(kernels_d);
}

template <typename T>
T* VoMax(T** outputs, int sigmaLen, int imageSize)
{
    T* maxOutput = new T[imageSize];
    for(int i=0; i<imageSize; i++)
    {
        maxOutput[i] = outputs[0][i];
        for(int j=1; j<sigmaLen; j++)
        {
            if(outputs[j][i] > maxOutput[i])
            {
                maxOutput[i] = outputs[j][i];
            }
        }
    }
    return maxOutput;
}



template <typename T>
map<string, py::array_t<T>> CudaFrangi3D(
    py::array_t<T> image, int device, std::vector<T> sigmas,
    T alpha, T beta, T gamma, bool blackRidges,
    int maxIters, T tolerance, int eigenVectorType)
{
    auto img = image.request();
    
    if(img.ndim != 3) throw std::runtime_error("Number of dimensions must be 3");
    if(device<0) throw std::runtime_error("Device number must be non-negative");
    if(alpha<0) throw std::runtime_error("Alpha must be non-negative");
    if(beta<0) throw std::runtime_error("Beta must be non-negative");
    if(gamma<0) throw std::runtime_error("Gamma must be non-negative");
    if(maxIters<0) throw std::runtime_error("MaxIters must be non-negative");
    if(tolerance<0) throw std::runtime_error("Tolerance must be non-negative");
    if(eigenVectorType<VEC_TYPE_NONE || eigenVectorType>VEC_TYPE_SPHERE) throw std::runtime_error("EigenVectorType must be 0, 1 or 2");
    for(int i=0; i<sigmas.size(); i++)
    {
        if(sigmas[i]<0) throw std::runtime_error("Sigma must be non-negative");
    }

    cudaSetDevice(device);
    CUDA_CHECK(cudaGetLastError());
    //
    T* ptr = (T*) img.ptr;
    int* imgShape = new int[3]{int(img.shape[0]), int(img.shape[1]), int(img.shape[2])};
    int imageSize = imgShape[0] * imgShape[1] * imgShape[2];
    if(!blackRidges){
        for(int i=0; i<imageSize; i++){
            ptr[i] = -ptr[i];
        }
    }
    
    dim3 dimBlock(2,2,2);
    dim3 dimGrid( ceil( imgShape[0] / dimBlock.x) + 1,
                    ceil( imgShape[1] / dimBlock.y) + 1,
                    ceil( imgShape[2] / dimBlock.z) + 1 );
    // hessian 子元素在device 本体在host
    // hessian_d 本体也在device
    SDM3D<T>* hessian = new SDM3D<T>(); // 卷积后的
    SDM3D<T>* hessian_d;   
    CudaMallocSDM3D<T>(hessian, imageSize); 
    CUDA_CHECK(cudaMalloc((void**)&hessian_d, sizeof(SDM3D<T>)));
    CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(SDM3D<T>), cudaMemcpyHostToDevice));
    //
    Eigen3D<T>* eigen = new Eigen3D<T>();
    Eigen3D<T>* eigen_d;
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenValues, sizeof(T) *3 *imageSize));
    if(eigenVectorType != VEC_TYPE_NONE){
        CUDA_CHECK(cudaMalloc((void**)&eigen->eigenVectors, sizeof(T) *9 *imageSize));
    }
    CUDA_CHECK(cudaMalloc((void**)&eigen_d, sizeof(Eigen3D<T>)));
    CUDA_CHECK(cudaMemcpy(eigen_d, eigen, sizeof(Eigen3D<T>), cudaMemcpyHostToDevice));
    //
    T* HFnorm = new T[imageSize];
    T* HFnorm_d;
    CUDA_CHECK(cudaMalloc((void**)&HFnorm_d, sizeof(T) *imageSize));
    //
    T** outputs = new T*[sigmas.size()];
    for(int i=0; i<sigmas.size(); i++)
    {
        outputs[i] = new T[imageSize];
    }
    T* output_d;
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(T) * imageSize));

    for(int i=0; i<sigmas.size(); i++)
    {
        // 3 sigma 原则
        int kernelSize = 2 * ceil(3 * sigmas[i]) + 1;
        SDM3D<T>* kernels = GetGaussianKernels<T>(sigmas[i], kernelSize);
        // padding
        int paddingSize = kernelSize / 2;
        int paddedImageSize = (imgShape[0] + 2 * paddingSize) * (imgShape[1] + 2 * paddingSize) * (imgShape[2] + 2 * paddingSize);
        T* paddedImage = new T[paddedImageSize];
        PaddingFlattenedArr_3D<T>(ptr, paddedImage, 
                                imgShape[0], imgShape[1], imgShape[2],
                                0, paddingSize, paddingSize, paddingSize);
        // image to device
        T* paddedImage_d;
        CUDA_CHECK(cudaMalloc((void**)&paddedImage_d, sizeof(T) * paddedImageSize));
        CUDA_CHECK(cudaMemcpy(paddedImage_d, paddedImage, sizeof(T) * paddedImageSize, cudaMemcpyHostToDevice));
        // Conv Hessian
        ConvHessian<T>(paddedImage_d, kernels, hessian, imgShape, kernelSize, dimBlock, dimGrid);
        // printf("Conv Hessian Done\n");
        CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(SDM3D<T>), cudaMemcpyHostToDevice));
        // Hessian Eigen
        CudaHessianEigen<T><<<dimGrid, dimBlock>>>(hessian_d, eigen_d, HFnorm_d,
            imgShape[0], imgShape[1], imgShape[2], maxIters, tolerance, eigenVectorType);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("CudaHessianEigen Done\n");
        T frangi_c;
        if(gamma <= 0){
            CUDA_CHECK(cudaMemcpy(HFnorm, HFnorm_d, sizeof(T) * imageSize, cudaMemcpyDeviceToHost));
            frangi_c = *std::max_element(HFnorm, HFnorm + imageSize) * 0.5;
        }else{
            frangi_c = gamma;
        }

        CudaFrangiVo<T><<<dimGrid, dimBlock>>>(eigen_d, output_d,
            imgShape[0], imgShape[1], imgShape[2], alpha, beta, frangi_c, blackRidges);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(outputs[i], output_d, sizeof(T) * imageSize, cudaMemcpyDeviceToHost));
        // free device
        CUDA_CHECK(cudaFree(paddedImage_d));
        // free host
        delete[] paddedImage;
        FreeSDM3D<T>(kernels);

        printf("%d/%d sigma %f done\n", i+1, int(sigmas.size()), sigmas[i]);

    }

    T* frangi = VoMax<T>(outputs, sigmas.size(), imageSize);

    auto frangi_pyArr = py::array_t<T>(imageSize, frangi);
    frangi_pyArr.resize({imgShape[0], imgShape[1], imgShape[2]});

    map<string, py::array_t<T>> result;
    result["frangi"] = frangi_pyArr;

    return result;

}



template map<string, py::array_t<float>> CudaFrangi3D<float>(
    py::array_t<float> image, int device, std::vector<float> sigmas,
    float alpha, float beta, float gamma, bool blackRidges,
    int maxIters, float tolerance, int eigenVectorType);
template map<string, py::array_t<double>> CudaFrangi3D<double>(
    py::array_t<double> image, int device, std::vector<double> sigmas,
    double alpha, double beta, double gamma, bool blackRidges,
    int maxIters, double tolerance, int eigenVectorType);





















