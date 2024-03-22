#include "common.h"
#include "define.h"
#include "utils.h"
#include "cuda.cuh"
#include "check.cuh"


// 卷积高斯二阶导 得到 Hessian矩阵
void ConvHessian(float* paddedImage_d, SDM3D* kernels, SDM3D* hessian,
                 int* imgShape, int kernelSize, dim3 dimBlock, dim3 dimGrid)
{
    int kernelSizeFlattened = kernelSize * kernelSize * kernelSize;
    // kernel on device
    SDM3D* kernels_d = new SDM3D();
    CudaMallocSDM3D(kernels_d, kernelSizeFlattened);
    Conv3DParam convParams = {int(kernelSize), 
    { int(imgShape[0]), int(imgShape[1]), int(imgShape[2]), } };
    // xx
    CUDA_CHECK(cudaMemcpy(kernels_d->xx, kernels->xx, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->xx, 
                                        kernels_d->xx, convParams);
    // yy
    CUDA_CHECK(cudaMemcpy(kernels_d->yy, kernels->yy, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->yy, 
                                        kernels_d->yy, convParams);
    // zz
    CUDA_CHECK(cudaMemcpy(kernels_d->zz, kernels->zz, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->zz, 
                                        kernels_d->zz, convParams);
    // xy
    CUDA_CHECK(cudaMemcpy(kernels_d->xy, kernels->xy, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->xy, 
                                        kernels_d->xy, convParams);
    // xz
    CUDA_CHECK(cudaMemcpy(kernels_d->xz, kernels->xz, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->xz, 
                                        kernels_d->xz, convParams);
    // yz
    CUDA_CHECK(cudaMemcpy(kernels_d->yz, kernels->yz, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
    CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->yz, 
                                        kernels_d->yz, convParams);
    //
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaFreeSDM3D(kernels_d);
}

float* VoMax(float** outputs, int sigmaLen, int imageSize)
{
    float* maxOutput = new float[imageSize];
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



map<string, py::array_t<float>> CudaFrangi3D(
    py::array_t<float> image, int device, std::vector<float> sigmas,
    float alpha, float beta, float gamma, bool blackRidges,
    int maxIters, float tolerance, int eigenVectorType)
{
    auto img = image.request();
    printf("img type ");
    cout << img.format << endl;
    if(img.ndim != 3) throw std::runtime_error("Number of dimensions must be 3");
    if(device<0) throw std::runtime_error("Device number must be non-negative");
    if(alpha<0) throw std::runtime_error("Alpha must be non-negative");
    if(beta<0) throw std::runtime_error("Beta must be non-negative");
    if(gamma<0) throw std::runtime_error("Gamma must be non-negative");
    if(maxIters<0) throw std::runtime_error("MaxIters must be non-negative");
    if(tolerance<0) throw std::runtime_error("Tolerance must be non-negative");
    if(eigenVectorType<0 || eigenVectorType>2) throw std::runtime_error("EigenVectorType must be 0, 1 or 2");
    for(int i=0; i<sigmas.size(); i++)
    {
        if(sigmas[i]<0) throw std::runtime_error("Sigma must be non-negative");
    }

    cudaSetDevice(device);
    CUDA_CHECK(cudaGetLastError());
    //
    float* ptr = (float*) img.ptr;
    int* imgShape = new int[3]{int(img.shape[0]), int(img.shape[1]), int(img.shape[2])};
    int imageSize = imgShape[0] * imgShape[1] * imgShape[2];
    
    dim3 dimBlock(2,2,2);
    dim3 dimGrid( ceil( imgShape[0] / dimBlock.x) + 1,
                    ceil( imgShape[1] / dimBlock.y) + 1,
                    ceil( imgShape[2] / dimBlock.z) + 1 );
    // hessian 子元素在device 本体在host
    // hessian_d 本体也在device
    SDM3D* hessian = new SDM3D(); // 卷积后的
    SDM3D* hessian_d;   
    CudaMallocSDM3D(hessian, imageSize); 
    CUDA_CHECK(cudaMalloc((void**)&hessian_d, sizeof(SDM3D)));
    CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(SDM3D), cudaMemcpyHostToDevice));
    //
    Eigen3D* eigen = new Eigen3D();
    Eigen3D* eigen_d;
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenValues, sizeof(float) *3 *imageSize));
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenVectors, sizeof(float) *9 *imageSize));
    CUDA_CHECK(cudaMalloc((void**)&eigen_d, sizeof(Eigen3D)));
    CUDA_CHECK(cudaMemcpy(eigen_d, eigen, sizeof(Eigen3D), cudaMemcpyHostToDevice));
    //
    float* HFnorm = new float[imageSize];
    float* HFnorm_d;
    CUDA_CHECK(cudaMalloc((void**)&HFnorm_d, sizeof(float) *imageSize));
    //
    float** outputs = new float*[sigmas.size()];
    for(int i=0; i<sigmas.size(); i++)
    {
        outputs[i] = new float[imageSize];
    }
    float* output_d;
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float) * imageSize));

    for(int i=0; i<sigmas.size(); i++)
    {
        // 3 sigma 原则
        int kernelSize = 2 * ceil(3 * sigmas[i]) + 1;
        SDM3D* kernels = GetGaussianKernels<float>(sigmas[i], kernelSize);
        // padding
        int paddingSize = kernelSize / 2;
        int paddedImageSize = (imgShape[0] + 2 * paddingSize) * (imgShape[1] + 2 * paddingSize) * (imgShape[2] + 2 * paddingSize);
        float* paddedImage = new float[paddedImageSize];
        PaddingFlattenedArr_3D(ptr, paddedImage, 
                                imgShape[0], imgShape[1], imgShape[2],
                                0, paddingSize, paddingSize, paddingSize);
        // image to device
        float* paddedImage_d;
        CUDA_CHECK(cudaMalloc((void**)&paddedImage_d, sizeof(float) * paddedImageSize));
        CUDA_CHECK(cudaMemcpy(paddedImage_d, paddedImage, sizeof(float) * paddedImageSize, cudaMemcpyHostToDevice));
        // Conv Hessian
        ConvHessian(paddedImage_d, kernels, hessian, imgShape, kernelSize, dimBlock, dimGrid);
        // printf("Conv Hessian Done\n");
        CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(SDM3D), cudaMemcpyHostToDevice));
        // eigens
        dim3 dimBlock1(1,1,1);
        dim3 dimGrid1(1,1,1);
        SetHessianParams<<<dimGrid1, dimBlock1>>>(imgShape[0], imgShape[1], imgShape[2], maxIters, tolerance, eigenVectorType);
        // printf("before CudaHessianEigen\n");
        CudaHessianEigen<<<dimGrid, dimBlock>>>(hessian_d, eigen_d, HFnorm_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("CudaHessianEigen Done\n");
        float frangi_c;
        if(gamma <= 0){
            CUDA_CHECK(cudaMemcpy(HFnorm, HFnorm_d, sizeof(float) * imageSize, cudaMemcpyDeviceToHost));
            frangi_c = *std::max_element(HFnorm, HFnorm + imageSize) * 0.5;
        }else{
            frangi_c = gamma;
        }
    
        // //
        SetFrangiParams<<<dimGrid1, dimBlock1>>>(imgShape[0],  imgShape[1], imgShape[2], alpha, beta, frangi_c, blackRidges);
        CudaFrangiVo<<<dimGrid, dimBlock>>>(eigen_d, output_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(outputs[i], output_d, sizeof(float) * imageSize, cudaMemcpyDeviceToHost));
        // free device
        CUDA_CHECK(cudaFree(paddedImage_d));
        // free host
        delete[] paddedImage;
        FreeSDM3D(kernels);

    }

    float* frangi = VoMax(outputs, sigmas.size(), imageSize);

    auto frangi_pyArr = py::array_t<float>(imageSize, frangi);

    // auto buf_res = frangi_pyArr.request();
    // float* ptr_res = (float*) buf_res.ptr;

    // for(int i=0; i<imageSize; i++)
    // {
    //     ptr_res[i] = frangi[i];
    // }

    frangi_pyArr.resize({imgShape[0], imgShape[1], imgShape[2]});

    map<string, py::array_t<float>> result;
    result["frangi"] = frangi_pyArr;

    return result;

}

























