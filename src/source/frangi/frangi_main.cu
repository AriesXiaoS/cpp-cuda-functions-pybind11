
#include "define.h"
#include "utils.h"
#include "cuda.cuh"

// 只是把hessian3D的子元素分配显存
void MallocHessian3D(Hessian3D* hessian, int imageSize)
{
    CUDA_CHECK(cudaMalloc((void**)&hessian->Ixx, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&hessian->Iyy, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&hessian->Izz, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&hessian->Ixy, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&hessian->Ixz, sizeof(float) * imageSize));
    CUDA_CHECK(cudaMalloc((void**)&hessian->Iyz, sizeof(float) * imageSize));
}




map<string, py::array_t<float>> CudaFrangi3D(
    py::array_t<float> image, int device, std::vector<float> sigmas,
    float alpha, float beta, float gamma, bool blackRidges,
    int maxIters, float tolerance, int eigenVectorType)
{
    auto img = image.request();
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
    int* imgShape = new int[3]{img.shape[0], img.shape[1], img.shape[2]};
    int imageSize = imgShape[0] * imgShape[1] * imgShape[2];
    Hessian3D* hessian = new Hessian3D(); // 卷积后的
    Hessian3D* hessian_d;   
    MallocHessian3D(hessian, imageSize); 
    CUDA_CHECK(cudaMalloc((void**)&hessian_d, sizeof(Hessian3D)));
    CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(Hessian3D), cudaMemcpyHostToDevice));
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
    float** outputs = new float[sigmas.size()];
    for(int i=0; i<sigmas.size(); i++)
    {
        outputs[i] = new float[imageSize];
    }
    float* output_d;
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float) * imageSize));

    for(int i=0; i<sigmas.size(); i++)
    {
        int kernelSize = 2 * ceil(3 * sigmas[i]) + 1;
        GaussianPartialDerivativeKernel* kernels = GetGaussianKernels<float>(sigmas[i], kernelSize);
        int paddingSize = kernelSize / 2;
        int paddedImageSize = (imgShape[0] + 2 * paddingSize) * (imgShape[1] + 2 * paddingSize) * (imgShape[2] + 2 * paddingSize);
        float* paddedImage = new float[paddedImageSize];
        //
        PaddingFlattenedArr_3D(ptr, paddedImage, 
        imgShape[0], imgShape[1], imgShape[2],
        0, paddingSize, paddingSize, paddingSize)
        //
        Conv3DParam convParams = {int(kernelSize), 
        { int(imgShape[0]), int(imgShape[1]), int(imgShape[2]), } }
        // image
        float* paddedImage_d;
        CUDA_CHECK(cudaMalloc((void**)&paddedImage_d, sizeof(float) * paddedImageSize));
        CUDA_CHECK(cudaMemcpy(paddedImage_d, paddedImage, sizeof(float) * paddedImageSize, cudaMemcpyHostToDevice));
        // kernel
        int kernelSizeFlattened = kernelSize * kernelSize * kernelSize;
        float* kernel_xx_d;
        float* kernel_yy_d;
        float* kernel_zz_d;
        float* kernel_xy_d;
        float* kernel_xz_d;
        float* kernel_yz_d;
        dim3 dimBlock(8,8,8);
        dim3 dimGrid( ceil( imgShape[0] / dimBlock.x),
                      ceil( imgShape[1] / dimBlock.y),
                      ceil( imgShape[2] / dimBlock.z)  );
        // xx
        CUDA_CHECK(cudaMalloc((void**)&kernel_xx_d, sizeof(float) * kernelSizeFlattened));
        CUDA_CHECK(cudaMemcpy(kernel_xx_d, kernels->xx, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
        CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->Ixx, 
                                            kernel_xx_d, convParams);
        // yy
        CUDA_CHECK(cudaMalloc((void**)&kernel_yy_d, sizeof(float) * kernelSizeFlattened));
        CUDA_CHECK(cudaMemcpy(kernel_yy_d, kernels->yy, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
        CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->Iyy, 
                                            kernel_yy_d, convParams);
        // zz
        CUDA_CHECK(cudaMalloc((void**)&kernel_zz_d, sizeof(float) * kernelSizeFlattened));
        CUDA_CHECK(cudaMemcpy(kernel_zz_d, kernels->zz, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
        CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->Izz, 
                                            kernel_zz_d, convParams);
        // xy
        CUDA_CHECK(cudaMalloc((void**)&kernel_xy_d, sizeof(float) * kernelSizeFlattened));
        CUDA_CHECK(cudaMemcpy(kernel_xy_d, kernels->xy, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
        CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->Ixy, 
                                            kernel_xy_d, convParams);
        // xz
        CUDA_CHECK(cudaMalloc((void**)&kernel_xz_d, sizeof(float) * kernelSizeFlattened));
        CUDA_CHECK(cudaMemcpy(kernel_xz_d, kernels->xz, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
        CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->Ixz, 
                                            kernel_xz_d, convParams);
        // yz
        CUDA_CHECK(cudaMalloc((void**)&kernel_yz_d, sizeof(float) * kernelSizeFlattened));
        CUDA_CHECK(cudaMemcpy(kernel_yz_d, kernels->yz, sizeof(float) * kernelSizeFlattened, cudaMemcpyHostToDevice));
        CudaConv3D<<<dimGrid, dimBlock>>>(paddedImage_d, hessian->Iyz, 
                                            kernel_yz_d, convParams);
        //
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // eigens
        dim3 dimBlock1(1,1,1);
        dim3 dimGrid1(1, 1, 1);

        SetHessianParams<<<dimGrid1, dimBlock1>>>(imgShape[0],  imgShape[1], imgShape[2], maxIters, tolerance, eigenVectorType);
        CudaHessianEigen<<<dimGrid, dimBlock>>>(hessian_d, eigen_d, HFnorm_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float frangi_c;
        if(gamma <= 0){
            CUDA_CHECK(cudaMemcpy(HFnorm, HFnorm_d, sizeof(float) * imageSize, cudaMemcpyDeviceToHost));
            frangi_c = *std::max_element(HFnorm, HFnorm + imageSize) * 0.5;
        }else{
            frangi_c = gamma;
        }
        
        
        //
        SetFrangiParams<<<dimGrid1, dimBlock1>>>(imgShape[0],  imgShape[1], imgShape[2], alpha, beta, frangi_c, blackRidges);
        CudaFrangiVo<<<dimGrid, dimBlock>>>(eigen_d, output_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(outputs[i], output_d, sizeof(float) * imageSize, cudaMemcpyDeviceToHost));

    }



}

























