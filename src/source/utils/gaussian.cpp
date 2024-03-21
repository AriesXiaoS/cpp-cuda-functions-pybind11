#include "common.h"
#include "utils.h"


template <typename T>
GaussianPartialDerivativeKernel* getGaussianKernels(T sigma, int kernel_size){
    GaussianPartialDerivativeKernel *kernel = new GaussianPartialDerivativeKernel;
    kernel->size = kernel_size;
    kernel->sigma = sigma;
    int flatten_size = kernel_size * kernel_size * kernel_size;
    kernel->xx = new T[flatten_size];
    kernel->yy = new T[flatten_size];
    kernel->zz = new T[flatten_size];
    kernel->xy = new T[flatten_size];
    kernel->xz = new T[flatten_size];
    kernel->yz = new T[flatten_size];

    int x,y,z,idx;
    T same;
    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            for(int k=0; k<kernel_size; k++){
                z = i - kernel_size/2; 
                y = j - kernel_size/2;
                x = k - kernel_size/2;

                same = 1 / (pow(sqrt(2*M_PI),3)*pow(sigma,7)) * exp(-(pow(x,2)+pow(y,2)+pow(z,2))/(2*pow(sigma,2)));

                idx = i*kernel_size*kernel_size + j*kernel_size + k;

                kernel->xx[idx] = (pow(x,2)-pow(sigma,2)) * same;
                kernel->yy[idx] = (pow(y,2)-pow(sigma,2)) * same;
                kernel->zz[idx] = (pow(z,2)-pow(sigma,2)) * same;

                kernel->xy[idx] = (x*y) * same;
                kernel->xz[idx] = (x*z) * same;
                kernel->yz[idx] = (y*z) * same;
            }
        }
    }
    return kernel;
}
