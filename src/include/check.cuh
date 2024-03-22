#ifndef _CHECK_H
#define _CHECK_H

#include <sstream>
#include <iostream>
#include "cuda.cuh"

// CUDA API error checking
#define CUDA_CHECK(call)                                \
    do{                                                 \
        cudaError_t err_ = (call);                      \
        if (err_ != cudaSuccess) {                      \
            std::stringstream strstr;                   \
            strstr << "CUDA error " << err_;            \
            strstr << " at " << __FILE__ ;              \
            strstr << ":" << __LINE__ << std::endl;     \
            strstr << cudaGetErrorString(err_);         \
            throw std::runtime_error(strstr.str());     \
        }                                               \
    } while (0)















#endif // !_CHECK_
