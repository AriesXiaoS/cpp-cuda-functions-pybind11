
#include "common.h"

#include "cuda.cuh"

void PrintDeviceInfo(){    
    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
 
    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device index: %d \n", i);
        printf("GPU name: %s \n", prop.name);
        printf("GPU version: %d.%d \n", prop.major,prop.minor);
        printf("total Memory: %d GB\n", int(prop.totalGlobalMem / (1000*1000*1000)) );
        printf("thread max shared memory: %d KB \n", int(prop.sharedMemPerBlock / 1000) );
        printf("max Thread num in Block: %d \n", prop.maxThreadsPerBlock );
        printf("max Thread num each dim: %d %d %d \n", 
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        printf("max Block num in Grid: %d %d %d \n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("max processor: %d \n",
            prop.multiProcessorCount);
        printf("----\n");
    }
}









