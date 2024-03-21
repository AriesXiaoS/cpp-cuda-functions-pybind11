
#include "utils.h"


void PaddingFlattenedArr_3D(float* arr, float* result, 
        int size_0, int size_1, int size_2,
        float pad_value, int pad_size_0, int pad_size_1, int pad_size_2){
    int new_size_0 = size_0 + 2*pad_size_0;
    int new_size_1 = size_1 + 2*pad_size_1;
    int new_size_2 = size_2 + 2*pad_size_2;
    // padding
    for(int i=0;i<new_size_0;i++){
        for(int j=0;j<new_size_1;j++){
            for(int k=0;k<new_size_2;k++){
                if(i<pad_size_0 || i>=size_0+pad_size_0 || 
                j<pad_size_1 || j>=size_1+pad_size_1 || 
                k<pad_size_2 || k>=size_2+pad_size_2){
                    result[i*new_size_1*new_size_2 + j*new_size_2 + k] = pad_value;
                }else{
                    result[i*new_size_1*new_size_2 + j*new_size_2 + k] = arr[(i-pad_size_0)*size_1*size_2 + (j-pad_size_1)*size_2 + (k-pad_size_2)];
                }
            }
        }
    }    
}

