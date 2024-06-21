#include "common.h"
#include "define.h"
#include "utils.h"

#include "mso.h"

template <typename T>
py::array_t<T> testMSO(py::array_t<T> array, py::array_t<T> fdt, 
std::vector<float> spacing)
{

    auto arr = array.request();    
    T* arr_p = (T*) arr.ptr;
    int size = arr.shape[0] * arr.shape[1] * arr.shape[2];

    auto fdt_arr = fdt.request();
    T* fdt_p = (T*) fdt_arr.ptr;

    MSO3D mso = MSO3D(arr_p, 
        std::array<float, 3>{spacing[0], spacing[1], spacing[2]}, 
        std::array<int, 3>{int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])});
    mso.setFDT(fdt_p);
    mso.Excute();

    // float* res = new float[size]{0};
    // for(int i=0; i<size; i++){
    //     if(mso.isSmax_arr[i]){
    //         res[i] = 1;
    //     }
    // }

    float* res = mso.getNormedFDT();


    auto result = py::array_t<T>(size, res);
    result.resize({arr.shape[0], arr.shape[1], arr.shape[2]});
    return result;
}

template py::array_t<float> testMSO(py::array_t<float> array, py::array_t<float> fdt, std::vector<float> spacing);











