#include "common.h"
#include "define.h"
#include "utils.h"

#include "fdt.h"

template <typename T>
py::array_t<T> testFuzzyObject(py::array_t<T> array)
{
    auto arr = array.request();    
    T* ptr = (T*) arr.ptr;
    FuzzyObject3D fo = FuzzyObject3D(ptr, 
                        std::array<int, 3>{int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])});
    float* res = fo.Excute();
    std::array<int, 3> newShape = fo.getO1Shape();

    auto result = py::array_t<T>(newShape[0]*newShape[1]*newShape[2], res);
    result.resize({newShape[0], newShape[1], newShape[2]});

    return result;
}

template py::array_t<float> testFuzzyObject(py::array_t<float> array);


// template <typename T>
// py::array_t<T> testFDT(py::array_t<T> array, std::vector<float> py_spacing)
// {
//     auto arr = array.request();    
//     T* ptr = (T*) arr.ptr;
//     float* spacing = new float[3]{py_spacing[0], py_spacing[1], py_spacing[2]};
//     int* shape = new int[3]{int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])};

//     FDT3D fdt = FDT3D(ptr, spacing, shape);
//     float* res = fdt.Excute();

//     cout << "fdt done" << endl;

//     auto result = py::array_t<T>(arr.shape[0]*arr.shape[1]*arr.shape[2], res);
//     cout << "fdt result" << endl;
//     result.resize({arr.shape[0], arr.shape[1], arr.shape[2]});
//     cout << "fdt resize" << endl;
//     return result;
// }
// template py::array_t<float> testFDT(py::array_t<float> array, std::vector<float> py_spacing);


















