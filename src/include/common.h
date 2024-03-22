#ifndef _COMMON_H
#define _COMMON_H

#include <sstream>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <map>
#include <typeinfo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
using namespace chrono;
namespace py = pybind11;

// TEST
void PrintDeviceInfo();
map<string, py::array_t<float>> MapDictTest();
py::array_t<float> AddNp(py::array_t<float> vec1, py::array_t<float> vec2, int device);
py::array_t<float> Padding3dTest(py::array_t<float> vec, float pad_value, int pad_size_0, int pad_size_1, int pad_size_2);
py::array_t<float> CudaConv3dTest(py::array_t<float> vec, py::array_t<float> kernel, int device);
std::vector<py::array_t<float>> QRSplitTest_3x3(py::array_t<float> A, int device);
std::vector<py::array_t<float>> QREigensTest_3x3(py::array_t<float> A, int device,
                                            int maxIters, float tolerance);
map<string, py::array_t<float>> HessianEigenTest_3x3(
    py::array_t<float> A, int vecType,
    int device, int maxIters, float tolerance);



#endif 
