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

// Frangi 
template <typename T>
map<string, py::array_t<T>> CudaFrangi3D(
    py::array_t<T> image, int device, std::vector<T> sigmas,
    T alpha, T beta, T gamma, bool blackRidges,
    int maxIters, T tolerance, int eigenVectorType,
    int verbose, std::vector<int> cudaDimBlock);








// TEST
void PrintDeviceInfo();
map<string, py::array_t<float>> MapDictTest();

// add
template <typename T>
py::array_t<T> AddNp(py::array_t<T> vec1, py::array_t<T> vec2, int device);

template <typename T>
py::array_t<T> padding3dNp(py::array_t<T> vec, T pad_value, int pad_size_0, int pad_size_1, int pad_size_2);

template <typename T>
py::array_t<T> CudaConv3dTest(py::array_t<T> vec, py::array_t<T> kernel, int device);

template <typename T>
std::vector<py::array_t<T>> QRSplitTest_3x3(py::array_t<T> A, int device);

template <typename T>
std::vector<py::array_t<T>> QREigensTest_3x3(py::array_t<T> A, int device,
                                            int maxIters, T tolerance, int vecType);

template <typename T>
map<string, py::array_t<T>> HessianEigenTest_3x3(
    py::array_t<T> A, int vecType,
    int device, int maxIters, T tolerance);



#endif 
