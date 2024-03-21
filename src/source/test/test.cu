#include "common.h"
#include "utils.h"

#include "cuda.cuh"




map<string, py::array_t<float>> MapDictTest(){
    map<string, py::array_t<float>> res;
    res["a"] = py::array_t<float>(10);
    res["b"] = py::array_t<float>(10);
    return res;
}










