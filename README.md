# C++/CUDA-functions-pybind11

Providing accelerated functions in C++ and CUDA as a library for Python.

Currently primarily targeting the Frangi vascular filtering algorithm for three-dimensional images.

Usage examples can be found in _test.py_.
```python
def frangi3D
def frangi3D_asDouble
```
Compute vessel features of a 3D image.
#### Arguments:
- `image` (numpy.ndarray): Input 3D image.
- `device` (int): GPU device index (default is 0).
- `sigmas` (List[float/double]): Scale parameters for Gaussian Filtering (default is [0.82, 1, 1.5]).
- `alpha` (float/double): plate sensitivity, range: (0, 1] (default is 0.5).
- `beta` (float/double): blobness, range: (0, 1] (default is 0.5).
- `gamma` (float/double): structuredness,, range: (0, +inf). 
  - if 0 : use half of maximum Hessian norm (default is 0).
- `blackRidges` (bool): If True, focus on black tubular structure; otherwise, white (default is False).
- `maxIters` (int): Maximum number of QR iterations (default is 30).
- `tolerance` (float/double): Tolerance in QR convergence (default is 1e-5).
- `eigenVectorType` (EigenVectorType): Type of eigenvectors (default is 0), 
  - 0: no eigenvectors 
  - 1: Cartesian eigenvectors
  - 2: Spherical eigenvectors
- `verbose` (int): Control for verbose mode (default is 0).
- `progressCallback_i_N` (function): Progress callback function.
  - e.g. 
    ```python
    def cb(i, N): # N = len(sigmas)
        print(f"Progress: {i}/{N}") 
    ```
#### Returns:
- dict: 
```python
{
    "frangi": numpy.ndarray (Computed feature image, HxWxD),
    "vectors" [if eigenVectorType > 0]: numpy.ndarray (Computed eigenvectors, HxWxDxC),
}
```

## pybind11-cuda

Starting point for GPU accelerated python libraries

Originally based on https://github.com/pkestene/pybind11-cuda


## Prerequisites

Cuda

Python 3.6 or greater

Cmake >= 3.12 (for CUDA support and the new FindPython3 module)

## Build instructions

### cmake

If you use cmake version >= 3.18, you can use [variable CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html) instead of CUDAFLAGS:


```bash
mkdir build; cd build
## provide a default cuda hardware architecture to build for
cmake -DCMAKE_CUDA_ARCHITECTURES="75" -DPython3_EXECUTABLE=`which python` .. && make
# cmake -DCMAKE_CUDA_ARCHITECTURES="86" -DPython3_EXECUTABLE=`which python` .. && make
```

Please note that specifiying `Python3_EXECUTABLE` is not required, but recommended if you have multiple python executable on your system (e.g. one from OS, another from conda, etc...); this way you can control which python installation will be used.

If you have an older version cmake, you can pass nvcc flags to cmake using env variable `CUDAFLAGS`

```bash
## change CUDAFLAGS according to your target GPU architecture
mkdir build; cd build
## provide a default cuda hardware architecture to build for
export CUDAFLAGS="-arch=sm_75"
cmake -DPython3_EXECUTABLE=`which python` ..
make
```

## test

Test it with
```shell
cd build/src
python3 test.py
```

_cpp_cuda_functions.so_ and test.py_ must be in the same folder. Alternatively you can path to _cpp_cuda_functions.so_ to your PYTHONPATH env variable.

