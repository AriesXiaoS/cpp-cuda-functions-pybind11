#include "common.h"
#include "define.h"


PYBIND11_MODULE(cpp_cuda_functions, m)
{

    m.doc() = R"pbdoc(
        Python Bindings for C++/CUDA-based Processing Functions
        --------------------------------------------------------

        This module provides Python bindings for C++/CUDA-accelerated processing functions.
        
        Functions:
        - frangi3D
          Computes vessel features of a 3D image using Frangi filtering.

    )pbdoc";

    m.def("frangi3D", CudaFrangi3D<float>, 
        R"pbdoc(
            Compute vessel features of a 3D image.

            Args:
                image (numpy.ndarray): Input 3D image.
                device (int): GPU device index (default is 0).
                sigmas (List[float]): Scale parameters for Gaussian Filtering (default is [0.82, 1, 1.5]).
                alpha (float): plate sensitivity, range: (0, 1] (default is 0.5).
                beta (float): blobness, range: (0, 1] (default is 0.5).
                gamma (float): structuredness,, range: (0, +inf). 
                    if 0 : use half of maximum Hessian norm (default is 0).
                blackRidges (bool): If True, focus on black tubular structure; otherwise, white (default is False).
                maxIters (int): Maximum number of QR iterations (default is 30).
                tolerance (float): Tolerance in QR convergence (default is 1e-5).
                eigenVectorType (EigenVectorType): Type of eigenvectors (default is 0), 
                    0: no eigenvectors 
                    1: Cartesian eigenvectors
                    2: Spherical eigenvectors
                verbose (int): Control for verbose mode (default is 0).
                progressCallback_i_N (function): Progress callback function.
                    e.g. 
                    def progressCallback_i_N(i, N):
                        print(f"Progress: {i}/{N}") 
                        # N = len(sigmas)

            Returns:
                dict: {
                    "frangi": numpy.ndarray (Computed feature image, HxWxD),
                    "vectors": numpy.ndarray (Computed eigenvectors, HxWxDxC) if eigenVectorType > 0,
                }
        )pbdoc",
        py::arg("image"), py::arg("device") = 0, 
        py::arg("sigmas") = std::vector<float>{0.82, 1, 1.5},
        py::arg("alpha") = 0.5, py::arg("beta") = 0.5, py::arg("gamma") = 0, 
        py::arg("blackRidges") = false,
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5, 
        py::arg("eigenVectorType") = VEC_TYPE_NONE,
        py::arg("verbose") = 0,
        py::arg("cudaDimBlock") = std::vector<int>{6, 6, 6},
        py::arg("progressCallback_i_N") = nullptr
        );


    // 这玩意直接重载的话 np.float64 输入进来还是以float32执行的 不知道咋回事
    m.def("frangi3D_asDouble", CudaFrangi3D<double>, 
        R"pbdoc(
            Compute vessel features of a 3D image.

            Args:
                image (numpy.ndarray): Input 3D image.
                device (int): GPU device index (default is 0).
                sigmas (List[double]): Scale parameters for Gaussian Filtering (default is [0.82, 1, 1.5]).
                alpha (double): plate sensitivity, range: (0, 1] (default is 0.5).
                beta (double): blobness, range: (0, 1] (default is 0.5).
                gamma (double): structuredness,, range: (0, +inf). 
                    if 0 : use half of maximum Hessian norm (default is 0).
                blackRidges (bool): If True, focus on black tubular structure; otherwise, white (default is False).
                maxIters (int): Maximum number of QR iterations (default is 30).
                tolerance (double): Tolerance in QR convergence (default is 1e-5).
                eigenVectorType (EigenVectorType): Type of eigenvectors (default is 0), 
                    0: no eigenvectors 
                    1: Cartesian eigenvectors
                    2: Spherical eigenvectors
                verbose (int): Control for verbose mode (default is 0).
                progressCallback_i_N (function): Progress callback function.
                    e.g. 
                    def progressCallback_i_N(i, N):
                        print(f"Progress: {i}/{N}") 
                        # N = len(sigmas)

            Returns:
                dict: {
                    "frangi": numpy.ndarray (Computed feature image, HxWxD),
                    "vectors": numpy.ndarray (Computed eigenvectors, HxWxDxC) if eigenVectorType > 0,
                }
        )pbdoc",
        py::arg("image"), py::arg("device") = 0, 
        py::arg("sigmas") = std::vector<double>{0.82, 1, 1.5},
        py::arg("alpha") = 0.5, py::arg("beta") = 0.5, py::arg("gamma") = 0, 
        py::arg("blackRidges") = false,
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5, 
        py::arg("eigenVectorType") = VEC_TYPE_NONE,
        py::arg("verbose") = 0,
        py::arg("cudaDimBlock") = std::vector<int>{6, 6, 6},
        py::arg("progressCallback_i_N") = nullptr
        );

    m.def("printDeviceInfo", &PrintDeviceInfo, "print CUDA device info");







    // TEST
    m.def("addNp", AddNp<float>, "add 2 np.ndarray", 
        py::arg("vec1"), py::arg("vec2"), py::arg("device") = -1);
    m.def("addNp", AddNp<double>, "add 2 np.ndarray", 
        py::arg("vec1"), py::arg("vec2"), py::arg("device") = -1);

    // m.def("dict_test", &MapDictTest, "dict_test");

    m.def("padding3D", padding3dNp<float>, "padding 3 dimension np.ndarray",
            py::arg("vec"), py::arg("pad_value") = 0, 
            py::arg("pad_size_0") = 1, py::arg("pad_size_1") = 1, py::arg("pad_size_2") = 1 );
    m.def("padding3D", padding3dNp<double>, "padding 3 dimension np.ndarray",
            py::arg("vec"), py::arg("pad_value") = 0, 
            py::arg("pad_size_0") = 1, py::arg("pad_size_1") = 1, py::arg("pad_size_2") = 1 );

    m.def("cudaConvTest3D", CudaConv3dTest<float>, "conv 3D", 
        py::arg("vec"), py::arg("kernel"), py::arg("device") = 0);
    m.def("cudaConvTest3D", CudaConv3dTest<double>, "conv 3D", 
        py::arg("vec"), py::arg("kernel"), py::arg("device") = 0);

    m.def("qrSplitTest3x3", QRSplitTest_3x3<float>, "QR split test",
        py::arg("A"), py::arg("device") = 0);
    m.def("qrSplitTest3x3", QRSplitTest_3x3<double>, "QR split test",
        py::arg("A"), py::arg("device") = 0);

    m.def("qrEigensTest3x3", QREigensTest_3x3<float>, "Eigen test",
        py::arg("A"), py::arg("device") = -1, py::arg("maxIters") = 30, 
        py::arg("tolerance") = 1e-5, py::arg("vecType") = VEC_TYPE_CARTESIAN);
    m.def("qrEigensTest3x3", QREigensTest_3x3<double>, "Eigen test",
        py::arg("A"), py::arg("device") = -1, py::arg("maxIters") = 30, 
        py::arg("tolerance") = 1e-5, py::arg("vecType") = VEC_TYPE_CARTESIAN);

    m.def("hessianEigensTest3x3", HessianEigenTest_3x3<float>, "test",
        py::arg("A"), py::arg("vecType") = 1, py::arg("device") = 0, 
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5 );
    m.def("hessianEigensTest3x3", HessianEigenTest_3x3<double>, "test",
        py::arg("A"), py::arg("vecType") = 1, py::arg("device") = 0, 
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5 );





}






