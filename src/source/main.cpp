#include "common.h"
#include "define.h"


PYBIND11_MODULE(cpp_cuda_functions, m)
{
    m.def("printDeviceInfo", &PrintDeviceInfo, "print CUDA device info");

    m.def("frangi3D", CudaFrangi3D<float>, "frangi 3D image",
        py::arg("image"), py::arg("device") = 0, 
        py::arg("sigmas") = std::vector<float>{0.82, 1, 1.5},
        py::arg("alpha") = 0.5, py::arg("beta") = 0.5, py::arg("gamma") = 0, 
        py::arg("blackRidges") = false,
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5, 
        py::arg("eigenVectorType") = VEC_TYPE_CARTESIAN);
    m.def("frangi3D", CudaFrangi3D<double>, "frangi 3D image",
        py::arg("image"), py::arg("device") = 0, 
        py::arg("sigmas") = std::vector<double>{0.82, 1, 1.5},
        py::arg("alpha") = 0.5, py::arg("beta") = 0.5, py::arg("gamma") = 0, 
        py::arg("blackRidges") = false,
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5, 
        py::arg("eigenVectorType") = VEC_TYPE_CARTESIAN);








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






