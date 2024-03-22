#include "common.h"

PYBIND11_MODULE(cuda_functions, m)
{
    m.def("device_info", &PrintDeviceInfo, "print Device Info");

    m.def("add_np", &AddNp, "np add test", 
        py::arg("vec1"), py::arg("vec2"), py::arg("device") = -1);

    m.def("dict_test", &MapDictTest, "dict_test");

    m.def("padding_3d_test", &Padding3dTest, "test");

    m.def("cuda_conv_3d_test", &CudaConv3dTest, "test");

    m.def("qr_split_test_3x3", &QRSplitTest_3x3, "test");

    m.def("qr_eigens_test_3x3", &QREigensTest_3x3, "test",
        py::arg("A"), py::arg("device") = -1, 
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5 );

    m.def("hessian_eigens_test_3x3", &HessianEigenTest_3x3, "test",
        py::arg("A"), py::arg("vecType") = 1, py::arg("device") = 0, 
        py::arg("maxIters") = 30, py::arg("tolerance") = 1e-5 );





}






