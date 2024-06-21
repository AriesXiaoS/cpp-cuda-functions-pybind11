#include "common.h"
#include "define.h"
#include "utils.h"

#include "A_star.h"



std::vector<std::vector<int>> testAStar(py::array_t<float> array, 
std::vector<int> start, 
std::vector<int> end, 
std::vector<float> spacing)
{
    
    auto arr = array.request();    
    float* arr_p = (float*) arr.ptr;
    int size = arr.shape[0] * arr.shape[1] * arr.shape[2];
    
    std::array<int, 3> shape = {int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])};
    std::array<float, 3> spacing_arr = {spacing[0], spacing[1], spacing[2]};

    VoxelAStar astar = VoxelAStar(arr_p, spacing_arr, shape);
    astar.initStartEnd(Voxel(start[0], start[1], start[2], shape), 
                        Voxel(end[0], end[1], end[2], shape));
    astar.Update();

    return astar.getPathList();

}



















