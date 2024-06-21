// mso.h
#pragma once

#include "../fdt/fdt.h"

#include "../thread_pool/thread_pool.h"

#include <vector>
#include <array>
#include <queue>

#include "utils.h"
#include "define.h"

#include "A_star.h"

class MSO3D{
private:
    float* img_O;
    std::array<float, 3> spacing; // zyx
    std::array<int, 3> shape; // zyx
    int size;

    float* fdt;
    float* fdt_normed;
    VoxelAStar astar;
    std::vector<Voxel> Smax;

    std::vector<Voxel> getNlp(Voxel p, int l);

    float shortestPathLength_AStar(Voxel start, Voxel end);
    
    float getLocalScale(Voxel p);
    void normalizeFDT();


public:
    MSO3D(float* arr, std::array<float, 3> input_spacing, 
                        std::array<int, 3> input_shape);
    ~MSO3D(){
        // delete[] isSmax_arr;
    }

    void computeFDT();
    void setFDT(float* arr);

    bool* isSmax_arr;
    void computeSmax();
    float* getNormedFDT();

    void Excute();
};







