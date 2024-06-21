// fdt.h
#pragma once

#include <vector>
#include <array>
#include <queue>

#include "thread_pool.h"
#include "utils.h"
#include "define.h"


std::vector<Voxel> getNeighborsVoxel(Voxel p, std::array<int, 3> shape);

class FuzzyObject3D{
private:
    float* src_arr;
    std::array<int, 3> src_shape;
    
    float* O1;
    int* O1_primary;    
    std::array<int, 3> O1_shape;

    void step_A1();
    void step_A2();

    /**
    检查点:
        出界 0 
        内部 1
        在图像边界 2
    */    
    int checkPointPos(Voxel p);
    std::vector<Voxel> getAdjacentPrimaryPoints(Voxel p);

public:
    FuzzyObject3D(float* arr, std::array<int, 3> imgShape);
    ~FuzzyObject3D(){
        // delete[] src_arr;
        // delete[] O1;
        delete[] O1_primary;
    }
    float* Excute();
    std::array<int, 3> getO1Shape();
};


class FDT3D{
private:
    int MAX = 9999;
    float* O1;
    float* fdt;

    std::array<float, 3> spacing_zyx;
    std::array<int, 3> src_shape; // zyx
    std::array<int, 3> O1_shape; // zyx

    float resolution_vec[26];
    void getResolutionVector();

    std::queue<Voxel> Q;
    std::vector<Voxel> Q0; // before
    std::vector<Voxel> Q1; // computed
    void initQ();
    float findDistMin(Voxel p);

    void mainLoop_singleThread();

    float* resampleBack();

    std::mutex* threadMutex; // 互斥锁
    int workingThreadNum = 0;
    int mainLoop_oneThread(Voxel p);
    void mainLoop_multiThread();
    

public:
    FDT3D(float* arr_O, std::array<float, 3> input_spacing_zyx, 
                        std::array<int, 3> input_shape);
    ~FDT3D(){
        delete[] O1;
        delete[] fdt;
        delete threadMutex;
    }

    float* Excute();

};








