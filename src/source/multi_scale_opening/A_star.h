// A_star.h
#pragma once

#include "utils.h"
#include "define.h"

#include <iostream>

struct AStarPoint{
    
    int z;
    int y;
    int x;
    int idx;

    float F;
    float G;
    float H;
    AStarPoint* parent;

    // 重载 == 运算符
    bool operator==(const AStarPoint& q) const {
        return  z==q.z && y==q.y && x==q.x;
    }

    AStarPoint(Voxel p):
        z(p.z), y(p.y), x(p.x), idx(p.idx), parent(nullptr){}
    AStarPoint(Voxel p, AStarPoint* parent):
        z(p.z), y(p.y), x(p.x), idx(p.idx), parent(parent){}
    // AStarPoint(Voxel p, AStarPoint* parent, float G, float H):
    //     z(p.z), y(p.y), x(p.x), idx(p.idx), parent(parent), G(G), H(H){
    //         F = G + H;
    //     }
    void setGH(float g, float h){
        G = g;
        H = h;
        F = G + H;
    }
    void setG(float g){
        G = g;
        F = G + H;
    }
    void setParent(AStarPoint* p){
        parent = p;
    }

    float updateF(){
        F = G + H;
        return F;
    }

};


class VoxelAStar{
private:
    float* img;
    std::array<float, 3> spacing; // zyx
    std::array<int, 3> shape; // zyx
    int size;

    Voxel start_voxel;
    Voxel end_voxel;

    std::vector<AStarPoint*> open_list;
    std::vector<AStarPoint*> close_list;
    //
    std::vector<Voxel> result_path;
    float result_distance = -1;

    bool isInOpenList(AStarPoint *p);
    bool isInCloseList(AStarPoint *p);
    int hanldOnePoint(AStarPoint *p);
    void freeList();

public:
    VoxelAStar(float* arr, std::array<float, 3> input_spacing, 
                        std::array<int, 3> input_shape);
    ~VoxelAStar(){
        // freeList();
    }

    void initStartEnd(Voxel start, Voxel end);
    void Update();

    std::vector<std::vector<int>> getPathList();
    float getDistance();
};





















