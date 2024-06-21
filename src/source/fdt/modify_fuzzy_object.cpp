#include "fdt.h"
#include <cassert>



FuzzyObject3D::FuzzyObject3D(float* arr, std::array<int, 3> imgShape)
{
    src_shape[0] = imgShape[0];
    src_shape[1] = imgShape[1];
    src_shape[2] = imgShape[2];

    O1_shape[0] = 2*src_shape[0] + 1;
    O1_shape[1] = 2*src_shape[1] + 1;
    O1_shape[2] = 2*src_shape[2] + 1;
    int O1_size = O1_shape[0]*O1_shape[1]*O1_shape[2];

    src_arr = arr;
    O1 = new float[O1_size]{0};
    O1_primary = new int[O1_size]{0};
}


void FuzzyObject3D::step_A1()
{
    int src_idx, O1_idx;
    for(int i=0; i<src_shape[0]; i++){
        for(int j=0; j<src_shape[1]; j++){
            for(int k=0; k<src_shape[2]; k++){
                src_idx = i*src_shape[1]*src_shape[2] + j*src_shape[2] + k;
                O1_idx = (2*i+1)*O1_shape[1]*O1_shape[2] + (2*j+1)*O1_shape[2] + (2*k+1);
                O1[O1_idx] = src_arr[src_idx];
                O1_primary[O1_idx] = 1;                
            }
        }
    }
}


int FuzzyObject3D::checkPointPos(Voxel p)
{
    if(p.z<0 || p.z>=O1_shape[0] 
    || p.y<0 || p.y>=O1_shape[1] 
    || p.x<0 || p.x>=O1_shape[2]){
        return 0;
    }
    if(p.z==0 || p.z==O1_shape[0]-1 
    || p.y==0 || p.y==O1_shape[1]-1 
    || p.x==0 || p.x==O1_shape[2]-1){
        return 2;
    }
    return 1;
}

std::vector<Voxel> FuzzyObject3D::getAdjacentPrimaryPoints(Voxel p)
{
    std::vector<Voxel> neighbours = getNeighborsVoxel(p, O1_shape);
    std::vector<Voxel> res;

    for(auto& n: neighbours){
        if(!n.valid){
            continue;
        }
        if(O1_primary[n.getIdx(O1_shape)]==1){
            res.push_back(n);
        }
    }
    return res;
}

void FuzzyObject3D::step_A2()
{
    int O1_idx;
    for(int i=0; i<O1_shape[0]; i++){
        for(int j=0; j<O1_shape[1]; j++){
            for(int k=0; k<O1_shape[2]; k++){
                if(i%2==1 && j%2==1 && k%2==1) continue;
                //
                Voxel p = Voxel(i, j, k, O1_shape);
                O1_idx = p.getIdx(O1_shape);
                int p_pos = checkPointPos(p);
                if(p_pos==0) continue; // # 出界
                if(p_pos==2){ // # 在图像边界 同视为 Object边界
                    O1[O1_idx] = 0;
                }else{
                    std::vector<Voxel> adjacent_primary_points = getAdjacentPrimaryPoints(p);
                    if(adjacent_primary_points.size()==0){
                        O1[O1_idx] = 0;
                    }else{
                        float sum = 0;
                        for(auto& p: adjacent_primary_points){
                            sum += O1[p.getIdx(O1_shape)];
                        }
                        O1[O1_idx] = sum / adjacent_primary_points.size();
                    }
                }
            }
        }
    }

}


float* FuzzyObject3D::Excute()
{
    step_A1();
    step_A2();
    return O1;
}

std::array<int, 3> FuzzyObject3D::getO1Shape()
{
    return {O1_shape[0], O1_shape[1], O1_shape[2]};
}





