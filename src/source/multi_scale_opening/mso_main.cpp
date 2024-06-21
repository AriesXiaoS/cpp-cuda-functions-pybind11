
#include "mso.h"
#include "A_star.h"
#include <iostream>
#include <cmath>


MSO3D::MSO3D(float* arr, std::array<float, 3> input_spacing, 
                        std::array<int, 3> input_shape)
    : img_O(arr), spacing(input_spacing), shape(input_shape), 
    size(shape[0]*shape[1]*shape[2]), astar(arr, input_spacing, input_shape)
{

    // img_O = arr;
    // spacing = input_spacing;
    // shape = input_shape;
    // size = shape[0]*shape[1]*shape[2];

    isSmax_arr = new bool[size]{false};

    // astar = VoxelAStar(arr, input_spacing, input_shape);

    fdt_normed = new float[size]{0};
}

void MSO3D::computeFDT()
{
    FDT3D fdt3d = FDT3D(img_O, spacing, shape);
    fdt = fdt3d.Excute();
}
void MSO3D::setFDT(float* arr)
{
    fdt = arr;
}

//
std::vector<Voxel> MSO3D::getNlp(Voxel p, int l)
{
    std::vector<Voxel> res;
    for(int i=p.z-l; i<=p.z+l; i++){
        for(int j=p.y-l; j<=p.y+l; j++){
            for(int k=p.x-l; k<=p.x+l; k++){
                if(i>=0 && i<shape[0] && j>=0 && j<shape[1] && k>=0 && k<shape[2]){
                    res.push_back(Voxel(i, j, k, shape));
                }
            }
        }
    }
    return res;
}
// Smax
void MSO3D::computeSmax()
{
    Smax.clear();
    for(int i=0; i<size; i++){
        if(img_O[i]>0){
            Voxel p = Voxel(i, shape);
            std::vector<Voxel> Nlp = getNlp(p, 2);
            bool isSmax = true;
            for(auto& q : Nlp){
                if(fdt[q.idx] > fdt[p.idx]){
                    isSmax = false;
                    break;
                }
            }
            if(isSmax){
                Smax.push_back(p);
                isSmax_arr[i] = true;
            }
        }
    }
}



// NPO
float MSO3D::getLocalScale(Voxel p)
{
    float min_dist = 1e10;
    Voxel min_voxel = Smax[0];
    for(auto q : Smax){
        if(q == p){
            return fdt[q.idx];
        }
        astar.initStartEnd(p, q);
        astar.Update();
        float dist = astar.getDistance();
        if(dist < min_dist){
            min_dist = dist;
            min_voxel = q;
        }
    }
    return fdt[min_voxel.idx];
}
void MSO3D::normalizeFDT()
{
    for(int i=0; i<size; i++){
        fdt_normed[i] = fdt[i] / getLocalScale(Voxel(i, shape));
    }
}
float* MSO3D::getNormedFDT()
{
    return fdt_normed;
}





void MSO3D::Excute()
{
    computeSmax();
    normalizeFDT();
}


