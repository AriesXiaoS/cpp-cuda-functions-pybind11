
#include "mso.h"
#include "A_star.h"
#include <iostream>
#include <cmath>

#include "../utils/progress_bar.h"

using namespace progresscpp;

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
            std::vector<Voxel> Nlp = getNlp(p, 1);
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
        
        VoxelAStar astar_ = VoxelAStar(img_O, spacing, shape);
        astar_.initStartEnd(p, q);
        astar_.Update();
        float dist = astar_.getDistance();

        // float dist = sqrt(
        //     pow(spacing[0]*(p.z-q.z), 2) +
        //     pow(spacing[1]*(p.y-q.y), 2) +
        //     pow(spacing[2]*(p.x-q.x), 2)
        // )*0.5*(fdt[p.idx]+fdt[q.idx]);


        if(dist < min_dist){
            min_dist = dist;
            min_voxel = q;
        }

        astar_.freeList();
    }
    return fdt[min_voxel.idx];
}
void MSO3D::normalizeFDT()
{

    std::cout << "normalizeFDT" << std::endl;
    ProgressBar progressBar(size, 70);
    for(int i=0; i<size; i++){
        fdt_normed[i] = fdt[i] / getLocalScale(Voxel(i, shape));
        // std::cout << "\r" << i << "/" << size;
        ++progressBar;
        progressBar.display();
    }
    progressBar.done();
}
float* MSO3D::getNormedFDT()
{
    return fdt_normed;
}





void MSO3D::Excute()
{
    computeSmax();
    // normalizeFDT();
}


