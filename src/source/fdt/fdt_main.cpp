
#include "common.h"
#include "define.h"
#include "utils.h"

#include "fdt.h"


#include <typeinfo>
#include <chrono>   
using namespace chrono;

template <typename T>
py::array_t<T> fdt_3d(
    py::array_t<T> image, std::vector<float> spacing
)
{
    int dim = spacing.size();
    if(dim!=3) throw std::runtime_error("Spacing must be 3D.");
    // img
    auto img = image.request();    
    if(img.ndim != 3) throw std::runtime_error("Number of dimensions must be 3 but got " + to_string(img.ndim) );
    T* ptr = (T*) img.ptr;

    //
    FDT3D fdt = FDT3D(ptr, 
        std::array<float, 3>{spacing[0], spacing[1], spacing[2]},
        std::array<int, 3>{int(img.shape[0]), int(img.shape[1]), int(img.shape[2])}
    );
    float* res = fdt.Excute();

    auto result = py::array_t<T>(int(img.shape[0]) * int(img.shape[1]) * int(img.shape[2]), res);
    result.resize({int(img.shape[0]), int(img.shape[1]), int(img.shape[2])});
    
    return result;

}

template py::array_t<float> fdt_3d(
    py::array_t<float> image, std::vector<float> spacing
);

////////////////////////////////////

FDT3D::FDT3D(float* arr_O, std::array<float, 3> input_spacing_zyx, 
                            std::array<int, 3> input_shape)
{
    spacing_zyx = input_spacing_zyx;
    src_shape = input_shape;

    FuzzyObject3D fuzzy_obj = FuzzyObject3D(arr_O, src_shape);
    O1 = fuzzy_obj.Excute();
    O1_shape = fuzzy_obj.getO1Shape();

    getResolutionVector();

    //
    int size = O1_shape[0]*O1_shape[1]*O1_shape[2];
    fdt = new float[size]{0};
    for(int i=0; i<size; i++){
        if(O1[i] == 0){
            fdt[i] = 0;
        }else if(O1[i]>0){
            fdt[i] = MAX;
        }
    }

    threadMutex = new std::mutex();
}

void FDT3D::getResolutionVector()
{
    float O1_spacing_zyx[3];
    for(int i=0; i<3; i++){
        O1_spacing_zyx[i] = spacing_zyx[i] / 2;
    }
    int index = 0;
    for(int i=-1; i<=1; i++){
        for(int j=-1; j<=1; j++){
            for(int k=-1; k<=1; k++){
                if(i==0 && j==0 && k==0) continue;
                resolution_vec[index++] = sqrt(
                    pow(O1_spacing_zyx[0]*i, 2) + 
                    pow(O1_spacing_zyx[1]*j, 2) + 
                    pow(O1_spacing_zyx[2]*k, 2)
                );
            }
        }
    }

}

void FDT3D::initQ()
{
    for(int i=0;i<O1_shape[0];i++){
        for(int j=0;j<O1_shape[1];j++){
            for(int k=0;k<O1_shape[2];k++){
                //
                Voxel p = Voxel(i, j, k);
                if(O1[p.getIdx(O1_shape)] > 0){
                    // p ∈ (O′) such that N ∗(P) ∩ (O′) is non-empty
                    std::vector<Voxel> neighbors = getNeighborsVoxel(p, O1_shape);
                    for(auto& n: neighbors){
                        if(n.valid && O1[n.getIdx(O1_shape)] == 0){
                            Q.push(p);
                            Q0.push_back(p);
                            break;
                        }
                    }
                }
            }
        }
    }

}

float FDT3D::findDistMin(Voxel p)
{
    int p_idx = p.getIdx(O1_shape);
    float distMin = MAX;
    std::vector<Voxel> neighbors = getNeighborsVoxel(p, O1_shape);
    for(int i=0; i< neighbors.size(); i++){
        Voxel q = neighbors[i];
        if(q.valid){
            int q_idx = q.getIdx(O1_shape);
            float dis = fdt[q_idx] + resolution_vec[i]*(O1[p_idx]+O1[q_idx])/2;
            if(dis<distMin){
                distMin = dis;
            }
        }
    }
    return distMin;
}

void FDT3D::mainLoop_singleThread()
{
    while(Q.size()>0){
        Voxel p = Q.front();
        Q.pop();
        int p_idx = p.getIdx(O1_shape);
        float distMin = findDistMin(p);
        if(distMin < fdt[p_idx]){
            fdt[p_idx] = distMin;
            std::vector<Voxel> neighbors = getNeighborsVoxel(p, O1_shape);
            for(auto& n: neighbors){
                if(n.valid && O1[n.getIdx(O1_shape)] > 0){
                    Q.push(n);
                }
            }
        }
    }
}

int FDT3D::mainLoop_oneThread(Voxel p)
{
    threadMutex->lock();
    workingThreadNum++;
    threadMutex->unlock();
    // std::lock_guard<std::mutex> lock(*threadMutex);
    //
    int p_idx = p.getIdx(O1_shape);
    float distMin = findDistMin(p);
    if(distMin < fdt[p_idx]){

        threadMutex->lock();
        fdt[p_idx] = distMin;
        threadMutex->unlock();

        std::vector<Voxel> neighbors = getNeighborsVoxel(p, O1_shape);
        for(auto& n: neighbors){
            if(n.valid && O1[n.getIdx(O1_shape)] > 0){
                // Q1.push_back(n);
            threadMutex->lock();
                Q.push(n);
            threadMutex->unlock();
            }
        }
    }

    threadMutex->lock();
    workingThreadNum--;
    threadMutex->unlock();
    return 0;
}

void FDT3D::mainLoop_multiThread()
{
    ThreadPool pool(16); 
    std::vector<std::future<int>> futures; 
    int cnt=0;
    while(true){
        if(Q.size()>0){
            threadMutex->lock();
            Voxel p = Q.front();
            Q.pop();
            threadMutex->unlock();

            pool.enqueue(
                [this, p]{
                    mainLoop_oneThread(p);
                }
            );
        }
        if(Q.size()==0 && workingThreadNum==0) break;

        // futures.clear();
        // for(auto& p: Q0){
        //     futures.push_back(
        //         pool.enqueue(
        //             [this, p]{
        //                 return mainLoop_oneThread(p);
        //             }
        //         )
        //     );
        // }
        // // 等待所有任务完成
        // for(auto& future : futures) {
        //     future.get(); // 这会阻塞，直到对应的任务完成
        // }
        // cout << "cnt: " << cnt++ << endl;

        // Q0 = Q1;
        // Q1.clear();
    }

}

float* FDT3D::resampleBack()
{
    float* res = new float[src_shape[0]*src_shape[1]*src_shape[2]];
    for(int i=0; i<src_shape[0]; i++){
        for(int j=0; j<src_shape[1]; j++){
            for(int k=0; k<src_shape[2]; k++){
                Voxel p = Voxel(i, j, k);
                int p_idx = p.getIdx(src_shape);
                Voxel p1 = Voxel(i*2+1, j*2+1, k*2+1);
                int p1_idx = p1.getIdx(O1_shape);
                res[p_idx] = fdt[p1_idx];
            }
        }
    }
    return res;
}

float* FDT3D::Excute()
{
    initQ();
    mainLoop_singleThread();
    // mainLoop_multiThread();
    return resampleBack();
    // return fdt;
}


