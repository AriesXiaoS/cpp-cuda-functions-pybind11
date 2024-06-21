#include "fdt.h"






std::vector<Voxel> getNeighborsVoxel(Voxel p, std::array<int, 3> shape)
{
    std::vector<Voxel> neighbors(26); // 除去中心点，周围共有26个邻居
    int index = 0;
    
    // 遍历周围的26个邻居（排除中心点）
    for(int dz = -1; dz <= 1; ++dz){
        for(int dy = -1; dy <= 1; ++dy){
            for(int dx = -1; dx <= 1; ++dx){
                // 跳过中心点
                if(dz == 0 && dy == 0 && dx == 0) continue;
                if(p.z+dz>=0 && p.z+dz<shape[0] 
                && p.y+dy>=0 && p.y+dy<shape[1] 
                && p.x+dx>=0 && p.x+dx<shape[2]){
                    neighbors[index++] = Voxel(p.z+dz, p.y+dy, p.x+dx);
                }else{
                    neighbors[index++] = Voxel(false);
                }
            }
        }
    }

    return neighbors;
}

















