#include "A_star.h"





VoxelAStar::VoxelAStar(float* arr, std::array<float, 3> input_spacing, 
                        std::array<int, 3> input_shape):
    img(arr), spacing(input_spacing), shape(input_shape)
{
    // cout << "A star init" << endl;
}


void VoxelAStar::initStartEnd(Voxel start, Voxel end)
{
    start_voxel = start;
    end_voxel = end;
}

bool VoxelAStar::isInOpenList(AStarPoint *p)
{
    for(auto q : open_list){
        if(*p == *q) return true;
    }
    return false;
}
bool VoxelAStar::isInCloseList(AStarPoint *p)
{
    for(auto q : close_list){
        if(*p == *q) return true;
    }
    return false;
}

int VoxelAStar::hanldOnePoint(AStarPoint *p)
{
    // 对当前方格的 26 个相邻方格的每一个方格
    for(int dz=-1; dz<=1; dz++){
        for(int dy=-1; dy<=1; dy++){
            for(int dx=-1; dx<=1; dx++){
                // 跳过中心点
                if(dz==0 && dy==0 && dx==0) continue;
                // 计算相邻方格的坐标
                int z = p->z + dz;
                int y = p->y + dy;
                int x = p->x + dx;
                // 判断相邻方格是否出界
                if(z>=0 && z<shape[0]
                && y>=0 && y<shape[1]
                && x>=0 && x<shape[2]){
                    //q 是相邻方格
                    Voxel v = Voxel(z, y, x, shape);
                    AStarPoint *q = new AStarPoint(v);
                    if(isInCloseList(q) || img[q->idx]==0){
                        delete q;
                        continue;
                    }
                    //
                    float dG = sqrt(
                        pow(dz*spacing[0], 2) +
                        pow(dy*spacing[1], 2) +
                        pow(dx*spacing[2], 2)
                    ) * 0.5 * (img[p->idx] + img[q->idx]);
                    if(!isInOpenList(q)){ // q 不在 open list 中
                        q->setParent(p);
                        float H = sqrt(
                            pow((end_voxel.z - q->z)*spacing[0], 2) +
                            pow((end_voxel.y - q->y)*spacing[1], 2) +
                            pow((end_voxel.x - q->x)*spacing[2], 2)
                        ) * 0.5 * (img[q->idx] + img[end_voxel.idx]);
                        q->setGH(p->G + dG, H);
                        open_list.push_back(q);
                    }else{ // q 在 open list 中
                        float newG = p->G + dG;
                        if(newG < q->G){
                            q->setParent(p);
                            q->setG(newG);
                        }
                    }
                    //
                    if(v==end_voxel){
                        close_list.push_back(q);
                        return 1;
                    }
                }
                //
            }
        }
    }
    return 0;
}



void VoxelAStar::Update()
{
    // init
    freeList();
    AStarPoint* s = new AStarPoint(start_voxel);
    open_list.push_back(s);
    //
    int result_found = 0;
    while(true){
        // open list 最小F
        float min_F = open_list[0]->F;
        int min_F_i = 0;
        for(int i = 1; i < open_list.size(); i++){
            if(open_list[i]->F < min_F){
                min_F = open_list[i]->F;
                min_F_i = i;
            }
        }
        //从open list中删除 加入close list
        AStarPoint *p = open_list[min_F_i];
        close_list.push_back(p);
        open_list.erase(open_list.begin() + min_F_i);
        // 判断是否到达终点
        if(hanldOnePoint(p)==1){ // 已找到终点
            result_found = 1;
            break;
        }
        if(open_list.size()==0){ // open list 为空 没有路径
            break;
        }
    }
    
    if(result_found==1){
        result_path.clear();
        // 从终点开始回溯
        AStarPoint *p = close_list[close_list.size()-1];
        result_distance = p->G;
        result_path.push_back(Voxel(p->z, p->y, p->x, p->idx));
        while(p->parent!=nullptr){
            p = p->parent;
            result_path.insert(result_path.begin(), Voxel(p->z, p->y, p->x, p->idx));
        }
    }else{
        result_path.clear();
        result_distance = -1;
    }

    // freeList();    
}

void VoxelAStar::freeList()
{
    if(open_list.size()==0 && close_list.size()==0) return;

    for (AStarPoint* point : open_list) {
        delete point;
    }

    for (AStarPoint* point : close_list) {
        delete point;
    }
    
    open_list.clear();
    close_list.clear();
}

std::vector<std::vector<int>> VoxelAStar::getPathList()
{
    std::vector<std::vector<int>> result;
    for(auto p : result_path){
        std::vector<int> v = {p.z, p.y, p.x};
        result.push_back(v);
    }
    return result;
}

float VoxelAStar::getDistance()
{
    return result_distance;
}




