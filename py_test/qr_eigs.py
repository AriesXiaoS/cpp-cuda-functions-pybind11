import numpy as np


#QR分解
def qrSplit(A):
    n=A.shape[0]#A的维度
    Q=[[]]
    R=A
    for i in range(0,n-1):
        B=R
        if i!=0:
            #删除一行一列,得n-1阶子阵
            B=B[i:,i:]
        #取第一列向量
        x=B[:,0]
        #向量摸长
        m=np.linalg.norm(x)
        #生成一个模长为m，其余项为0的向量y
        y=[0 for j in range(0,n-i)]
        y[0]=m
        #计算householder反射矩阵
        #w = (x-y)/||x-y||
        w=x-y
        w=w/np.linalg.norm(w)
        #H=E-2*WT*W
        H=np.eye(n-i)-2*np.dot(w.reshape(n-i,1),w.reshape(1,n-i))#H是个正交矩阵
        #第一次计算不需对正交正H升维
        if i==0: 
            #第一次迭代
            Q=H
            R=np.dot(H,R)
        else:
            #因为降了维度，所以要拼接单位矩阵
            D=np.c_[np.eye(i),np.zeros((i,n-i))]
            H=np.c_[np.zeros((n-i,i)),H]
            H=np.r_[D,H]
            #迭代计算正交矩阵Q和上三角R
            Q=np.dot(H,Q)
            R=np.dot(H,R)
    Q=Q.T
    return [Q,R]


#QR迭代求特征值特征向量
def  qrEgis(A):
    # QR迭代(尽量让它多迭代几次，以至于AK收敛为上三角)
    qr = []
    n = A.shape[0]  # A的维度
    Q = np.eye(n)
    for i in range(0, 100):
        # A=QR
        qr = qrSplit(A)
        # 将Q右边边累成
        Q = np.dot(Q,qr[0])
        # A1=RQ
        A = np.dot(qr[1], qr[0])

    AK = np.dot(qr[0], qr[1])
    #把e取出来
    e=[]
    for i in range(0,n):
        e.append(AK[i][i])
    #对特征值按升序排序，特征向量与其对应
    for i in range(0,n-1):
        min=e[i]
        for j in range(i+1,n):
            if e[j]<min:
                min=e[j]
                #交换特征值
                tmp=e[i]
                e[i]=e[j]
                e[j]=tmp
                #交换特征向量
                r=np.copy(Q[:,i])
                Q[:,i]=Q[:,j]
                Q[:,j]=r
    return [e,Q]



if __name__=='__main__':
    print('### 测试QR分解 ###')
    A=np.array([1,2,3,4,2,1,2,3,3,2,1,2,4,3,2,1])
    A=A.reshape(4,4)
    print('A原来的样子')
    print(A)
    qr = qrSplit(A)
    print('打印Q,R')
    print(qr[0])
    print(qr[1])
    print('打印Q*R')
    print(np.dot(qr[0],qr[1]))

    print('### 测试QR迭代求特征值特征向量 ###')
    A=np.array([1,2,3,4,2,1,2,3,3,2,1,2,4,3,2,1])
    A=A.reshape(4,4)
    egis =qrEgis(A)
    print('自己写的QR分解')
    print(egis[0])
    print('......')
    print(egis[1])
    print('numpy自带的分解器')
    e,u=np.linalg.eigh(A)
    print(e)
    print('.....')
    print(u)
























