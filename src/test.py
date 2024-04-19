import cpp_cuda_functions as ccf
import numpy as np
import time
import SimpleITK as sitk    

T = np.float32
# T = np.float64
SIZE = (10, 10, 10)
# SIZE = (2,2,2)

def test_add():
    print('### Test addNp ###')
    a = np.random.rand(*SIZE).astype(T) * 10
    b = np.random.rand(*SIZE).astype(T) * 10

    res_np = a + b 
    res_cuda = ccf.addNp(a, b, 0) 
    allclose = np.allclose(res_cuda, res_np)
    print(f'allclose: {allclose}')
    if not allclose:
        print(f'res_np: \n{res_np}')
        print(f'res_c: \n{res_cuda}')

def test_padding():
    print('### Test padding ###')
    # padding
    SIZE = (3,3,3)
    a = np.random.rand(*SIZE).astype(T)
    pad_val = 0.3
    pad_size = 2
    res_cuda = ccf.padding3D(a, pad_val, pad_size, pad_size, pad_size)
    res_np = np.pad(a, pad_size, mode='constant', constant_values=pad_val) # warm up
    print(f'allclose: {np.allclose(res_cuda, res_np)}')

def test_conv():
    print('### Test conv3D ###')
    def conv3D_np(input, kernel):
        padding = (kernel.shape[0] -1 )// 2
        img = np.pad(input, padding, mode='constant', constant_values=0)
        res = np.zeros_like(input)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    res[i,j,k] = np.sum(img[i:i+kernel.shape[0], j:j+kernel.shape[1], k:k+kernel.shape[2]] * kernel)
        return res
    #

    A_SIZE = (10,100,100)
    A = np.random.rand(*A_SIZE).astype(T)
    K_SIZE = (5,5,5)
    kernel = np.random.rand(*K_SIZE).astype(T)

    t0 = time.time()
    res_cuda = ccf.cudaConvTest3D(A, kernel, 0)
    cuda_time = time.time() - t0
    t0 = time.time()
    res_np = conv3D_np(A, kernel)
    np_time = time.time() - t0
    print(f'allclose: {np.allclose(res_cuda, res_np)}')
    print(f'cuda_time: {cuda_time:.6f}, np_time: {np_time:.6f}')


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
        min=abs(e[i])
        for j in range(i+1,n):
            if abs(e[j])<abs(min):
                min=abs(e[j])
                #交换特征值
                tmp=e[i]
                e[i]=e[j]
                e[j]=tmp
                #交换特征向量
                r=np.copy(Q[:,i])
                Q[:,i]=Q[:,j]
                Q[:,j]=r
    return [e,Q]

def test_qr():

    print('### Test QR Split ###')

    A = np.array([[1,2,3],
                [2,3,4],
                [3,4,5]]).astype(np.float32)
    print(f'A: \n{A}')
    qr_py = qrSplit(A)
    qr_cpu = ccf.qrSplitTest3x3(A.copy(), -1)
    qr_cuda = ccf.qrSplitTest3x3(A.copy(), 0)
    qr_np = np.linalg.qr(A.copy())

    def compareQRSplit(a, b, name):
        # if np.allclose(a, b):
        if np.allclose(a, b, 1e-6, 1e-6):
            print(f'{name} allclose')
        else:
            print(f'{name} not allclose: \n{a} \n{b}')

    ## py
    A_py = np.dot(qr_py[0], qr_py[1])
    print(f'py : \n{A_py}')

    A_cpu = np.dot(qr_cpu[0], qr_cpu[1])
    print(f'cpu: \n{A_cpu}')

    A_cuda = np.dot(qr_cuda[0], qr_cuda[1])
    print(f'cuda: \n{A_cuda}')

    compareQRSplit(qr_py[0], qr_cpu[0], 'cpu Q')
    compareQRSplit(qr_py[1], qr_cpu[1], 'cpu R')

    compareQRSplit(qr_py[0], qr_cuda[0], 'cuda Q')
    compareQRSplit(qr_py[1], qr_cuda[1], 'cuda R')

    # compareQRSplit(qr_py[0], qr_np[0], 'np Q')
    # compareQRSplit(qr_py[1], qr_np[1], 'np R')

def test_qr_eigen():
    print('### Test QR Eigens ###')

    A = np.random.rand(3**2).reshape(3,3)
    A = np.triu(A)
    A += A.T - np.diag(A.diagonal())
    print(A)

    eig_np = np.linalg.eig(A.copy())
    eig_py = qrEgis(A.copy())
    eig_cpu = ccf.qrEigensTest3x3(A.copy(),-1, vecType=0)
    eig_cuda = ccf.qrEigensTest3x3(A.copy(), 0, vecType=0)
    print(f'np eig: \n{eig_np[0]}')
    print(f'py eig: \n{eig_py[0]}')
    print(f'cpu eig: \n{eig_cpu[0]}')
    print(f'cuda eig: \n{eig_cuda[0]}')

    print(f'np eig vec: \n{eig_np[1]}')
    print(f'py eig vec: \n{eig_py[1]}')
    print(f'cpu eig vec: \n{eig_cpu[1]}')
    print(f'cuda eig vec: \n{eig_cuda[1]}')

def test_hessian_eigen():
    print('### Test Hessian Eigens ###')
    A = np.random.rand(3**2).reshape(3,3)
    A = np.triu(A)
    A += A.T - np.diag(A.diagonal())
    print(A)

    res = ccf.hessianEigensTest3x3(A.copy())
    eig_cuda = ccf.qrEigensTest3x3(A.copy())
    eig_np = np.linalg.eig(A.copy())

    print(f'hessian eig: \n{res["eigenValues"]}')
    print(f'hessian eig vec: \n{res["eigenVectors"]}')

    print(f'np eig: \n{eig_np[0]}')
    print(f'np eig vec: \n{eig_np[1]}')

    print(f'cuda eig: \n{eig_cuda[0]}')
    print(f'cuda eig vec: \n{eig_cuda[1]}')


def callBack(i, N):
    print(f'progress: {i}/{N}')

def frangiTest():
    img_path = r'/home/HDD-16T-2022/sunxiao/temp/lung_img.nii.gz'
    res_path = r'/home/HDD-16T-2022/sunxiao/temp/lung_frangi.nii.gz'

    image = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    VEC_TYPE = 0
    t0 = time.time()
    frangi = ccf.frangi3D(img, cudaDimBlock = [6,6,6], eigenVectorType = VEC_TYPE,
                         verbose = 1,
                         progressCallback_i_N = callBack,
                         )
    
    print(f'frangi time: {time.time() - t0:.6f}')

    res_arr = frangi['frangi'].astype(np.float32)
    res = sitk.GetImageFromArray(res_arr)
    res.SetSpacing(spacing)
    res.SetOrigin(origin)
    res.SetDirection(direction)
    sitk.WriteImage(res, res_path)

    if VEC_TYPE>0:
        vec = frangi['vectors']
        print(vec.shape)
        for i in range(3):
            vec_arr = vec[:,:,:,i]
            vec_img = sitk.GetImageFromArray(vec_arr)
            vec_img.SetSpacing(spacing)
            vec_img.SetOrigin(origin)
            vec_img.SetDirection(direction)
            sitk.WriteImage(vec_img, res_path.replace('.nii.gz', f'_vec{i}.nii.gz'))



if __name__=='__main__':
    
    ccf.printDeviceInfo()

    # test_add()
    # test_padding()
    # test_qr()
    # test_qr_eigen()

    # test_conv()

    # test_hessian_eigen()

    frangiTest()
    

