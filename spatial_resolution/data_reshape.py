import numpy as np
from module_io import *
from skimage.transform import resize

def split_into_patches(array, patch_size, overlap_ratio=0):
    """
    3차원 배열을 입력받아 패치 단위로 나누는 함수.
    
    Parameters:
    array: numpy.ndarray
        입력 배열 (N, H, W) 형태
    patch_size: tuple
        (patch_height, patch_width) 형태의 패치 크기
    overlap_ratio: float
        패치 간 겹침 비율 (0~1 사이 값)
        
    Returns:
    numpy.ndarray
        패치로 나뉜 배열 (N*n_patches, patch_height, patch_width) 형태
    """
    
    if len(array.shape) != 3:
        raise ValueError("입력 배열은 3차원이어야 합니다.")
        
    N, H, W = array.shape
    ph, pw = patch_size
    
    if ph > H or pw > W:
        raise ValueError("패치 크기가 입력 배열보다 큽니다.")
        
    # stride 계산 (overlap 고려)
    stride_h = int(ph * (1 - overlap_ratio))
    stride_w = int(pw * (1 - overlap_ratio))
    
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("overlap_ratio가 너무 큽니다.")
    
    # 패치 개수 계산
    n_patches_h = (H - ph) // stride_h + 1
    n_patches_w = (W - pw) // stride_w + 1
    
    # 결과 배열 초기화
    total_patches = n_patches_h * n_patches_w
    result = np.zeros((N * total_patches, ph, pw))
    
    # 패치 추출
    for i in range(N):
        patch_idx = 0
        for h in range(0, H - ph + 1, stride_h):
            for w in range(0, W - pw + 1, stride_w):
                result[i * total_patches + patch_idx] = array[i, h:h+ph, w:w+pw]
                patch_idx += 1
                  
    return result

def resizing(array,op):

    n, nx, ny = array.shape
    arr_re=np.zeros((n,nx*2,ny*2))
    for i in range(len(array)):
        if op==0:
            arr_re[i,:nx*2-1,:ny*2-1]=resize(array[i],(nx*2-1,ny*2-1),mode='symmetric', preserve_range=True, order=3)
        else:
            arr_re[i,:nx*2-1,:ny*2-1]=resize(array[i],(nx*2-1,ny*2-1),mode='constant', preserve_range=True, order=0, anti_aliasing=False)
        arr_re[i,nx*2-1,:ny*2-1]=arr_re[i,nx*2-2,:ny*2-1]
        arr_re[i,:nx*2,ny*2-1]=arr_re[i,:nx*2,ny*2-1]

    return arr_re

def sampling(array):
    n, nx, ny = array.shape
    # nx와 ny를 한 개씩 건너뛰어 샘플링
    sampled_array = array[:, ::2, ::2]
    return sampled_array

d1=np.load('s_train.npy')
d2=np.load('s0_val.npy')
e1=np.load('e_train.npy')
e2=np.load('e0_val.npy')

dp1=split_into_patches(d1, patch_size=(128,128), overlap_ratio=0)
dp2=split_into_patches(d2, patch_size=(128,128), overlap_ratio=0)
ep1=split_into_patches(e1, patch_size=(128,128), overlap_ratio=0)
ep2=split_into_patches(e2, patch_size=(128,128), overlap_ratio=0)

np.save('s1.npy', dp1)
np.save('s1_val.npy', dp2)
np.save('e1.npy', ep1)
np.save('e1_val.npy', ep2)

dr1=resizing(d1,0)
dr2=resizing(d2,0)
er1=resizing(e1,1)
er2=resizing(e2,1)

np.save('s2.npy', dr1)
np.save('s2_val.npy', dr2)
np.save('e2.npy', er1)
np.save('e2_val.npy', er2)
to_bin('test1.bin',dr1[0])

dpr1=resizing(dp1,0)
dpr2=resizing(dp2,0)
epr1=resizing(ep1,1)
epr2=resizing(ep2,1)

np.save('s3.npy', dpr1)
np.save('s3_val.npy', dpr2)
np.save('e3.npy', epr1)
np.save('e3_val.npy', epr2)
to_bin('test2.bin',dpr1[0])

drp1=split_into_patches(dr1, patch_size=(256,256), overlap_ratio=0)
drp2=split_into_patches(dr2, patch_size=(256,256), overlap_ratio=0)
erp1=split_into_patches(er1, patch_size=(256,256), overlap_ratio=0)
erp2=split_into_patches(er2, patch_size=(256,256), overlap_ratio=0)

np.save('s4.npy', drp1)
np.save('s4_val.npy', drp2)
np.save('e4.npy', erp1)
np.save('e4_val.npy', erp2)

dpo1=split_into_patches(d1, patch_size=(128,128), overlap_ratio=0.5)
dpo2=split_into_patches(d2, patch_size=(128,128), overlap_ratio=0.5)
epo1=split_into_patches(e1, patch_size=(128,128), overlap_ratio=0.5)
epo2=split_into_patches(e2, patch_size=(128,128), overlap_ratio=0.5)

np.save('s5.npy', dpo1)
np.save('s5_val.npy', dpo2)
np.save('e5.npy', epo1)
np.save('e5_val.npy', epo2)

dpo1=split_into_patches(d1, patch_size=(64,64), overlap_ratio=0.5)
dpo2=split_into_patches(d2, patch_size=(64,64), overlap_ratio=0.5)
epo1=split_into_patches(e1, patch_size=(64,64), overlap_ratio=0.5)
epo2=split_into_patches(e2, patch_size=(64,64), overlap_ratio=0.5)

np.save('s6.npy', dpo1)
np.save('s6_val.npy', dpo2)
np.save('e6.npy', epo1)
np.save('e6_val.npy', epo2)

d641=split_into_patches(d1, patch_size=(64,64), overlap_ratio=0.5)
d642=split_into_patches(d2, patch_size=(64,64), overlap_ratio=0.5)
e641=split_into_patches(e1, patch_size=(64,64), overlap_ratio=0.5)
e642=split_into_patches(e2, patch_size=(64,64), overlap_ratio=0.5)

np.save('s7.npy', d641)
np.save('s7_val.npy', d642)
np.save('e7.npy', e641)
np.save('e7_val.npy', e642) 

d128s=sampling(d1)
d128v=sampling(d2)
e128s=sampling(e1)
e128v=sampling(e2)

np.save('s8.npy', d128s)
np.save('s8_val.npy', d128v)
np.save('e8.npy', e128s)
np.save('e8_val.npy', e128v)
