import os
import sys
import cv2
import math
import numpy as np
from einops import rearrange, reduce
import copy
import argparse


def DKr(a, b, h, w):
    ave_a = a.sum()/(h*w)-128
    ave_b = b.sum()/(h*w)-128
    M_a = 0
    M_b = 0
    a1 = a.flatten()
    b1 = b.flatten()
    histA = cv2.calcHist([a1],[0],None,[256],[0,255])
    histB = cv2.calcHist([b1],[0],None,[256],[0,255])
    for y in range(256):
        M_a += float(abs(y-128-ave_a))*histA[y]/(w*h)
        M_b += float(abs(y -128- ave_b))* histB[y]/(w *h)
    r = math.sqrt(M_a*M_a+M_b*M_b)
    D = math.sqrt(ave_a*ave_a+ave_b*ave_b)
    try:
        K = (D-r)/r
    except:
        K = 0
    return D, K, r

def calculate_L(l, alpha):
    l1 = (l/255*100).astype(np.uint8) 
    l1 = l1.flatten()
    histL = cv2.calcHist([l1],[0],None,[100],[0,100])
    max_S = max(histL)
    histL[np.where(histL<(max_S*alpha))]= 0
    yita = len(np.where(histL>0)[0])*0.01
    return yita

# 图像纹理丰富度
def calculate_sigma2(lab):
    imGray = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2GRAY)         
    var_gray  = np.var(imGray)
    return var_gray

def block(block_h,block_w, image, m, n):
    img_block = rearrange(image,'(m block_h) (n block_w) c-> (m n) block_h block_w c',block_h=block_h,block_w=block_w)
    img_block = img_block.astype(np.float16)
    averaged = reduce(img_block, '(m n) block_h block_w c -> m n c', 'mean',m=m,n=n)
    averaged = averaged.astype(np.uint8)
    return averaged

def NNO(h, w, image):
    m, n = 64, 64
    h_new = h-(h%m)
    w_new = w-(w%n)
    image = image[0:h_new,0:w_new]
    block_h = h_new//m
    block_w = w_new//n
    averaged = block(block_h,block_w,image, m, n)
    lab = cv2.cvtColor(averaged,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    h, w = l.shape
    l1 = (l/255*100).astype(np.uint8) 
    L_area = np.zeros((h,w))
    L_area[np.where(l1>=20)and np.where(l1<=95)] = 1
    
    abvalue = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            aij = (a[i,j]-128)*(a[i,j]-128)
            bij = (b[i,j]-128)*(b[i,j]-128)
            abvalue[i,j] = math.sqrt(aij*aij+bij*bij)
    F = abvalue.sum()
    T = 0
    for i in range(m):
        for j in range(n):
            T += ((F-abvalue[i,j])/((m*n-1)*F))*abvalue[i,j]
    T = 1/2 * T
    mask = np.zeros((h,w))     
    mask[np.where(abvalue<T)] = 1    
    NNO = L_area * mask
    image_copy = copy.deepcopy(averaged)
    image_copy[np.where(NNO!=1)] = 0
    lab_nno = cv2.cvtColor(image_copy,cv2.COLOR_BGR2LAB)
    l_nno, a_nno,b_nno = cv2.split(lab_nno)
    u,D_sigma,sigma = DKr(a, b, m, n)

    u_nno,D_sigma_nno,sigma_nno = DKr(a_nno,b_nno, m, n)
    try:
        D_sigma_cr = (D_sigma - D_sigma_nno)/D_sigma
        u_cr = (u - u_nno)/u
        sigma_cr = (sigma - sigma_nno)/sigma
    except:
        if D_sigma == 0:
            D_sigma_cr = 0
        if u == 0:
            u_cr = 0
        if sigma == 0:
            sigma_cr = 0
    return D_sigma_cr, u_cr, sigma_cr,D_sigma_nno

def check_cast(image, h, w, l, lab):
    L_yita = calculate_L(l,alpha=0.01)
    # print(L_yita*100,"%L_yita")
    omiga2 = calculate_sigma2(lab)
    # print(omiga2,"omiga2")  
    if L_yita < 0.8 and L_yita >= 0.5 and omiga2 <= 2500 or L_yita < 0.5 and omiga2 < 300: # 
        print("本质偏色")
        return False               
    else:
        D_sigma_cr, u_cr, sigma_cr,D_sigma_nno = NNO(h,w,copy.deepcopy(image))
        # print("K_nno,r_ch,D_ch,K_ch",D_sigma_nno,sigma_cr,u_cr,D_sigma_cr)
        if D_sigma_nno<-0.3 and sigma_cr>0.7 and u_cr>0.6 and D_sigma_cr>2:
            print("本质偏色2")
            return False
        else:
            print("真实偏色")
            return True

def get_args():
    parser = argparse.ArgumentParser('color bias', add_help=False)
    parser.add_argument('--path', default='', type=str)
    return parser.parse_args()




if __name__ == "__main__":
    opts = get_args()
    if opts.path != '':
        frame = cv2.imread(opts.path)
        h,w,_ = frame.shape
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b =  cv2.split(lab)
        # print(lab)
        u,D_sigma,sigma = DKr(a, b, h, w)
        is_bias = False

        if (u-sigma>9 and D_sigma>0.6) or D_sigma>1.5:
            print("初步偏色")
            is_bias = check_cast(frame, h, w, l, lab)
        else:
            print("初步正常")
            D_sigma_cr, u_cr, sigma_cr,D_sigma_nno = NNO(h,w,copy.deepcopy(frame))
            # print("D_sigma_cr, u_cr, sigma_cr,D_sigma_nno",D_sigma_cr, u_cr, sigma_cr,D_sigma_nno)
            if D_sigma_nno <-0.5 or (sigma_cr > 0.7 and u_cr>0.6) or D_sigma_cr<0.4:
                print("正常无偏色")
                is_bias = False
            else:
                if D_sigma_nno > 0.5 or (sigma_cr <= 0.5 and u_cr<=0.4) or D_sigma_cr>1:
                    print("初步正常后偏色")
                    is_bias = check_cast(frame, h, w, l, lab)
                else:
                    print("无法识别")
                    is_bias = False
        if is_bias:
            print("color bias")
        else:
            print("no color bias")
    
