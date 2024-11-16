import os
import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
import platform

import torch
import tqdm
import argparse

def nonlinear_convolution(W1,W2,size,steps,x,args):
    """
       y = a*x^T W1 x +W2 x
    Args:
        W1:
        W2:
        x:
    Returns:
       input x (x1,...,xn)
       ouput y (y1,...,ym)
    """
    a = args.a
    s = size
    w = s//2
    y = np.zeros(len(steps))
    for i, s in enumerate(steps):
        x_temp = x[s - w:s + w + 1]
        y_temp = a * x_temp.T @ W1 @ x_temp + W2.T @ x_temp
        y[i] = y_temp
    return y


def main(W1,W2,size,steps,args,ind):
    a = args.a
    n = args.dim_x
    m = len(steps)
    num = args.num

    np.random.seed(ind)              
    time.sleep(np.random.rand()*3)   

    save_file_num = str(ind)
    data_type = f'a{a:.1f}'
    save_path = os.path.join(os.getcwd(),'datasets-toy-nonlinear', data_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hyper_file = os.path.join(save_path, 'hypers')
    if not os.path.exists(hyper_file):
        np.savez(hyper_file,
                 W1=W1,W2=W2,size=size,steps=steps,
                 y_dim=m,x_dim=n,n_samples=num,a=a)

    scale = 0.5
    xs = np.zeros((n,num))
    cnt = 0
    while cnt<num:
        temp = [np.random.randn()]
        for j in range(n-1):
            x = np.random.laplace(temp[-1], scale, 1)[0]
            temp.append(x)
        if (np.array(temp)>0).all():
            temp = np.array(temp)
            temp = (temp-temp.min())/(temp.max()-temp.min())
            temp = temp*2-1
            xs[:,cnt] = temp
            cnt += 1
 
    ys = np.zeros((m, num))
    for i in range(num):
        y = nonlinear_convolution(W1,W2,size,steps,xs[:,i],args)
        # y = y+np.max(y)*0.02*np.random.randn(m)
        ys[:,i] = y


    data_file = os.path.join(save_path,save_file_num+'.npz')
    np.savez(data_file,xs=np.array(xs),xs_inv=np.array(xs_inv),ys=np.array(ys))
    print(f'save data in {data_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate data')
    parser.add_argument('--a', type=float, default=1.5, help='power of x')
    parser.add_argument('--ind', type=int, default=22, help='index of saved file')
    parser.add_argument('--num', type=int, default=500, help='number of samples')
    parser.add_argument('--dim_x', type=int, default=53, help='number of dimension of x')
    args = parser.parse_args()
    print(args)

    size = 9
    stride = 4
    steps = np.arange(size // 2, 50, stride)

    W1 = np.abs(np.random.rand(size, size))
    for i in range(size):
        for j in range(size):
            if i>j: W1[i,j]=0
    W1 = W1/np.sum(W1)
    plt.imshow(W1)
    plt.show()

    W2 = np.abs(np.random.randn(size))
    W2 = W2/np.sum(W2)
    # plt.plot(W2)
    # plt.show()
    for ind in range(24):
        main(W1,W2,size,steps,args,ind)

