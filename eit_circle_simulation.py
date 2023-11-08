import os
os.environ["OMP_NUM_THREADS"] = "1"             # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"        # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"             # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"      # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"         # export NUMEXPR_NUM_THREADS=1
import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
import platform

# pyEIT 2D algorithms modules
import tqdm
from pyeit.mesh import create, set_perm
import pyeit.eit.jac as jac
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax,circle
import argparse
from pyeit.eit.interp2d import meshgrid,weight_idw
import shelve
# xxx-------------------------------------------------
n_el,h0 = 16, 0.07
# n_el, h0 = 16, 0.03
background = 1.0
margin = 0.05
el_dist, step = 1, 1
seed = 2022
mesh_obj, el_pos = create(seed, n_el=n_el, fd=circle, h0=h0)
fwd = Forward(mesh_obj, el_pos)
ex_mat = eit_scan_lines(n_el, el_dist)

eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser="std")
eit.setup(p=0.25, lamb=1.0, method="lm")
# xxx-------------------------------------------------
def show_tripcolor(mesh_obj,ax=None,show=True,title=None,is_set_colorbar=False,vmin=0,vmax=2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    perm = mesh_obj["perm"]
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(perm),
                      shading="flat", alpha=1.0, cmap=plt.cm.viridis,
                      vmin=vmin,vmax=vmax)  #xxx
    if is_set_colorbar: plt.colorbar(im, ax=ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("equal")
    ax.set_title(title)
    ax.axis("off")
    if (ax is None) and show: plt.show()
    return im
def save_tripcolor(save_path, save_file_num, i, mesh_obj):
    show_tripcolor(mesh_obj, ax=None, show=False, title=None,is_set_colorbar=True)
    plt.savefig(os.path.join(save_path, f"{save_file_num}-{i}.png"))
    plt.close()
def save_u(save_path, save_file_num, i, u):
    # plot tripcolor shows values on nodes (shading='flat' or 'gouraud')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.plot(u)
    plt.savefig(os.path.join(save_path, f"{save_file_num}-{i}-interp-u.png"))
    plt.close()
def get_anomaly(margin=0.05,num_inclusion=2):
    """
    Args:
        margin:        inclusion边界距离
        num_inclusion: 数量

    Returns:
         anomaly：
        is_intersect：是否相交
    """
    if num_inclusion==2:
        x1, x2 = np.random.uniform(-0.6, 0.6, 2)
        y1, y2 = np.random.uniform(-0.6, 0.6, 2)
        r = 0.2 + np.random.rand(2) * 0.1
        assert 0.2 <= r.mean() <= 0.3
        anomaly = [
            {"x": x1, "y": y1, "d": r[0], "perm": 2},
            {"x": x2, "y": y2, "d": r[1], "perm": 0.5},
        ]
        is_intersect = (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r.sum() + margin) ** 2
    elif num_inclusion==3:
        x1, x2, x3 = np.random.uniform(-0.55, 0.55, 3)
        y1, y2, y3 = np.random.uniform(-0.55, 0.55, 3)
        r = 0.15 + np.random.rand(3) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly = [
            {"x": x1, "y": y1, "d": r[0], "perm": 2.0},
            {"x": x2, "y": y2, "d": r[1], "perm": 1.5},
            {"x": x3, "y": y3, "d": r[2], "perm": 0.5}  # perm随意设的
        ]
        is_intersect = ((x1 - x2) ** 2 + (y1 - y2) ** 2 < (r[0] + r[1] + margin) ** 2) or \
                       ((x1 - x3) ** 2 + (y1 - y3) ** 2 < (r[0] + r[2] + margin) ** 2) or \
                       ((x3 - x2) ** 2 + (y3 - y2) ** 2 < (r[2] + r[1] + margin) ** 2)
    elif num_inclusion==4:
        x1, x2, x3, x4 = np.random.uniform(-0.55, 0.55, 4)
        y1, y2, y3, y4 = np.random.uniform(-0.55, 0.55, 4)
        r = 0.15 + np.random.rand(4) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly = [
            {"x": x1, "y": y1, "d": r[0], "perm": 2.0},
            {"x": x2, "y": y2, "d": r[1], "perm": 1.5},
            {"x": x3, "y": y3, "d": r[2], "perm": 0.5},
            {"x": x4, "y": y4, "d": r[3], "perm": 0.3}  # perm随意设的
        ]
        is_intersect = ((x1 - x2) ** 2 + (y1 - y2) ** 2 < (r[0] + r[1] + margin) ** 2) or \
                       ((x1 - x3) ** 2 + (y1 - y3) ** 2 < (r[0] + r[2] + margin) ** 2) or \
                       ((x3 - x2) ** 2 + (y3 - y2) ** 2 < (r[1] + r[2] + margin) ** 2) or \
                       ((x4 - x1) ** 2 + (y4 - y1) ** 2 < (r[3] + r[0] + margin) ** 2) or \
                       ((x4 - x2) ** 2 + (y4 - y2) ** 2 < (r[3] + r[1] + margin) ** 2) or \
                       ((x4 - x3) ** 2 + (y4 - y3) ** 2 < (r[3] + r[2] + margin) ** 2)
    return anomaly, is_intersect
def main(ind,num,data_type):
    """
    Args:
        ind: index of save file
        num: number of samples
        n: resolution

    Returns:

    """
    np.random.seed(ind)            # 在create后，重新设置随机数种子
    time.sleep(np.random.rand()*3) #随机停几秒避免同时检查文件是否存在

    save_path = os.path.join(os.getcwd(),'datasets-600', data_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存训练数据要用的各种参数
    d = shelve.open(os.path.join(save_path,'hypers.db'))
    try:
        d['data'] = {'n_el':n_el,'h0':h0,'background':background,'margin':margin,
                     'el_dist':el_dist,'step':step,'seed':seed,'mesh_obj': mesh_obj, 'el_pos': el_pos,'ex_mat':ex_mat}
        d['fun'] = {'fwd':fwd, 'eit':eit}
    finally:
        d.close()

    save_file_num = str(ind)
    """ 1. setup  ref xxx"""
    i = 0
    xs,xs_gn,ys = [],[],[]
    pbar = tqdm.tqdm(total=num)
    while i < num:
        anomaly, is_intersect = get_anomaly(margin,num_inclusion=num_inclusion)
        if is_intersect: continue
        mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=background)      # background changed to values other than 1.0 requires more iterations
        # plot_interpolate(mesh_new, img)
        """ 2. calculate simulated data """
        try:  # 有时会出现error numpy.linalg.LinAlgError: Singular matrix,在出错的前提才画图
            f1 = fwd.solve_eit(ex_mat, step, perm=mesh_new["perm"], parser="std")
            xs.append(mesh_new['perm'])
        except Exception as e:
            print(repr(e))
            continue

        # 加噪音 xxx
        # y_clean = f1.v
        # y = y_clean+np.random.randn(208)*(np.max(y_clean)*0.005)  # 1%
        # y[y<0] = 0
        # ys.append(y)
        # x_gn, _ = eit.gn(y, lamb_decay=0.1, lamb_min=1e-5, maxiter=8, verbose=False)

        # 不加噪音
        ys.append(f1.v)
        x_gn, _ = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=8, verbose=False)
        # plt.plot(y_clean)
        # plt.plot(y)
        # plt.show()
        # 计算初始值 Gaussian-newton
        xs_gn.append(x_gn)

        # plot_interpolate(mesh_new,img_inv)
        i += 1
        pbar.update(1)
        if i%5==0:
            save_tripcolor(save_path, save_file_num, i, mesh_new)
            # save_u(save_path, save_file_num, i, f1.v)
    pbar.close()
    data_file = os.path.join(save_path,save_file_num+'.npz')
    np.savez(data_file,xs=np.array(xs),xs_gn=np.array(xs_gn),ys=np.array(ys))
    print(f'save data in {data_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate data')
    parser.add_argument('--n_inclusion', type=int, default=4,help='number of inclusion')
    parser.add_argument('--ind', type=int, default=12, help='index of saved file')
    parser.add_argument('--num', type=int, default=10, help='number of samples')
    args = parser.parse_args()

    num_inclusion = args.n_inclusion
    data_type = f'circle{num_inclusion}_h0{h0}_{platform.system()}'
    main(args.ind,args.num,data_type)
    # for i in range(52):
    #     main(i,args.num,data_type)

"""
# 提交多个后台任务
for n in [4]:
    for i in range(60): # linux
        print(f'nohup python -u  eit_circle_simulation.py --n_inclusion {n} --ind {i} --num 10  > log_{n}{i}.log 2>&1 &')
        # print(f'python -u  eit_circle_simulation.py --n_inclusion {n} --ind {i} --num 20  > log_{n}{i}.log 2>&1 &')
        
"""

