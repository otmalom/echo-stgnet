import torch
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import os.path as osp

def cal_list_corr(pr_list, gt_list):
    pr_array = np.array(pr_list)
    gt_array = np.array(gt_list)
    corr = pearsonr(pr_array, gt_array)
    return corr[0]

def cal_corr(T):
    corr_metrix = np.zeros((T.shape[1],T.shape[1]))
    for idx_col_1 in range(T.shape[1]):
        col_1 = T[:,idx_col_1]
        for idx_col_2 in range(idx_col_1, T.shape[1]): # 自环包含当前col
            col_2 = T[:,idx_col_2]
            corr = cal_list_corr(col_1, col_2) 
            corr_metrix[idx_col_1,idx_col_2] = corr
            corr_metrix[idx_col_2,idx_col_1] = corr
    return corr_metrix

def save_feature(data_array, save_dir):
    plt.figure()
    plt.imshow(data_array)
    plt.savefig(save_dir)
    plt.close()

def neighborhood_construction(pts_dir, save_dir, npts):
    T_x = []
    T_y = []
    with open(pts_dir, "r") as f:
        n=0
        for line in f:
            n+=1
            data = line.strip().split(' ')
            filename = data[-1]
            single_img_x = [int(data[i]) for i in range(0,len(data)-1,2)]
            single_img_y = [int(data[i]) for i in range(1,len(data)-1,2)]
            T_x.append(single_img_x)
            T_y.append(single_img_y)
        print("Num of images: {}".format(n))
    T_x = np.array(T_x)
    T_y = np.array(T_y)
    corr_metrix_x = cal_corr(T_x)
    corr_metrix_y = cal_corr(T_y)
    corr_metrix = (abs(corr_metrix_x)+abs(corr_metrix_y))/2

    save_feature(corr_metrix_x, osp.join(save_dir, f"corrmetrix_{npts}pts_x.png"))
    save_feature(corr_metrix_y, osp.join(save_dir, f"corrmetrix_{npts}pts_y.png"))
    save_feature(corr_metrix, osp.join(save_dir, f"corrmetrix_{npts}pts.png"))

    
    save_feature(corr_metrix, osp.join(save_dir, f"corrmetrix_shift_{npts}pts.png"))    
    save_feature(T_x, osp.join(save_dir, f"T_{npts}pts_x.png"))
    save_feature(T_y, osp.join(save_dir, f"T_{npts}pts_y.png"))
    return corr_metrix, corr_metrix_x, corr_metrix_y, T_x, T_y



if __name__ == "__main__":
    n_pt = 46
    top_list = [5]
    datasets = ['camus_4ch','camus_2ch','dynamic']
    for dataset in datasets:
        
        pts_dir = f"{dataset}/Points/{n_pt}pts/points.txt"
        metrix_save_dir = f"{dataset}/Metrix/"
        os.makedirs(f'{metrix_save_dir}/corr', exist_ok=True)
        os.makedirs(f'{metrix_save_dir}/adj', exist_ok=True)
        os.makedirs(f'{dataset}/Points/{n_pt}pts', exist_ok=True)
        corr_metrix, corr_metrix_x, corr_metrix_y, T_X, T_Y = neighborhood_construction(pts_dir, f'{metrix_save_dir}/corr', n_pt)
        for top_k in top_list:
            with open(f"{dataset}/Points/{n_pt}pts/adjmetrix_top{top_k}.txt","w+") as adj_f:
                metrix_M = np.zeros((n_pt,n_pt))
                for cur_pt_idx in range(corr_metrix.shape[0]):
                    col = corr_metrix[cur_pt_idx,:]
                    corr_sort = sorted(range(len(list(col))),key=lambda k: list(col)[k], reverse=True)
                    # top_sort表示检索的位置，对应的数值是从大到小排列的。比如列表[5,4,1,3]中，最大值的索引是0，最小值的索引是2，所以top_sort是[0,1,3,2]
                    top_corr_sort = corr_sort[:top_k]
                    # 直接取top_k就可以了
                    for top_pt_idx in top_corr_sort:
                        metrix_M[cur_pt_idx, top_pt_idx]=1
                        # top_pt_idx = np.where(np.array(top_sort)==top_i)[0][0]
                        if top_pt_idx ==cur_pt_idx:
                            continue
                        adj_f.write("{} {} {}\n".format(cur_pt_idx, top_pt_idx, col[top_pt_idx]))
                    #         cmp_f.write("{},".format(top_pt_idx))
                    # cmp_f.write("\n")
                save_feature(metrix_M, f"{metrix_save_dir}/adj/adjmetrix_{n_pt}pts_top{top_k}.png")
