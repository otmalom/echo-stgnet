import os
import os.path as osp
import numpy as np
import collections
import cv2
import sys

class MiscMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 9e15
        self.max = -9e15
        self.vals = []
        self.std = 0
    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = val if val < self.min else self.min
        self.max = val if val > self.max else self.max
        self.std = np.std(self.vals)

def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


def val_NME(pred, targ, num_points=46, mode='IP'):
    """Evaluation metric for face alignment, numpy version
    Args:
        pred (numpy - shape of batchsize x num_points*2 x 1): predicted face landmarks 
        targ (numpy - shape of batchsize x num_points*2 x 1): target face landmarks
        num_points (int): # face landmarks
        mode (string): 'IP' inter pupil  # 瞳距或者眼平均距离
                       'IO' inter ocular # 外眼角间距
    Return:
        (float): normalized mean error according to mode
        (numpy array):
        (int):
    """
    if num_points == 3 and mode == 'IO':
        le_idx = [2]
        re_idx = [0]
    elif num_points == 3 and mode == 'IP':
        le_idx = [2]
        re_idx = [0]

    if num_points == 10 and mode == 'IO':
        le_idx = [5,6,7,8]
        re_idx = [0,1,2,3]
    elif num_points == 10 and mode == 'IP':
        le_idx = [8]
        re_idx = [0]

    if num_points == 20 and mode == 'IO':
        le_idx = [9,10,11,12,13,14,15,16]
        re_idx = [0,1,2,3,4,5,6,7]
    elif num_points == 20 and mode == 'IP':
        le_idx = [16]
        re_idx = [0]

    if num_points == 32 and mode == 'IO':
        le_idx = [15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        re_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    elif num_points == 32 and mode == 'IP':
        le_idx = [28]
        re_idx = [0]

    if num_points == 46 and mode == 'IO':
        le_idx = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
        re_idx = [44,45,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    elif num_points == 46 and mode == 'IP':
        le_idx = [40]
        re_idx = [0]
        
    pred_vec2 = pred.reshape(-1, num_points, 2) # batchsize x 68 x 2
    targ_vec2 = targ.reshape(-1, num_points, 2)

    le_loc = np.mean(targ_vec2[:,le_idx,:], axis=1) # batchsize x 2
    re_loc = np.mean(targ_vec2[:,re_idx,:], axis=1)  

    norm_dist = np.sqrt(np.sum((le_loc - re_loc)**2, axis=1))  # batchsize
    mse = np.mean(np.sqrt(np.sum((pred_vec2 - targ_vec2)**2, axis=2)), axis=1) # batchsize

    nme = mse / norm_dist

    return np.mean(nme), nme

def val_metrics(pr_point_dict, gt_point_dict, num_points):
    NME = MiscMeter()
    with open('nme.txt','w+') as f:
    
        for patient, frame_pts in pr_point_dict.items():
            for frame, pr_pts in frame_pts.items():
                if frame in gt_point_dict[patient]:
                    gt_pts = gt_point_dict[patient][frame]
                else:
                    continue
                pr_pts = [float(p) for p in pr_pts]
                gt_pts = [float(p) for p in gt_pts]

                pr_pts = np.array([[np.array([pr_pts[i],pr_pts[i+1]]) for i in range(0,len(pr_pts),2)]])
                gt_pts = np.array([[np.array([gt_pts[i],gt_pts[i+1]]) for i in range(0,len(gt_pts),2)]])
                
                nme, nmes = val_NME(pr_pts, gt_pts, num_points)
                f.write(f'{nme} {patient}_{frame:03d}.png\n')
                NME.update(round(nme,4))
    return {'nme':NME.avg}

def draw_pts(image_root, save_root, pr_point_dict, gt_point_dict):
    for patient, frame_pts in pr_point_dict.items():
        for frame, pr_pts in frame_pts.items():
            img_dir = osp.join(image_root, patient, '{:03d}.png'.format(frame))
            image = cv2.imread(img_dir)
            image = cv2.resize(image, (224,224))
            if frame in gt_point_dict[patient]:
                gt_pts = gt_point_dict[patient][frame]
                gt_pts_list = [np.array([gt_pts[i], gt_pts[i+1]]) for i in range(0, len(gt_pts), 2)]
                for idx, pt in enumerate(gt_pts_list):
                    cv2.circle(image, (int(pt[0]),int(pt[1])), 1, (0, 255, 0), -1)
            pr_pts_list = [np.array([pr_pts[i], pr_pts[i+1]]) for i in range(0, len(pr_pts), 2)]
            for pt in pr_pts_list:
                cv2.circle(image, (int(pt[0]),int(pt[1])), 1, (0, 0, 255), -1)
            
            os.makedirs(osp.join(save_root, patient), exist_ok=True)
            cv2.imwrite(osp.join(save_root, patient, '{:03d}.png'.format(frame)), image)


def get_pts_dict(txt_dir):
    pts_dict = collections.defaultdict(_defaultdict_of_lists)
    with open(txt_dir, "r") as f:
        for line in f:
            data = line.strip().split(' ')
            info = data[-1].split('_')
            name = info[0]
            frame = int(info[1].split('.')[0])
            pts = [float(pt) for pt in data[:-1]]
            if len(pts) > 92:
                pts = pts[:92]
            pts_dict[name][frame]=pts
    return pts_dict


if __name__ == '__main__':
    exp_path = 'result/lmks'
    visulize = False
    data_root = '/home/lihh/Projects/data'
    img_save_root = 'result/image'
    metrics_save_root = 'result/metrics'
    os.makedirs(metrics_save_root, exist_ok=True)
    for dataset in os.listdir(exp_path):
        
        with open(f'{metrics_save_root}/{dataset}_nme.csv', 'w+') as result_f:
            result_f.write('Dataset,ModelName,eNME,tNME\n')    
            for modelname in sorted(os.listdir(osp.join(exp_path, dataset))):
                if 'pts' in modelname or 'top' in modelname:
                    for item in modelname.split('_'):
                        if 'pts' in item:
                            n_pts = int(item[:-3])
                        if 'top' in item:
                            top_k = int(item[:-3])
                        else:
                            top_k = 5
                else:
                    n_pts = 46
                    top_k = 5
                    
                gt_lmk_dir = f'/home/lihh/Projects/data/{dataset}/Points/{n_pts}pts/points.txt'
                gt_lmk_dict = get_pts_dict(gt_lmk_dir)
                TEST_NME = MiscMeter()
                VAL_NME = MiscMeter()            
                exp_dir = f'{exp_path}/{dataset}/{modelname}'  

                val_pd_lmk_dir = f'{exp_dir}/val_lmk.txt'
                if os.path.exists(val_pd_lmk_dir):
                    val_pd_lmk_dict = get_pts_dict(val_pd_lmk_dir)
                    val_result = val_metrics(val_pd_lmk_dict, gt_lmk_dict, n_pts)
                    VAL_NME.update(val_result['nme'])

                
                test_pd_lmk_dir = f'{exp_dir}/test_lmk.txt'
                if os.path.exists(test_pd_lmk_dir):
                    test_pd_lmk_dict = get_pts_dict(test_pd_lmk_dir)
                    test_result = val_metrics(test_pd_lmk_dict, gt_lmk_dict, n_pts)
                    TEST_NME.update(test_result['nme'])
                
                if visulize:
                    test_img_save_root = osp.join(img_save_root, dataset, modelname, 'test')
                    os.makedirs(test_img_save_root, exist_ok=True)
                    draw_pts(osp.join(data_root, dataset, 'Image'), test_img_save_root, test_pd_lmk_dict, gt_lmk_dict)
                    # image_root, save_root, view, pr_point_dict, gt_point_dict
                # print('{}-{}:\t\t val:{:.3f} test:{:.3f}'.format(dataset, modelname, val_result['nme'], test_result['nme']))
                
                result_f.write(f'{dataset},{modelname},{VAL_NME.avg:.3f},{TEST_NME.avg:.3f}\n')
                result_f.flush()
                print(f'{dataset},{modelname},{VAL_NME.avg:.3f},{TEST_NME.avg:.3f}\n')
