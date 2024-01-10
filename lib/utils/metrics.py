import numpy as np
import torch
from medpy import metric

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds

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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = val if val < self.min else self.min
        self.max = val if val > self.max else self.max

def eval_NME(pred, targ, num_points=41, mode='IO'):
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

    return round(np.mean(nme),4), nme


def dist_acc(dists, thr):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def cal_acc(idx, dists, thr):
    acc = np.zeros((len(idx)))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0

    return avg_acc

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def eval_PCK(pred, target, num_pts, thr=0.2):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    pred    : np.array bs x n_pts x 2
    target  : np.array bs x n_pts x 2

    '''
    pred = pred.reshape(-1, num_pts, 2) # batchsize x 68 x 2
    target = target.reshape(-1, num_pts, 2)
    idx = list(range(num_pts)) #  点的个数
    norm = np.ones((pred.shape[0], 2)) * np.array([112, 112]) / 10
    dists = calc_dists(pred, target, norm)
    pck = cal_acc(idx,dists,thr)
    return pck