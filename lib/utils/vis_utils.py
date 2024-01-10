import os
import logging
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

logger = logging.getLogger('FLD')


def get_logger(logdir):
    logger.info("init logger in {}".format(logdir))
    writer = SummaryWriter(logdir)
    return writer


def get_model_graph(writer, input_shape, net):
    inputs = torch.rand(input_shape).cuda()
    writer.add_graph(net.cuda(), inputs)

 
def add_scalar(writer, scalar_name, val, n_iter):
    writer.add_scalar(scalar_name, val, n_iter)


def add_image(writer, img_name, imgs, n_iter, nrow=1):
    x = vutils.make_grid(imgs, nrow=nrow)
    writer.add_image(img_name, x, n_iter)


def get_show_img(config, input_img, show_idx):
    means = torch.from_numpy(np.array(config.means)).float()
    stds = torch.from_numpy(np.array(config.stds)).float()
    img_norm_factor = float(config.img_norm_factor)
    sample_img = input_img[show_idx].permute(1, 2, 0)
    show_img = ((sample_img * stds + means) * img_norm_factor).permute(2, 0, 1).type(torch.ByteTensor)

    return show_img


def get_hm_tensor(show_idx, input_tensor):
    lst = []
    hm = input_tensor[show_idx]
    for i in range(hm.shape[0]):
        lst.append(hm[i].unsqueeze(0))
    return lst

def save_result_img(image, landmark_pred, landmark_targ, save_path):
    if landmark_targ is not None:
        for pt in landmark_targ:
            targ_pt = (round(float(pt[0])), round(float(pt[1])))
            image = cv2.circle(image, targ_pt, 1, (0, 255, 0), -1)
    for pt in landmark_pred:
        pred_pt = (round(float(pt[0])), round(float(pt[1])))
        image = cv2.circle(image, pred_pt, 1, (0, 0, 255), -1)

    cv2.resize(image,(512,512))
    cv2.imwrite(save_path, image)

def save_result_imgs(images, save_dir, name, landmarks_pred, landmarks_targ):
    # seg_results:'dice','hd95','dices','hd95s','gt_mask','pr_mask'
    # images: [T, 224, 224, 3]
    # names: 
    # landmarks_pred: [T, 92]
    # labdmarks_targ: [2, 92]
    T = images.shape[0]
    for idx in range(T):
        image = images[idx]
        lmk_pred = np.array([np.array([landmarks_pred[idx][i], landmarks_pred[idx][i+1]]) for i in range(0,len(list(landmarks_pred[idx]))-1,2)])
        if idx == 0:
            result_path = os.path.join(save_dir, f'{name}-{idx}.png')
            lmk_targ = np.array([np.array([landmarks_targ[0][i], landmarks_targ[0][i+1]]) for i in range(0,len(list(landmarks_targ[0]))-1,2)])
        elif idx == T-1:
            result_path = os.path.join(save_dir, f'{name}-9.png')
            lmk_targ = np.array([np.array([landmarks_targ[-1][i], landmarks_targ[-1][i+1]]) for i in range(0,len(list(landmarks_targ[-1]))-1,2)])
        else:
            result_path = os.path.join(save_dir, f'{name}-{idx}.png')
            lmk_targ=None
        save_result_img(image, lmk_pred, lmk_targ, result_path)

def save_result_metrics(results, save_path, names):
    with open(save_path, 'a+') as f:
        for dice, hd95, name in zip(results['dices'],results['hd95s'],names):
            f.write(str(dice)+' '+str(hd95)+' '+name+'\n')

def save_result_nmes(nmes, save_path, names, frames):
    frame_name = []
    for name, frame in zip(names, frames):
        
        frame_name.append(f"{name}_{frame[0]}")
        frame_name.append(f"{name}_{frame[-1]}")
    with open(save_path, 'a+') as f:
        for nme, name in zip(nmes, frame_name):
            f.write(str(round(nme,4))+' '+name+'\n')

def save_result_lmks(lmks, save_path, names, frames):
    frame_name = []
    for name, frame in zip(names, frames):
        for f in frame:
            frame_name.append(f"{name}_{f}")
    with open(save_path, 'a+') as f:
        for lmk, name in zip(lmks, frame_name):
            for p in lmk:
                f.write(str(p) + ' ')
            f.write(name + '\n')

class CsvHelper(object):
    """docstring for ClassName"""
    def __init__(self, head, save_path):
        super(CsvHelper, self).__init__()
        self.head = head
        self.save_path = save_path
        self.data = []

    def update(self, data):
        self.data.append(data)

    def save(self):
        result = pd.DataFrame(columns=self.head, data=self.data)
        result.to_csv(self.save_path, index=None)
        