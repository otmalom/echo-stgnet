import logging
from easydict import EasyDict

import os
import os.path as osp
import time
import shutil
import pandas
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from lib.dataset.dataset import Dataset
from lib.dataset.transform import TransformBuilder
from lib.model.models import getmodel
from lib.losses.wing_loss import WingLoss, SmoothWingLoss, WiderWingLoss, NormalizedWiderWingLoss, L2Loss, EuclideanLoss, NMELoss, LaplacianLoss
from lib.utils.log_helper import init_log
from lib.utils.vis_utils import get_logger, save_result_imgs, save_result_nmes, save_result_lmks
from lib.utils.misc import save_checkpoint, print_speed, load_model
from lib.utils.metrics import MiscMeter, eval_NME

init_log('Engine')
logger = logging.getLogger('Engine')

class Engine(object):
    def __init__(self, config):
        self.config = EasyDict(config)
        cudnn.benchmark = True
        self._build()

    def run_epoch(self, samples, phase, n_epoch=999):
        image = samples['images']
        landmarks = samples['landmarks']
        names = samples['names']
        BS, T, CH, H, W = image.shape

        frames = []
        for i_bs in range(BS):
            frames.append([])
        for i_t in range(T):
            for i_bs in range(BS):
                frames[i_bs].append(samples['frames'][i_t][i_bs].split('/')[-1])

        # forward
        if self.config.is_keyframe:
            image = image[:,0::T-1].contiguous()
        
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        landmarks = landmarks.view(BS*2, -1)
        pred_dict = self.model(image.cuda())
        gcn_pred = pred_dict['pred']
        
        if self.config.is_keyframe:
            key_pred = gcn_pred.view(BS*2,-1)
        else:
            key_pred = gcn_pred[:,0::T-1].contiguous().view(BS*2,-1)
        
        # 这里确保landmarks的维度是bs*2,92
        ion, ions = eval_NME(key_pred.cpu().data.numpy(), landmarks.numpy(), self.config.num_kpts, mode='IO')
        ipn, ipns = eval_NME(key_pred.cpu().data.numpy(), landmarks.numpy(), self.config.num_kpts, mode='IP')
        losses={}
        if phase == 'train':
            loss_a = self.pos_criterion(key_pred, landmarks.cuda()) * self.config.abs_scm_weight
            loss_r = 0
            loss_t = 0
            loss = 0 
            if 'a' in self.config.loss_fn:
                loss += loss_a 
            if 'r' in self.config.loss_fn:
                loss_r = self.smooth_criterion(key_pred, landmarks.cuda()) * self.config.smooth_weight
                loss += loss_r
            if 't' in self.config.loss_fn:
                gcn_feat = pred_dict['feat']
                
                for t in range(T-1):
                    loss_t += self.temporal_criterion(gcn_feat[t],gcn_feat[t+1])
                loss_t /= (T-1)
                loss += loss_t
            losses = {'loss_a':loss_a, 'loss_t':loss_t, 'loss_r':loss_r}
            self.lr_scheduler.optimizer.zero_grad()
            loss.backward()
            self.lr_scheduler.optimizer.step()

        if (phase == 'val' and (n_epoch>=self.config.epochs-1)) or phase == 'test':
            
            # BS, T, C, H, W
            save_result_nmes(ipns, self.nme_path, names, frames)
            save_result_lmks(gcn_pred.view(BS*T,-1).cpu().data.numpy(), self.lmk_path, names, frames)
            if self.config.visualize:
                if self.config.train_param.is_keyframe:
                    images_cv2 = samples['images_cv2'][:,0::T-1]
                    n_seq = 2
                else:
                    images_cv2 = samples['images_cv2']
                    n_seq = T
                for i_batch in range(BS):
                    images_cv2_seq = images_cv2[i_batch].numpy()
                    save_result_imgs(images_cv2_seq, self.vis_dir[phase], names[i_batch], gcn_pred.view(BS,n_seq,-1)[i_batch].cpu().data.numpy(), landmarks.view(BS,2,-1)[i_batch].numpy())
        
        return losses, ion, ipn
    
    def train(self):
        self._build_train_loader()
        config = self.config

        train_loader = self.train_loader

        # MiscMeter用于计算和存储均值及当前值
        ION_MIN = MiscMeter()
        IPN_MIN = MiscMeter()
        echo_time=MiscMeter()
        for epoch in range(self.start_epoch, config.epochs):
            epoch_start = time.time()
            lr = self.lr_scheduler.get_last_lr()[0]
            # train for one epoch
            batch_time = MiscMeter()
            data_time = MiscMeter()

            loss_a = MiscMeter()
            loss_r = MiscMeter()
            loss_t = MiscMeter()
            ION = MiscMeter()            
            IPN = MiscMeter()
            end = time.time()
            for i, samples in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                # retrive data 
                # 读取数据，现在需要知道的是图像和ladmarks的张量是什么维度的，各个维度是什么意义
                losses, ion, ipn = self.run_epoch(samples, 'train', epoch)
                loss_a.update(losses['loss_a'].item())
                if losses['loss_r']!=0:
                    loss_r.update(losses['loss_r'].item())
                if losses['loss_t']!=0:
                    loss_t.update(losses['loss_t'].item())
                ION.update(ion)
                IPN.update(ipn)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                data_time_ratio = data_time.avg/batch_time.avg
                # print info
                if i % config.print_freq == 0:
                    
                    print('Epoch: [{0}/{1}] [{2}] '
                                'ION:{ION.avg:.4f} '
                                'IPN:{IPN.avg:.4f} '
                                'LR:{lr:.5f} '
                                'LOSS_A:{loss_a.avg:.4f} '
                                'LOSS_R:{loss_r.avg:.4f} '
                                'LOSS_T:{loss_t.avg:.4f} '.format(epoch,config.epochs,len(train_loader),
                                                            
                                                            ION=ION,
                                                            IPN=IPN,
                                                            loss_a=loss_a,
                                                            loss_r=loss_r,
                                                            loss_t=loss_t,
                                                            lr=lr))
                    
            self.lr_scheduler.step()
            if (epoch + 1) % config.snapshot_freq == 0:
                # deallocate memory
                del losses
                # evaluate running models
                ion, ipn = self.evaluate(epoch)
                is_best = True if ion < ION_MIN.min else False
                # save current checkpoint and best checkpoint
                ION_MIN.update(ion)
                IPN_MIN.update(ipn)
                print('[VAL] ION min {:.4f} IPN min {:.4f}'.format(ION_MIN.min, IPN_MIN.min))
                if is_best:
                    print('Saving best checkpoint.')
                save_checkpoint({
                        'epoch': epoch + 1,
                        'optimizer': self.optimizer.state_dict(),
                        'state_dict': self.model.state_dict(),}, 
                        checkpoint=self.config.proj_path, is_best=is_best)
                echo_time.update(time.time() - epoch_start)
                speed=print_speed(epoch, echo_time.avg, config.epochs)
                print(speed)

    @torch.no_grad()
    def evaluate(self, epoch=2000):
        self._build_val_loader()
        self._report_path_setting('val')
        ION = MiscMeter() 
        IPN = MiscMeter() 
        for batch_id, samples in enumerate(self.val_loader):
            _, ion, ipn = self.run_epoch(samples, 'val', epoch)
            ION.update(ion)
            IPN.update(ipn)
           
        print('[VAL] ION: {ION.avg:.4f} IPN: {IPN.avg:.4f}'.format(ION=ION, IPN=IPN))
        return ION.avg, IPN.avg

    @torch.no_grad()
    def test(self):
        self._build_test_loader()
        self._report_path_setting('test')
        data_loader = self.test_loader
        ION = MiscMeter() 
        IPN = MiscMeter() 

        for batch_id, samples in enumerate(data_loader):
            _, ion, ipn = self.run_epoch(samples, 'test')
            ION.update(ion)
            IPN.update(ipn)
           
        print('[TEST] ION: {ION.avg:.4f} IPN: {IPN.avg:.4f}'.format(ION=ION, IPN=IPN))
        return ION.avg, IPN.avg
    
    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting() # 设置目标路径
        self._vis_dir_setting()
        self._build_model()
        self._build_optimizer()
        if config.train:
            self._build_scheduler()
            self._build_criterion()

        self._load_model()
        self._prepare_data()
        self._build_dataloader()
            
    def _prepare_data(self):
        config = self.config
        self.pt_dir = f'{config.data_path}/Points/{config.num_kpts}pts/points.txt'
        filelist_dir = config.filelist
        with open(filelist_dir, 'r') as f:
            data = pandas.read_csv(f)
        self.train_names = data['FileName'][data['Split']=='TRAIN'].tolist()
        self.val_names = data['FileName'][data['Split']=='VAL'].tolist()
        self.test_names = data['FileName'][data['Split']=='TEST'].tolist()       

    def _load_model(self):
        config = self.config
        if hasattr(config, 'load_path') and config.load_path:
            if os.path.isfile(config.load_path):
                if hasattr(config, 'resume') and config.resume:
                    self.start_epoch = load_model(config.load_path, self.model, self.optimizer)
                else:
                    load_model(config.load_path, self.model)
            else:
                logger.error("=> no checkpoint found at '{}'".format(config.load_path))
                exit(-1)

    def _build_model(self):
        self.model = self.build_model_helper(config=self.config)
        
    def _build_criterion(self):
        config = self.config.criterion
        adj_dir = self.config.adj_dir
        num_points = self.config.num_kpts
        if config.type == 'SmoothL1':
            self.pos_criterion = nn.SmoothL1Loss().cuda()
        elif config.type == 'MSE':
            self.pos_criterion = nn.MSELoss().cuda()  
        elif config.type == 'NME':
            self.pos_criterion = NMELoss(**config.kwargs).cuda()  
        elif config.type == 'L1':
            self.pos_criterion = nn.L1Loss().cuda()
        elif config.type == 'L2':    
            self.pos_criterion = L2Loss().cuda()
        elif config.type == 'Euclidean':    
            self.pos_criterion = EuclideanLoss().cuda()
        elif config.type == 'Wing':
            self.pos_criterion = WingLoss(**config.kwargs).cuda()
        elif config.type == 'SmoothWing':
            self.pos_criterion = SmoothWingLoss(**config.kwargs).cuda()
        elif config.type == 'WiderWing':
            self.pos_criterion = WiderWingLoss(**config.kwargs).cuda()
        elif config.type == 'NormalizedWiderWing':    
            self.pos_criterion = NormalizedWiderWingLoss(**config.kwargs).cuda()
        self.temporal_criterion = nn.L1Loss().cuda()
        self.smooth_criterion = LaplacianLoss(WiderWingLoss(**config.kwargs).cuda(), adj_dir=adj_dir, num_points=num_points).cuda()
        # loss, adj_dir, num_points, reduction="mean"
    @staticmethod
    def build_model_helper(config=None):
        model = getmodel(config)
        if torch.cuda.device_count() > 1:
            logger.debug("using {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model = model.cuda()
        return model
    
    def _build_optimizer(self):
        config = self.config.optimizer
        model = self.model
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)
        optimizer = optim(model.parameters(), **config.kwargs)
        self.optimizer = optimizer

    def _build_scheduler(self):
        config = self.config.scheduler
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=self.start_epoch - 1)
        
    def _build_train_loader(self):
        train_transforms = self.transform_builder.train_transforms()
        train_dataset = Dataset(
            data_root=self.config.data_path,
            names=self.train_names, 
            pt_dir = self.pt_dir,
            transform=train_transforms)
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True,
                num_workers=self.config.workers, pin_memory=False, drop_last=True)

    def _build_val_loader(self):
        val_transforms = self.transform_builder.test_transforms()
        val_dataset = Dataset(
            data_root=self.config.data_path,
            names=self.val_names, 
            pt_dir = self.pt_dir,
            transform=val_transforms)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.workers, pin_memory=False)

    def _build_test_loader(self):
        test_transforms = self.transform_builder.test_transforms()
        test_dataset = Dataset(
            data_root=self.config.data_path,
            names=self.test_names, 
            pt_dir = self.pt_dir,
            transform=test_transforms)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.workers, pin_memory=False)

    def _build_dataloader(self):
        self.transform_builder = TransformBuilder()

    def _dir_setting(self):
        os.makedirs(self.config.proj_path, exist_ok=True)

    def _vis_dir_setting(self):
        if self.config.visualize:
            # 设置保存结果可视化图像的路径
            '''Set up the directory for saving landmarked images'''
            val_vis_dir = os.path.join(self.config.proj_path, 'vis_result', 'val')
            test_vis_dir = os.path.join(self.config.proj_path, 'vis_result', 'test')
            os.makedirs(val_vis_dir, exist_ok=True)
            os.makedirs(test_vis_dir, exist_ok=True)
            self.vis_dir = {'test':test_vis_dir, 'val':val_vis_dir}

        elif osp.exists(osp.join(self.config.proj_path, 'vis_result')):
            shutil.rmtree(osp.join(self.config.proj_path, 'vis_result'))

    def _report_path_setting(self, phase):
        self.nme_path=os.path.join(self.config.proj_path, f'{phase}_nme.txt')
        self.lmk_path=os.path.join(self.config.proj_path, f'{phase}_lmk.txt')
       
        if os.path.exists(self.nme_path):
            os.remove(self.nme_path)
        if os.path.exists(self.lmk_path):
            os.remove(self.lmk_path)

        if os.path.exists(os.path.join(self.config.proj_path, 'nme.txt')):
            os.remove(os.path.join(self.config.proj_path, 'nme.txt'))
        if os.path.exists(os.path.join(self.config.proj_path, 'lmk.txt')):
            os.remove(os.path.join(self.config.proj_path, 'lmk.txt'))


def get_engine_tem_seq(config):
    agent = Engine(config)
    return agent