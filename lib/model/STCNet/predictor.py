# Base Predictor and MultiBranch Predictor
# Date: 2019/09/13
# Author: Beier ZHU

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.model.STCNet.gcn_model.map_to_node import BaseMapToNode
from lib.model.STCNet.gcn_model.sem_gcn import SemGCN
from lib.model.STCNet.gcn_model.tsemgcn import TSemGCN
from lib.model.STCNet.gcn_model.tgcn import TGCN
from lib.model.STCNet.gcn_model.stgcn import STGCN
from lib.utils.graph_utils import adj_matrix_from_num_points

import logging
logger = logging.getLogger('FLD')

class SemGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(SemGCNPredictor, self).__init__()
        print("Creating SemGCNPredictor ......")
        self.config = kwargs['config']
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.map_to_node = BaseMapToNode(in_channels=in_channels, num_points=num_points)
        top_k = self.config.top_k
        adj_matrix = adj_matrix_from_num_points(adj_dir=self.config.adj_dir, num_points=self.config.num_kpts)
        hid_dim = self.config.gcn_param.hidden_dim
        num_layers = self.config.gcn_param.num_layers
        p_dropout = None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(feat_size*feat_size*4, 2), p_dropout=p_dropout)

    def forward(self, x):
        BS, T, C, H, W = x.shape
        x = x.view(BS*T, C, H, W)
        out = self.map_to_node(x)
        feat, out = self.sem_gcn(out)
        out = out.view(BS, T, -1)
        return {'pred':out, 'fpred':None, 'bpred':None}


class FCPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(FCPredictor, self).__init__()
        print("Creating FCPredictor ......")
        _, _, _, num_out_feat4 = in_channels  
        self.fc1 = nn.Linear(7*7*num_out_feat4, 256)
        self.fc2 = nn.Linear(256, num_points*2)
        self.relu = nn.ReLU()        


    def forward(self, x):
        BS, T, C, H, W = x.shape
        x = x.view(BS*T, C, H, W)
        out = x.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(BS, T, -1)
        return {'pred':out, 'fpred':None, 'bpred':None}

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1, tsteps=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        if tsteps==2:
            self.cycle = 1
        elif tsteps==1:
            self.cycle = 0
        else:
            self.cycle = 2

        self.layer = nn.Linear(in_channels * (self.seq_length+self.cycle), out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
            x = self.layer(x)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
            x = self.layer(x)
        elif x.dim() == 4:
            bs = x.size(0)
            frames = x.size(3)
            pre_idx = torch.roll(torch.arange(0, frames), 1)
            post_idx = torch.roll(torch.arange(0, frames), 1)
            for idx in range(0,frames):

                x1 = torch.index_select(x[:,:,:,idx], self.dim, self.indices.view(-1))
                x1 = torch.cat((x1,torch.index_select(x[:,:,:,pre_idx[idx]], self.dim, self.indices[0].view(-1))),dim=1)
                if(frames>3):
                    x1 = torch.cat((x1,torch.index_select(x[:,:,:,post_idx[idx]], self.dim, self.indices[0].view(-1))), dim=1)
                x1 = x1.view(bs, n_nodes, -1)

                x1 = self.layer(x1)
                if idx==0:
                    x_out= x1.unsqueeze(3)
                else:
                    x_out = torch.cat((x_out, x1.unsqueeze(3)), dim=3)
            x= x_out
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2, 3 and 4, but received {}'.format(
                    x.dim()))
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

class GCPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(GCPredictor, self).__init__()
        self.kpt_channels = 2
        self.gcn_channels = [4] # [4,8,8,16,16,32,32,48] [4] [4, 16] [16, 48]
        _, _, _, num_out_feat4 = in_channels
        # construct nodes for graph CNN decoder:
        self.num_nodes = num_points

        # construct edges for graph CNN decoder:
        adjacency = self.create_graph(self.num_nodes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # init GCN:
        self.init_gcn(adjacency, num_out_feat4, 1, True)   

    def create_graph(self, num_nodes: int) -> np.ndarray:
        adjacency = []
        for ii in range(num_nodes):
            x = list(range(num_nodes))
            x.insert(0, x.pop(ii))
            adjacency.append(x)
        adjacency = np.array(adjacency)

        return adjacency

    def init_gcn(self, adjacency: np.ndarray, features_size: int, tsteps: int, is_gpu: bool):

        self.spiral_indices = torch.from_numpy(adjacency)
        if not is_gpu:
            self.spiral_indices = self.spiral_indices.cpu()
        else:
            self.spiral_indices = self.spiral_indices.cuda()

        #self.regression_layer = nn.Sequential(nn.Linear(self.gcn_channels[0]*num_kpts,1), nn.Sigmoid())
        # construct graph CNN layers:
        # self.gcn_channels = [2, 2, 3, 3, 4, 4, 4]
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(features_size, self.num_nodes * self.gcn_channels[-1])) # 512 , 80 * 4
        for idx in range(len(self.gcn_channels)): # range(7)
            if idx == 0:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx - 1], # 4
                               self.gcn_channels[-idx - 1], # 4
                               self.spiral_indices, tsteps=tsteps)) # 1
            else:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx], self.gcn_channels[-idx - 1],
                               self.spiral_indices, tsteps=tsteps))
        self.decoder_layers.append(
            SpiralConv(self.gcn_channels[0], self.kpt_channels, self.spiral_indices, tsteps=tsteps))

    def forward(self, x):
        BS, T, C, H, W = x.shape
        x = x.view(BS*T, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_nodes, self.gcn_channels[-1])
            elif i != num_layers - 1:
                x = F.elu(layer(x))
            else:
                x = layer(x)
        x = x.view(BS, T, -1)
        return {'pred':x, 'fpred':None, 'bpred':None}

class TGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs) -> None:
        super(TGCNPredictor, self).__init__()
        print("Creating TGCNPredictor ......")
        self.config = kwargs['config']
        self.map_to_node = BaseMapToNode(in_channels=in_channels, num_points=self.config.num_kpts)
        adj_matrix = adj_matrix_from_num_points(adj_dir=self.config.adj_dir, num_points=self.config.num_kpts)
        self.tgcn = TGCN(adj_matrix, feat_size*feat_size*4, self.config.gru_param.hidden_dim)
        
        self.fc1 = nn.Linear(num_points*self.config.gru_param.hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_points*2)
        self.relu = nn.ReLU()        

    def forward(self, x):

        BS, T, C, H, W = x.shape
        x = x.view(BS*T, C, H, W)

        node_vectors = self.map_to_node(x) # [BS*T, num_pts, 196]
        _, NUMPTS, HIDDENSIZ = node_vectors.shape

        tgcn_input = node_vectors.view(BS, T, NUMPTS, HIDDENSIZ)
        tgcn_feats = self.tgcn(tgcn_input)
        
        
        BS, T, NUMPTS, HIDDENSIZE = tgcn_feats.shape
        tgcn_feats = tgcn_feats.view(BS*T, NUMPTS, HIDDENSIZE)
        fc_feats = tgcn_feats.reshape(x.size(0), -1)

        fc_feats = self.fc1(fc_feats) # BS, 10, npts, 2
        out = self.fc2(fc_feats)
        out = out.view(BS, T, -1)
        # gcn_pred: BS*T, 

        result_dict = {'pred':out}
        return result_dict
    
class TSemGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs) -> None:
        super(TSemGCNPredictor, self).__init__()
        print("Creating TSemGCNPredictor ......")
        self.config = kwargs['config']
        self.map_to_node = BaseMapToNode(in_channels=in_channels, num_points=self.config.num_kpts)
        adj_matrix = adj_matrix_from_num_points(adj_dir=self.config.adj_dir, num_points=self.config.num_kpts)

        self.tsemgcn = TSemGCN(adj_matrix, feat_size*feat_size*4, self.config.gru_param.hidden_dim)

    def forward(self, x):
        # backbone decoding
        BS, T, C, H, W = x.shape
        x = x.view(BS*T, C, H, W)

        # node embedding
        node_vectors = self.map_to_node(x) # [BS*T, num_pts, 196]
        _, NUMPTS, HIDDENSIZ = node_vectors.shape
        tsemgcn_input = node_vectors.view(BS, T, NUMPTS, HIDDENSIZ)

        # tsemgcn predicting
        outputs = self.tsemgcn(tsemgcn_input)
        
        # output
        result_dict = {'pred':outputs['pred'], 'feat':outputs['feat'],'fpred':None, 'bpred':None}
        return result_dict


class STGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs) -> None:
        super(STGCNPredictor, self).__init__()
        print("Creating STGCNPredictor ......")
        self.config = kwargs['config']
        self.map_to_node = BaseMapToNode(in_channels=in_channels, num_points=self.config.num_kpts)
        adj_matrix = adj_matrix_from_num_points(adj_dir=self.config.adj_dir, num_points=self.config.num_kpts)

        self.tsemgcn = STGCN(adj_matrix, feat_size*feat_size*4, self.config.gru_param.hidden_dim)

    def forward(self, x):
        # backbone decoding
        BS, T, C, H, W = x.shape
        x = x.view(BS*T, C, H, W)

        # node embedding
        node_vectors = self.map_to_node(x) # [BS*T, num_pts, 196]
        _, NUMPTS, HIDDENSIZ = node_vectors.shape
        tsemgcn_input = node_vectors.view(BS, T, NUMPTS, HIDDENSIZ)

        # tsemgcn predicting
        outputs = self.tsemgcn(tsemgcn_input)
        
        # output
        result_dict = {'pred':outputs['pred'], 'feat':outputs['feat'],'fpred':None, 'bpred':None}
        return result_dict