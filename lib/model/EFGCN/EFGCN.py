import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from lib.model.EFGCN.encoders import video_encoder
from lib.model.EFGCN.CNN_GCN import kpts_decoder


class EFGCN(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """
    def __init__(self, backbone: str = 'r3d_18', num_kpts=46,
                 GCN_channels_kpts: List = [4], is_gpu=False):

        super(EFGCN, self).__init__()


        self.GCN_channels_kpts = GCN_channels_kpts

        self.encoder = video_encoder(backbone=backbone)
        self.kpts_regressor = kpts_decoder(features_size=self.encoder.img_feature_size,
                                         kpt_channels=2,
                                         gcn_channels=self.GCN_channels_kpts,
                                         num_kpts=num_kpts*2,
                                         is_gpu=is_gpu)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        kpts = self.kpts_regressor(features)
        return kpts


if __name__ == '__main__':
    frame_size = 112
    num_frames = 2
    num_kpts = 40
    num_batches = 4
    kpt_channels = 2 # kpts dim
    print('load model')
    img = torch.rand(num_batches, 3, num_frames, frame_size, frame_size).cuda()
    m = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[16, 32, 32, 48], is_gpu=True)
    m = m.cuda()
    print('model loaded')
    ef_pred, kpts_pred = m(img)
    ef, kpts = torch.rand(num_batches).cuda(), torch.rand(num_batches, 160).cuda()
    loss = nn.L1Loss()
    print(loss(ef_pred, ef) + loss(kpts_pred, kpts))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    print("hi")
    pass
