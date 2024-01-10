# Model Builder
# Date: 2019/09/13
# Author: Beier ZHU
import torch.nn as nn
from lib.model.STCNet import predictor 
from lib.model.STCNet.backbones import resnet, resnet_pure
from lib.model.STCNet.backbones import mobilenet
from lib.model.STCNet.backbones import hrnet
from lib.model.STCNet.backbones import efficient_net
from lib.model.STCNet.backbones import vgg


backbone_zoo = {
    'HRNetw32': hrnet.hrnetw32,
    'HRNet': hrnet.get_face_alignment_net,

    'Efficientnet_b0': efficient_net.efficientnet_b0,
    'Efficientnet_b1': efficient_net.efficientnet_b1,
    
    'ResNet18': resnet_pure.ResNet18,
    'ResNet34': resnet_pure.ResNet34,
    'ResNet50': resnet.ResNet50,
    'ResNet101': resnet.ResNet101,
    'ResNet18Pure': resnet_pure.ResNet18,
    'ResNet34Pure': resnet_pure.ResNet34,

    'MobileNetA': mobilenet.MobileNetA,
    'MobileNetB': mobilenet.MobileNetB,
    'MobileNetC': mobilenet.MobileNetC,

    'Vgg16': vgg.Vgg16,
}

predictor_zoo = {
    'FCPredictor': predictor.FCPredictor, # FC a
    'GCPredictor':predictor.GCPredictor, # SOTA方法 a
    'TGCNPredictor': predictor.TGCNPredictor, # TGCN+FC a
    'SemGCNPredictor': predictor.SemGCNPredictor, # SemGCN a
    # 我们的
    'TSemGCNPredictor': predictor.TSemGCNPredictor, # 双状态 art
    'STGCNPredictor' :predictor.STGCNPredictor # 单状态 art
}

class ModelBuilder(nn.Module):
    """
    Build model from cfg
    """
    def __init__(self, config):
        super(ModelBuilder, self).__init__()
        kwargs={}
        kwargs['config'] = config
        self.backbone = backbone_zoo[config.backbone_name](is_color=True, 
                                                           pretrained_path=None, 
                                                           receptive_keep=False)
        feat_size = config.image_size//self.backbone.downsample_ratio
        self.predictor = predictor_zoo[config.predictor_name](in_channels=self.backbone.num_out_feats, 
                                                              feat_size=feat_size, 
                                                              num_points=config.num_kpts,
                                                              **kwargs)

    def forward(self, x):
        BS,T,C,H,W = x.shape
        x = x.view(BS*T, C, H, W)
        x = self.backbone(x)
        _,C,H,W = x['out4'].shape
        x = x['out4'].view(BS,T,C,H,W)
        x = self.predictor(x)

        return x

def get_models(config):
    model = ModelBuilder(config)
    return model