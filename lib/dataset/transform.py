from torchvision import transforms
from .align_transforms import RandomAffine,Normalize,ToTensor

class TransformBuilder(object):
    def __init__(self):
        name = None
        
    def train_transforms(self):

        train_transforms = []
        train_transforms.append(RandomAffine(degrees=20, # 旋转正负20度 
                                             translate=(0.1, 0.1), # 随机上下或者左右平移0.1×image_size
                                             scale=(0.9, 1.1), # 放缩正负0.1倍
                                             mirror=None, corr_list=None))
        train_transforms.append(Normalize())
        train_transforms.append(ToTensor())  
        return transforms.Compose(train_transforms)

    def test_transforms(self):

        test_transforms = []
        test_transforms.append(Normalize())
        test_transforms.append(ToTensor()) 
        return transforms.Compose(test_transforms)
