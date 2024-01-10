import torch
import torchvision
import numbers
import os.path as osp
import cv2
import numpy as np
import collections
import os
import time
'''
CLDDS返回一个字典sample:
names       - list  [bs]
images      - torch [bs, 10, 3, 224, 224]
masks       - torch [bs, 2,  1, 224, 224]
heatmaps    - torch [bs, 2, 46, 112, 112]
landmarks   - torch [bs, 2, 92]
'''

class Dataset(torchvision.datasets.VisionDataset):
    def __init__(self, data_root, names, pt_dir, transform=None, img_size=224, hmp_size=112, iskeyframe=False) -> None:
        super().__init__(data_root)
        self.names = names
        self.img_size = img_size # 图像尺寸
        self.hmp_size = hmp_size
        self.data_root = data_root # home/lihh/Projects/data/dataset_name
        self.transform = transform
        
        assert len(self.names)>0,"Error"
        self.frames = {}
        self.labels = {}

        for name in self.names:
       
            frames = os.listdir(f'{data_root}/Image/{name}')
            if iskeyframe:
                frames = [frames[0], frames[-1]]
          
            self.frames[name] = [f'{data_root}/Image/{name}/{frame}' for frame in frames]

            frames = os.listdir(f'{data_root}/Mask/{name}')
            self.labels[name] = [f'{data_root}/Mask/{name}/{frame}' for frame in frames]

        self.landmarks = collections.defaultdict(list)
       
        with open(pt_dir, encoding='utf-8-sig') as f:
            for line in f:
                data = line.strip().split(' ')
                name = data[-1].split('_')[0] # name_000.png
                self.landmarks[name].append(np.array([float(i) for i in data[:-1]]))

    def __getitem__(self, index):
       
        name = self.names[index]
        frames = sorted(self.frames[name])
        labels = self.labels[name]
        landmarks = self.landmarks[name]
        assert len(landmarks)>=2, f'{name}缺少关键点标签'
        
        images = []
        for frame in frames:
            image = cv2.imread(frame)
            image = cv2.resize(src = image, dsize = (self.img_size, self.img_size))
            images.append(image)
    
        masks = []
        for frame in labels:
            mask = cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = cv2.resize(src = mask, dsize = (self.img_size, self.img_size))
            # mask = np.array(np.expand_dims(mask/mask.max(), axis=2))
            mask = np.array(mask/mask.max())
            masks.append(mask)
     
        heatmaps = []
        for landmark in landmarks:
            hmp_lmks = [int(i*self.hmp_size/self.img_size) for i in landmark]
            hmp_pts = np.array([[hmp_lmks[i] for i in range(0,len(hmp_lmks),2)],[hmp_lmks[i] for i in range(1,len(hmp_lmks),2)]])
            hmps = self.generate_label_map(hmp_pts)
            if len(hmps.shape)<3:
                print(hmps.shape)
            heatmaps.append(hmps)

        sample = {}
        sample['images'] = np.array(images)
        sample['masks'] = np.array(masks)
        sample['landmarks'] = np.array(landmarks)
        sample['heatmaps'] = np.array(heatmaps)
        sample['names'] = name
        sample['frames'] = frames
        if self.transform is not None:
            sample = self.transform(sample)
    
        sample['key_images'] = torch.FloatTensor(np.array([sample['images'][0], sample['images'][-1]]))
        sample['images'] = torch.FloatTensor(sample['images'])          # [10, 3, 224, 224]
        sample['masks'] = torch.FloatTensor(sample['masks'])            # [2,  1, 224, 224]
        sample['landmarks'] = torch.FloatTensor(sample['landmarks'])    # [2,  npts*2]
        sample['heatmaps'] = torch.FloatTensor(sample['heatmaps'])      # [2,  npts, 112, 112]
        return sample

    def __len__(self):
        return len(self.names)
    
    def generate_label_map(self, pts, height=112, width=112, sigma=4, ctype='gaussian'):
        assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 2, 'The shape of points : {}'.format(pts.shape)

        if isinstance(sigma, numbers.Number):
            sigma = np.zeros((pts.shape[1])) + sigma
        assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

        num_points, threshold = pts.shape[1], 0.01

        transformed_label = np.fromfunction( lambda y, x, pid : ((x - pts[0,pid])**2 + (y - pts[1,pid])**2) / -2.0 / sigma[pid] / sigma[pid],
                                                                (height, width, num_points), dtype=int)
        
        if ctype == 'laplacian':
            transformed_label = (1 + transformed_label) * np.exp(transformed_label)
        elif ctype == 'gaussian':
            transformed_label = np.exp(transformed_label)
        else:
            raise TypeError('Does not know this type [{:}] for label generation'.format(ctype))
        transformed_label[ transformed_label < threshold ] = 0
        transformed_label[ transformed_label >         1 ] = 1
        heatmap           = transformed_label.astype('float32')
        return heatmap
