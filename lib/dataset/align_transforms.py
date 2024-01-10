# Custom "Dataset" class for LAB
# Author: Beier ZHU
# Date: 2019/07/25

import numpy as np
import random
import cv2
cv2.ocl.setUseOpenCL(False)
import torchvision.transforms.functional as TF
import numbers

def _get_corr_list(num_pts):
    """Show indices for landmarks mirror"""
    assert num_pts in [41, 46], '#landmarks should be 41 or 46, but got {}'.format(num_pts)
    
    right_pts = [i for i in range(int((41-1)/2))]
    left_pts = [i for i in range(41-1,int((41-1)/2),-1)]
    
    if num_pts == 46:
        right_pts = [44, 45]+right_pts
        left_pts = [42, 41]+left_pts
    assert len(left_pts)==len(right_pts), "#length of left and right should be the same, but got left {}, right {}".format(len(left_pts), len(right_pts))
    corr_list = []
    for i in range(len(left_pts)):
        corr_list.append(left_pts[i])
        corr_list.append(right_pts[i])
    corr_list = np.array(corr_list, dtype=np.uint8).reshape(-1, 2)
    return corr_list

def _get_affine_matrix(center, angle, translations, zoom, shear, do_mirror=False):
    """Compute affine matrix from affine transformation"""
    # Rotation & scale
    matrix = cv2.getRotationMatrix2D(center, angle, zoom).astype(np.float64)
    # translate
    matrix[0, 2] += translations[0] * zoom
    matrix[1, 2] += translations[1] * zoom

    mirror_flag = False
    if do_mirror:
        print('Mirror')
        mirror_rng = random.uniform(0., 1.)
        if mirror_rng > 0.5:
            mirror_flag = True
            matrix[0, 0] = -matrix[0, 0]
            matrix[0, 1] = -matrix[0, 1]
            matrix[0, 2] = (center[0] + 0.5) * 2.0 - matrix[0, 2]

    return matrix, mirror_flag

class Normalize(object):
    """
    Normalize the image channel-wise for each input image,
    the mean and std. are calculated from each channel of the given image
    """

    def __call__(self, sample, type='z-score'):        
        images, masks = sample['images'], sample['masks']
        sample['images_cv2'] = images.astype(np.uint8)
        if len(images.shape) == 4:
            #color img
            if type == 'z-score':
                dst_images = []
                for image in images:
                    # cv2.imwrite("seq_image_ori.png",image)
                    mean, std = cv2.meanStdDev(image)
                    mean, std = mean[:,0], std[:,0]
                    std = np.where(std < 1e-6, 1, std)
                    image = (image - mean)/std
                    image = image.astype(np.float32)
                    # cv2.imwrite("seq_image_nor.png",image)
                    # image_rec = image-image.min()
                    # image_rec = image_rec/image_rec.max()*255
                    # image_rec = np.ascontiguousarray(image_rec).astype(np.float32)
                    # cv2.imwrite("seq_image_rec.png",image_rec)
                    dst_images.append(image)
                dst_masks = []
                for mask in masks:
                    mask = mask[:, :].astype(np.float32)
                    dst_masks.append(mask)
        else:
            print("图像维度有误,应该是4维,但是现在是{}维:{}".format(images.shape[0], images.shape))
        sample['images'] = np.array(images)
        sample['masks'] = np.array(masks)
        return sample

class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # images: [frame, h, w, c]
        images, masks, landmarks, heatmaps = sample['images'], sample['masks'],sample['landmarks'],sample['heatmaps']

        h, w = images.shape[1],images.shape[2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        images = images[:,top: top + new_h, left: left + new_w]
        masks = masks[:,top: top + new_h, left: left + new_w]
        heatmaps = heatmaps[:, int(top/2): int((top + new_h)/2), int(left/2): int((left + new_w/2))]
        for i in range(landmarks.shape[0]):
            for j in range(int(landmarks.shape[1] / 2)):
                landmarks[i][2 * j] -= left
                landmarks[i][2 * j + 1] -= top

        sample['images'] = np.array(images)
        sample['masks'] = np.array(masks)
        sample['landmarks'] = np.array(landmarks)
        sample['heatmaps'] = np.array(heatmaps)
        # sample['image_cv2'] = image.astype(np.uint8)

        return sample

class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False,
                 fillcolor=0, mirror=False, corr_list=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.mirror = mirror
        self.corr_list = corr_list

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            if scale_ranges[0] == scale_ranges[1]:
                scale = scale_ranges[0]
            else:
                scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        images, masks, landmarks, heatmaps = sample['images'], sample['masks'],sample['landmarks'],sample['heatmaps']
        # images    10, 224, 224, 3
        # masks     2, 224, 224
        # landmarks 2, 92
        # heatmaps  2, 112, 112, 46
        h, w = images.shape[1],images.shape[2]
        center = (w/2 - 0.5, h/2 - 0.5)
        angle, translations, zoom, shear = \
            self.get_params(self.degrees, self.translate, self.scale, self.shear, [h, w])

        matrix, mirrored = _get_affine_matrix(center, angle, translations, zoom, shear, self.mirror)
        
        dst_images = []
        for image in images:
            image = cv2.warpAffine(image, matrix, (w,h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
            dst_images.append(image)
        
        dst_masks = []
        for mask in masks:
            mask = cv2.warpAffine(mask, matrix, (w,h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
            dst_masks.append(mask)

        dst_heatmaps = []
        for heatmap in heatmaps:
            heatmap = cv2.warpAffine(heatmap, matrix, (w,h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
            dst_heatmaps.append(heatmap)

        # landmarks tranformation
        matrix = np.resize(matrix, (6,))

        for i in range(landmarks.shape[0]):
            points = np.resize(np.array(landmarks[i]).copy(), (int(len(landmarks[i]) / 2), 2))

            for j in range(landmarks.shape[1]//2):
                x, y = points[j, :]
                x_new = matrix[0] * x + matrix[1] * y + matrix[2]
                y_new = matrix[3] * x + matrix[4] * y + matrix[5]

                landmarks[i][2 * j] = x_new
                landmarks[i][2 * j + 1] = y_new

            if mirrored:
                # reorder index of landmarks after mirroring
                for k in range(self.corr_list.shape[0]):
                    temp_x = landmarks[i][2 * self.corr_list[k, 0]]
                    temp_y = landmarks[i][2 * self.corr_list[k, 0] + 1]
                    landmarks[i][2 * self.corr_list[k, 0]], landmarks[i][2 * self.corr_list[k, 0] + 1] = \
                        landmarks[i][2 * self.corr_list[k, 1]], landmarks[i][2 * self.corr_list[k, 1] + 1]
                    landmarks[i][2 * self.corr_list[k, 1]], landmarks[i][2 * self.corr_list[k, 1] + 1] = temp_x, temp_y

        sample['images'] = np.array(dst_images)
        sample['masks'] = np.array(dst_masks)
        sample['landmarks'] = np.array(landmarks)
        sample['heatmaps'] = np.array(dst_heatmaps)
        return sample

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image (scale by 1/255).
        """
        images = sample['images']
        dst_images = []
        for image in images:
            dst_images.append(image.transpose(2, 0, 1))
        masks = sample['masks']
        dst_masks = []
        for mask in masks:
            dst_masks.append(np.expand_dims(mask, axis=0))
        
        heatmaps = sample['heatmaps']
        dst_heatmaps = []
        for heatmap in heatmaps:
            dst_heatmaps.append(heatmap.transpose(2, 0, 1))

        sample['images'] = np.array(dst_images) # [10, 3, 224, 224]
        sample['masks'] = np.array(dst_masks)
        sample['heatmaps'] = np.array(dst_heatmaps)
        return sample


