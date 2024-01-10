import cv2
import numpy as np
import os.path as osp

def draw_tensor_seq_image(sample, save_root):
    names, imgs_tensor, pts_tensor, masks_tensor = sample['names'], sample['images'], sample['landmarks'], sample['masks']
    batch_size = imgs_tensor.shape[0]
    for i_batch in range(batch_size):
        name = names[i_batch]
        images = imgs_tensor[i_batch].cpu().clone().detach().numpy().transpose(0,2,3,1) # cv2: 3, 10, 224, 224 - 10, 224, 224, 3
        pts = pts_tensor[i_batch].cpu().clone().detach().numpy()
        masks = masks_tensor[i_batch].cpu().clone().detach().numpy().astype(np.uint8).squeeze()
        # for i in range(1,9):
        #     image = images[i]
        #     image -= image.min()
        #     image = image/image.max()*255
        #     image = np.ascontiguousarray(image).astype(np.float32)
        #     frame = f'{name}-00{i}.png'
        #     save_dir = osp.join(save_root, frame)
        #     cv2.imwrite(save_dir,image)
        #     image = cv2.imread(save_dir)
        for i in [0,-1]:
            image = images[i]
            mask = masks[i]
            pt = pts[i]
            contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            pt = [int(i) for i in pt]
            pt = np.array([[pt[i],pt[i+1]] for i in range(0,len(pt)-1,2)])
            image -= image.min()
            image = image/image.max()*255
            image = np.ascontiguousarray(image).astype(np.float32)
            if i == 0:
                frame = f'{name}-000.png'
            else:
                frame = f'{name}-009.png'
            save_dir = osp.join(save_root, frame)
            cv2.imwrite(save_dir,image)
            image = cv2.imread(save_dir)

            cv2.drawContours(image, contours, -1, (0,224,0), 1)
            for p in pt:
                # cv2.circle(image, (int(pts[idx]),int(pts[idx+1])), 2, (0,0,238), -1)
                cv2.circle(image, p, 2, (0,0,238), -1)
            cv2.imwrite(save_dir,image)


def draw_tensor_sig_image(sample, save_root):
    names, imgs_tensor, pts_tensor, masks_tensor = sample['names'], sample['images'], sample['landmarks'], sample['masks']
    images = imgs_tensor.cpu().clone().detach().numpy().transpose(0,2,3,1)
    pts = pts_tensor.cpu().clone().detach().numpy()
    masks = masks_tensor.cpu().clone().detach().numpy().astype(np.uint8)

    for name, image, mask, pt in zip(names, images, masks, pts):
        contours,_ = cv2.findContours(mask[:,:,0],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        pt = [int(i) for i in pt]
        pt = np.array([[pt[i],pt[i+1]] for i in range(0,len(pt)-1,2)])
        image -= image.min()
        image = image/image.max()*255
        image = np.ascontiguousarray(image).astype(np.float32)
        save_dir = osp.join(save_root, name)
        cv2.imwrite(save_dir, image)
        image = cv2.imread(save_dir)

        cv2.drawContours(image, contours, -1, (0,224,0), 1)
        for idx in range(0,len(pt)-1,2):
            cv2.circle(image, pt[idx], 2, (0,0,238), -1)
        cv2.imwrite(save_dir,image)