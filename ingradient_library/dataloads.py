import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from torch.utils.data import Dataset
import numpy as np
import pickle
import numpy as numpy
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, device = None, normalizer = None, mode = 'train', dim = '3d'):
        self.root_dir = root_dir
        self.image_name = []
        self.info_name = []
        self.device = device
        self.normalizer = normalizer
        self.mode = mode
        self.dim = dim


        for file_name in os.listdir(root_dir):            
            if 'npz' in file_name:
                self.image_name.append(file_name)
            
            elif 'pkl' in file_name:
                self.info_name.append(file_name)
                    
        self.image_name = sorted(self.image_name)
        self.info_name = sorted(self.info_name)

    def __len__(self):
        return len(self.image_name)

    
    def __getitem__(self, idx):
        info_file = open(os.path.join(self.root_dir,self.info_name[idx]), 'rb')
        info_data = pickle.load(info_file)
        info_file.close()
        data = np.load(os.path.join(self.root_dir, self.image_name[idx]))
        

        x = data['x']
        y = data['y']
        if len(x.shape) == 3 and self.dim == '3d':
            #modality가 1개인 경우
            x = np.expand_dims(x, axis = 0)
        elif len(x.shape) == 2 and self.dim == '2d':
            x = np.expand_dims(x, axis = 0)

        if self.normalizer:
            x = self.normalizer(x)

        if self.mode == 'train' :
            return x, y, info_data
        
        elif self.mode == 'test':
            return x, info_data



class DataLoader3D(object):

    def __init__(self, dataset, resampling = None, augmentation = None, num_iteration = 2, patch_size = (128,128,128), device = 0, batch_size = 2, seg_one_hot = False, is_half = False, info_class = None):
        self.dataset = dataset
        self.current_index = 0
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.now_iter = 0
        self.images, self.seg, self.info = self.get_image_seg(self.current_index)
        self.device = device
        self.pass_patient_index = []
        self.num_iteration = num_iteration
        self.seg_one_hot = seg_one_hot
        self.is_half = is_half
        self.info_class = info_class #[0, 1, 3, 4] 이런식
        self.augmentation = augmentation
        self.resampling = resampling

    
    def new_epoch(self):
        self.now_iter = 0
        self.current_index = 0
    
    def is_end(self):
        return self.current_index == len(self.dataset) and self.now_iter == 0
    
    def return_class(self):
        return np.unique(self.seg)

    def next_index(self):
        if self.now_iter == self.num_iteration:
            self.images, self.seg, self.info = self.get_image_seg(self.current_index)
            self.current_index += 1
            self.now_iter = 0
        
        else:
            self.now_iter += 1

    def get_image_seg(self, current_index):
        images, seg, info = self.dataset[current_index]
        image_shape = np.array(images[0].shape)
        odd_pad = ((self.patch_size - image_shape) % 2 != 0).astype(int)
        even_pad = np.repeat(np.clip(((self.patch_size - image_shape)/2).astype(int), 0, np.inf), 2)
        for i in range(len(odd_pad)):
            even_pad[i*2] += odd_pad[i]
        images = F.pad(torch.tensor(images), list(even_pad.astype(int))[::-1], "constant", 0 )
        seg = F.pad(torch.tensor(seg), list(even_pad.astype(int))[::-1], "constant", 0 )

        return images, seg, info


    def get_oversample_patch(self, non_zero_coords, shape_to_index):
        oversample_high = np.clip(np.max(non_zero_coords, axis = 1) + 1, self.patch_size/2 - 1, shape_to_index - self.patch_size/2)
        oversample_low = np.clip(np.min(non_zero_coords, axis = 1), self.patch_size/2 - 1, shape_to_index - self.patch_size/2)
        oversample_center = np.random.randint(low = oversample_low, high = oversample_high + 1)

        oversample_patch_upper_bound = oversample_center + (self.patch_size/2) + 1
        oversample_patch_lower_bound = oversample_center - (self.patch_size/2 - 1)
        oversample_patch_bound = np.concatenate((oversample_patch_lower_bound.reshape(-1,1),oversample_patch_upper_bound.reshape(-1,1)), axis = 1).astype(int)

        oversample_patch_images = self.images[:, oversample_patch_bound[0][0] : oversample_patch_bound[0][1],
                                          oversample_patch_bound[1][0] : oversample_patch_bound[1][1],
                                          oversample_patch_bound[2][0] : oversample_patch_bound[2][1]]
        oversample_patch_seg = self.seg[oversample_patch_bound[0][0] : oversample_patch_bound[0][1],
                                          oversample_patch_bound[1][0] : oversample_patch_bound[1][1],
                                          oversample_patch_bound[2][0] : oversample_patch_bound[2][1]]
        
        return np.expand_dims(oversample_patch_images , axis = 0), np.expand_dims(oversample_patch_seg , axis = 0)

    def get_normal_patch(self, non_zero_coords, shape_to_index):
        normal_center = np.random.randint(low = self.patch_size/2 - 1, high = shape_to_index - (self.patch_size/2) + 1 )
        normal_patch_upper_bound = normal_center + (self.patch_size/2) + 1
        normal_patch_lower_bound = normal_center - (self.patch_size/2 - 1)
        normal_patch_bound = np.concatenate((normal_patch_lower_bound.reshape(-1,1),normal_patch_upper_bound.reshape(-1,1)), axis = 1).astype(int)
        normal_patch_images = self.images[:, normal_patch_bound[0][0] : normal_patch_bound[0][1],
                                          normal_patch_bound[1][0] : normal_patch_bound[1][1],
                                          normal_patch_bound[2][0] : normal_patch_bound[2][1]]
        normal_patch_seg = self.seg[normal_patch_bound[0][0] : normal_patch_bound[0][1],
                                          normal_patch_bound[1][0] : normal_patch_bound[1][1],
                                          normal_patch_bound[2][0] : normal_patch_bound[2][1]]
        
        return np.expand_dims(normal_patch_images , axis = 0), np.expand_dims(normal_patch_seg , axis = 0)
    



    def generate_train_batch(self):
        non_zero_coords = np.array(np.where(self.seg != 0))
        shape_to_index = np.array(self.images[0].shape) - 1 

        if self.batch_size == 1 :
            result_images, result_seg = self.get_normal_patch(non_zero_coords, shape_to_index)


        for i_b in range(self.batch_size//2):
            oversample_images, oversample_seg = self.get_oversample_patch(non_zero_coords, shape_to_index)
            normal_images, normal_seg = self.get_normal_patch(non_zero_coords, shape_to_index)

            if i_b == 0:
                result_images = np.concatenate((normal_images, oversample_images), axis = 0)
                result_seg = np.concatenate((normal_seg, oversample_seg), axis = 0)
            
            else:
                result_images = np.concatenate((result_images, oversample_images), axis = 0)
                result_seg = np.concatenate((result_seg, oversample_seg), axis = 0)
                result_images = np.concatenate((result_images, normal_images), axis = 0)
                result_seg = np.concatenate((result_seg, normal_seg), axis = 0)

        self.next_index()
        
    
        
        result_images = torch.Tensor(result_images).to(self.device)
        result_seg = torch.Tensor(result_seg).to(self.device)

        if self.resampling:
            result_images = self.resampling(result_images, self.info, mode = 'x')
            result_seg = self.resampling(result_seg, self.info, mode = 'y')

        if self.augmentation:
            result_images, result_seg = self.augmentation(result_images, result_seg)
        
        if self.seg_one_hot:
            result_seg = F.one_hot(result_seg.long(),  num_classes = len(self.info_class)).permute(0, 4, 1, 2, 3)


        if self.is_half:
            result_images = result_images.half()
    

        return result_images, result_seg.long()