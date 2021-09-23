import torch
from patch_transform import *
import numpy as np

class DataAugmentation(object):
    def __init__(self, affine_kwargs = None, noise_kwargs = None, blur_kwargs = None, contrast_kwargs = None,
                 gamma_kwargs = None, bright_kwargs = None, mirror_kwargs = None,
                 affine_prob = 0.2, noise_prob = 0.15, blur_prob = 0.2, contrast_prob = 0.15, gamma_prob = 0.15, bright_prob = 0.15, mirror_prob = 0.5):

#    def __init__(self, gamma_range = (0.5, 2), epsilon = 1e-7, device = None, retain_stats = True):
        if affine_kwargs == None:
            affine_kwargs = {'degree':[30,30,30], 'axis':[0,1,2], 'scale':[0.85, 1.15], 'use_gpu': True, 'device':0}
        
        if noise_kwargs ==  None:
            noise_kwargs = {'device': 0, 'prob_per_modalities': 0.5}
        
        if blur_kwargs == None:
            blur_kwargs = {'sigma': 1.4, 'width':3, 'device':0}
        
        if contrast_kwargs == None:
            contrast_kwargs = {'contrast_range' : [0.65, 1.5], 'preserve_range' : True, 'device':0}
        
        if gamma_kwargs == None:
            gamma_kwargs = {'gamma_range': (0.5, 1.5), 'epsilon':1e-7, 'device':0, 'retain_stats':True}
        
        if bright_kwargs == None:
            bright_kwargs = {'device':0 , 'rng': [0.7, 1.3]}
        
        if mirror_kwargs == None:
            mirror_kwargs = {'x_prob':0.5, 'y_prob':0.5,'z_prob':0.5}


        self.affine_transform = Batch_Affine_3D(**affine_kwargs)
        self.noise_transform = Batch_Gaussian_Noise(**noise_kwargs)
        self.blur_transform = Batch_Gaussian_Blur_3D(**blur_kwargs)
        self.contrast_transform = Batch_Contrast(**contrast_kwargs)
        self.bright_transform = Batch_Brightness(**bright_kwargs)
        self.gamma_transform = Batch_GammaTransform(**gamma_kwargs)
        self.mirror_transform = Batch_Mirroring(**mirror_kwargs)
        
        self.affine_prob = affine_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.contrast_prob = contrast_prob
        self.bright_prob = bright_prob
        self.gamma_prob = gamma_prob
        self.mirror_prob = mirror_prob


    def __call__(self, images, seg, device = 0):
        images = images.float()
        seg = seg.float()
        if images.device.index != device:
            images = images.to(device)

        if seg.device.index != device:
            seg = seg.to(device)
        

        affine_prob = np.random.uniform(0, 1)
        noise_prob = np.random.uniform(0, 1)
        blur_prob = np.random.uniform(0, 1)
        contrast_prob = np.random.uniform(0, 1)
        bright_prob = np.random.uniform(0, 1)
        gamma_prob = np.random.uniform(0, 1)
        mirror_prob = np.random.uniform(0,1)

        
        if affine_prob < self.affine_prob:
            self.affine_transform.get_matrices_and_coords(images)
            images = self.affine_transform(images)
            seg = self.affine_transform(seg)
        
        if noise_prob < self.noise_prob:
            images = self.noise_transform(images)
        
        if blur_prob < self.blur_prob:
            images = self.blur_transform(images)

        if contrast_prob < self.contrast_prob:
            images = self.contrast_transform(images)

        if bright_prob < self.bright_prob:
            images = self.bright_transform(images)
        
        if gamma_prob < self.gamma_prob:
            images = self.gamma_transform(images)
        
        if affine_prob < self.affine_prob:
            self.mirror_transform.get_mirror_axis()
            images = self.mirror_transform(images)
            seg = self.mirror_transform(seg)
        
        

        return images, seg

        


        