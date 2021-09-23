import torch
import numpy as np
import pickle
import os
import torch.nn.functional as F

class Normalizer(object):
    def __init__(self, percentile = None):
        self.percentile = percentile

    def __call__(self, images):
        if isinstance(images, np.ndarray):
            images = torch.Tensor(images)
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        if self.percentile != None:
            images = self.percentile_clipping(images)
        n_modalities, nx, ny, nz = images.shape
        mmean = torch.mean(images.view(n_modalities, -1), -1, True).view(n_modalities,1,1,1)
        mstd = torch.std(images.view(n_modalities, -1), -1, True).view(n_modalities,1,1,1)
        return (images - mmean) / mstd
        
    
    def percentile_clipping(self,images):
        n_modalities, nx, ny, nz = images.shape
        images = images.view(n_modalities, -1)
        vals, indices = torch.sort(images, -1)
        upper_index = indices[:, int(indices.shape[1] * self.percentile[1]):]
        lower_index = indices[:, :int(indices.shape[1] * self.percentile[0])]
        upper_val = vals[:, int(indices.shape[1] * self.percentile[1])]
        lower_val = vals[:, int(indices.shape[1] * self.percentile[0])]
    
        for i in range(n_modalities):
            images[i, upper_index[i]] = upper_val[i]
            images[i, lower_index[i]] = lower_val[i]
        
        return images.view(n_modalities, nx, ny, nz)


class Cropping(object):
    def __call__(self, images):
        non_zero_index = np.where(images != 0)
        min_val = np.min(non_zero_index, axis = 1)
        max_val = np.max(non_zero_index, axis = 1)
        return images[:, min_val[1]:max_val[1], min_val[2]:max_val[2], min_val[3]:max_val[3]]


class Resampling(object):
    def __init__(self, anisotropy_axis_index = None, device = None, target_spacing = [1.0, 1.0, 1.0]):
        self.device = device
        self.anisotropy_axis_index = anisotropy_axis_index
        self.target_spacing = np.array(target_spacing)

    def __call__(self, data, info_data, mode = 'x'):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        if data.device.index != self.device and self.device:
            data = data.to(self.device)

        original_spacing = info_data['spacing']
        new_shape = list((original_spacing[::-1]/self.target_spacing[::-1] * np.array([data.shape[-3], data.shape[-2], data.shape[-1]])).astype(int))
        print(new_shape)
        if mode == 'x':
            data = F.interpolate(data, new_shape, mode = 'trilinear')
        elif mode == 'y':
            data = F.interpolate(data.unsqueeze(0), new_shape).squeeze(0)

        return data



class Get_target_spacing(object):
    def __init__(self, anisotropy_threshold = 3.0, image_dimension = 3):
        self.spaces = None
        self.first = True
        self.anisotropy_threshold = anisotropy_threshold
        self.isotropy_percentile_value = 0.50
        self.anisotropy_percentile_value = 0.90
        self.image_dimension = image_dimension

    def run(self, dir_path):
        for file_name in os.listdir(dir_path):
            if file_name[-3:] == 'pkl':
                path = os.path.join(dir_path, file_name)
                file = open(path, 'rb')
                self.append(pickle.load(file)['spacing'])
                file.close()
    
    def reset(self):
        self.spaces = None

    def append(self, item):
        item = np.array(item).reshape(1,self.image_dimension)
        if self.first:
            self.spaces = item
            self.first = False
        else:
            self.spaces = np.vstack((self.spaces, item)) 
    
    def get_target_space(self):
        anisotropy_axis_index = self.is_anisotropy()
        isotropy_axis_index = np.arange(self.image_dimension) != anisotropy_axis_index
        iso = np.percentile(self.spaces, self.isotropy_percentile_value, 0)
        aniso = np.percentile(self.spaces, self.anisotropy_percentile_value, 0)
        result = np.zeros(self.image_dimension)
        result[anisotropy_axis_index] = aniso[anisotropy_axis_index]
        result[isotropy_axis_index] = iso[isotropy_axis_index]
        return result, anisotropy_axis_index
    
    def is_anisotropy(self):
        return np.where((np.max(self.spaces, axis = 0) / np.min(self.spaces, axis = 0)) > self.anisotropy_threshold)[0].astype(int)
    