import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Inference(object):
    #images.shape = (modlaities, x, y, z) 형태일 것
    
    def __init__(self, dataset, n_classes, patch_size = (128,128,128), filter = 'gaussian',
                 step_size = 0.5, sigma = 1.0, save_path = None, device = 0):
        self.n_classes = n_classes
        self.dataset = dataset
        self.patch_size = patch_size
        self.sigma = sigma
        self.device = device
        self.step_size = step_size
        if filter == 'gaussian' and len(patch_size) == 3:
            self.filter = self.create_3D_filter().to(device)

        self.patch_size = np.array(patch_size).astype(int)
        self.save_path = save_path
        self.score = []
        self.iter_score =[]


    def new_epoch(self):
        self.is_end = False
        self.current_coordinates = np.array([1,1,1])
        self.current_index = 0
        self.n_modalities = self.dataset[self.current_index][0].shape[0]
        self.image_shape = self.get_image_shape(self.dataset[self.current_index][0])
        self.images = self.get_pad_images(self.dataset[self.current_index][0])
        self.padded_image_shape = self.get_image_shape(self.images)
        self.result_map = self.create_result_tensor()
    
    def get_epoch_score(self):
        self.score.append(torch.vstack(self.iter_score).mean().item())
        self.iter_score = []

    def run(self, model, mode = 'save'):
        self.mode = mode
        while not self.is_end:
            model.eval()
            patch = self.get_inference_patch()
            output = model(patch.unsqueeze(0).float().to(0)).squeeze(0)[0]
            output = output * self.filter
            output = output.detach().cpu()
            lower_bound, upper_bound = self.get_patch_bound()
            self.result_map[:, lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]] = output + self.result_map[: , lower_bound[0]:upper_bound[0],lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]]
            self.get_next_coords()
        

    
    def get_patch_bound(self):
        criteria = (self.current_coordinates * self.patch_size * self.step_size).astype(int)
        return criteria - self.patch_size//2, criteria + self.patch_size//2
    

    def get_inference_patch(self):
        lower_bound, upper_bound = self.get_patch_bound()
        return self.images[:, lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]]
    
    def get_next_coords(self):

        criteria = (self.current_coordinates + 1) * self.patch_size * self.step_size
        if (criteria[0] < self.padded_image_shape[0]):
            self.current_coordinates[0] += 1
        
        elif (criteria[1] < self.padded_image_shape[1]):
            self.current_coordinates[0] = 1
            self.current_coordinates[1] += 1

        elif (criteria[2] < self.padded_image_shape[2]):
            self.current_coordinates[0] = 1
            self.current_coordinates[1] = 1
            self.current_coordinates[2] += 1
        
        else:
            self.save_inference_result()
            self.check_end()
            if not self.is_end:
                self.next_patient()
    
    def create_result_tensor(self):
        return torch.zeros([self.n_classes] + self.padded_image_shape.tolist())


    def is_final_step(self):
        return np.all(self.current_coordinates * (self.patch_size * self.step_size + 1) >= self.images.shape) #다음번 grid가 넘칠 경우

    def save_inference_result(self):
        self.result_map = self.result_map[: , self.pad_bounds[0]:-self.pad_bounds[1], self.pad_bounds[2]:-self.pad_bounds[3], self.pad_bounds[4]:-self.pad_bounds[5]]
        self.result_map = torch.argmax(self.result_map.permute(1,2,3,0), dim = -1)
        if self.mode == 'save':
            np.savez(self.save_path+"result_"+"{0:003d}".format(self.current_index)+".npz", x = self.result_map.detach().cpu(), y = None)
        
        elif self.mode == 'dice':
            seg = self.dataset[self.current_index][0]
            self.result_map[torch.where(self.result_map == 0)] = -1
            intersection = (self.result_map == seg).sum()
            union = (self.result_map != -1).sum() + (seg != 0).sum()
            self.iter_score.append(intersection *2 / union)
    
    def check_end(self):
        if self.current_index == len(self.dataset) - 1:
            self.is_end = True

    def next_patient(self):
        self.current_index += 1
        self.current_coordinates = np.array([1,1,1])
        self.image_shape = self.get_image_shape(self.dataset[self.current_index][0])
        self.images = self.get_pad_images(self.dataset[self.current_index][0])
        self.padded_image_shape = self.get_image_shape(self.images)
        self.result_map = self.create_result_tensor()


    def get_image_shape(self, images):
        return torch.tensor([images.shape[1], images.shape[2], images.shape[3]])

    def get_pad_images(self, images):
        images = torch.tensor(images).to(self.device)
        prev_shape = torch.tensor(self.image_shape)
        patch_size = torch.tensor(self.patch_size)
        new_shape = (patch_size//2) - prev_shape % (patch_size//2)
        is_div_two = new_shape % 2 != 0
        pad_shape = new_shape.repeat_interleave(2) // 2
        for i in range(len(is_div_two)):
            if is_div_two[i]:
                pad_shape[i*2] += 1

        self.pad_bounds = pad_shape
        return F.pad(images, pad_shape.tolist()[::-1], "constant", 0)
    
    def create_3D_filter(self):
        tmp = tuple([torch.arange(i) for i in self.patch_size])
        gx, gy, gz = torch.meshgrid(*tmp)
        gx = gx - 1
        gy = gy - 1
        gz = gz - 1
        kernel = torch.exp(-((gx**2 + gy**2 + gz**2)/(2*(self.sigma**2))))

        return (kernel / kernel.sum()).view(1, *kernel.shape).to(self.device)