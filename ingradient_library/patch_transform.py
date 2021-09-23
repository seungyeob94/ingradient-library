
import torch
import torch.nn.functional as F
import numpy as np
import math

class Batch_Affine_3D(object):
    def __init__(self, degree = [30, 30, 30], axis = [0,1,2], scale = [0.7, 1.4], use_gpu = True, is_3d_img = True, device = None, cal_time = True):
        self.use_gpu = use_gpu
        self.transform_matrices = []
        
        self.axis = axis
        self.degree = np.array(degree)
        self.scale = np.array(scale)
        
        if self.use_gpu:
            self.device = device
    

    def get_matrices_and_coords(self, images):
        if len(images.shape) == 5:
            seg = False
        else:
            seg = True
        if not seg:
            bs, n_modalities, nx, ny, nz = images.shape
        else:
            bs, nx, ny, nz = images.shape
        img_size = [nx,ny,nz] #patch size이기 때문에 동일
        self.coords = self.create_coords(img_size) #Coordinates 생성
        self.matrices = self.get_matrices()

    
    def __call__(self, images):

        if len(images.shape) == 5:
            seg = False
        else:
            seg = True
        if not seg:
            bs, n_modalities, nx, ny, nz = images.shape
        else:
            bs, nx, ny, nz = images.shape
        img_size = [nx,ny,nz] #patch size이기 때문에 동일

        for batch_index in range(bs):
            transformed_coords = self.gpu_calculate_coornidates(self.coords, self.matrices, img_size) #coorniates를 Affine 변환
            if not seg:
                images[batch_index, :, :, :, :] = images[batch_index, :, transformed_coords[0], transformed_coords[1], transformed_coords[2]]
                images = images.reshape(bs, n_modalities, nx, ny, nz)
            else:
                images[batch_index, :, :, :] = images[batch_index, transformed_coords[0], transformed_coords[1], transformed_coords[2]]
                images = images.reshape(bs, nx, ny, nz)
            
        return images
    
    def set_rotate_matrix(self, axis, degree):
        function_list = [self.rotation_x_3d, self.rotation_y_3d, self.rotation_z_3d]
        return function_list[axis](degree)

    def set_scale_matrix(self, scale):
        return torch.eye(3) * 1/scale

    def rotation_x_3d(self, angle):
        return torch.Tensor([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

    def rotation_y_3d(self, angle):
        return torch.Tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

    def rotation_z_3d(self, angle):
        return torch.Tensor([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    def create_coords(self, img_shape):
        tmp = tuple([torch.arange(i) for i in img_shape])
        grid_x, grid_y, grid_z = torch.meshgrid(*tmp)

        return torch.vstack((grid_x.unsqueeze_(0), grid_y.unsqueeze_(0), grid_z.unsqueeze_(0)))
    
    def set_random_scale_degree(self):
        degree = np.random.uniform( size = self.degree.shape) * self.degree
        if np.any(self.scale):
            scale = np.random.uniform(low = self.scale[0], high = self.scale[1], size = (1,3))
        
        else:
            scale = 1
        return scale, degree

    def get_matrices(self):
        transform_matrices = []
        scale, degree = self.set_random_scale_degree()
        for a, d in zip(self.axis, degree):
            transform_matrices.append(self.set_rotate_matrix(a, d))#Rotation Matrix 생성

        if isinstance(scale, np.ndarray):
            transform_matrices.append(self.set_scale_matrix(scale))

        return torch.stack(transform_matrices)

    def gpu_calculate_coornidates(self, coords, matrices, img_size):
        before_shape = coords.shape
        coords = coords.to(self.device).double()
        coords_bound = np.array(img_size).reshape(-1,1)-1 #원본 이미지의 최소, 최대 좌표

        half_img_size = ((torch.Tensor(img_size).to(self.device) - 1)/2).reshape(3,1,1,1)
        coords = coords - half_img_size
        matrices = matrices.to(self.device).double()

        for i in range(len(matrices)-1):
            matrices[i+1] = torch.matmul(matrices[i], matrices[i+1])
        coords = torch.matmul(matrices[-1], coords.reshape(3, -1)).reshape(*before_shape)
        coords = coords + half_img_size
        coords = coords.reshape(3, -1).round() #Mapping 해주는 함수
        coords = torch.clamp(coords, torch.Tensor([[0], [0], [0]]).to(0).int(), torch.Tensor(coords_bound).to(0).int()).long()
        
        return coords.reshape(*before_shape)


class Batch_Gaussian_Noise(object):
    def __init__(self, device = 0, prob_per_modalities = 0.5):
        self.device = device
        self.prob_per_modalities = prob_per_modalities
    
    def __call__(self, images):
        if len(images.shape) == 4:
            return images

        bs, n_modalities, nx, ny, nz = images.shape
        do_operation = np.random.uniform(0, 1, size = n_modalities) < self.prob_per_modalities

        for i in range(n_modalities):
            if i == 0:
                if do_operation[i]:
                    noise_array = torch.randn((nx,ny,nz)).unsqueeze(0)
                else:
                    noise_array = torch.zeros((nx,ny,nz)).unsqueeze(0)
                continue
            if do_operation[i]:
                noise_array = torch.vstack((noise_array, torch.randn((nx,ny,nz)).unsqueeze(0)))
            else:
                noise_array = torch.vstack((noise_array, torch.zeros((nx,ny,nz)).unsqueeze(0)))
        noise_array = torch.tile(noise_array, (bs, 1,1,1,1)).to(self.device)
        if images.device.index != self.device:
            images = images.to(self.device)

        return images + noise_array


class Batch_Gaussian_Blur_3D(object):
    def __init__(self, sigma = 1.2, width = 3, prob_per_modality = 0.5, device = 0):
        array = np.random.uniform(size = 4)
        self.device = device
        gaussian_filter = self.get_kernel(np.random.uniform(1.0, sigma), width)
        self.prob_per_modality = prob_per_modality
        self.weight = gaussian_filter.unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, images):
        #각 모달리티 별로 다른 연산을 지원한다.

        batch_size = images.shape[0]
        modality_size = images.shape[1]
        do_per_modality = np.random.uniform(0,1, size = modality_size)

        if len(images.shape) == 4:
            return images
    
        result = []
        images = images.permute(1, 0, 2, 3, 4).unsqueeze(1) #n_modalities, batch_size

        for i in range(modality_size):
            if do_per_modality[i] > self.prob_per_modality:
                continue
            images[i] = F.conv3d(images[i], weight = torch.tile(self.weight, (batch_size, 1, 1, 1, 1)), stride = 1, padding = 1, groups = batch_size)
        

        images = images.squeeze(1).permute(1,0,2,3,4)
    
        return images


    def get_kernel(self, sigma, width):
        kernel_shape = (width, width, width)
        tmp = tuple([torch.arange(i) for i in kernel_shape])
        gx, gy, gz = torch.meshgrid(*tmp)
        gx = gx - 1
        gy = gy - 1
        gz = gz - 1
        kernel = torch.exp(-((gx**2 + gy**2 + gz**2)/(2*(sigma**2))))

        return (kernel / kernel.sum())


class Batch_Brightness(object):
  def __init__(self, device = None, rng = [0.7,1.3]):
    self.device = device
    self.rng = rng
    
  def __call__(self, images):
    x = np.random.uniform(self.rng[0], self.rng[1])

    return images * x


class Batch_GammaTransform(object):
    def __init__(self, gamma_range = (0.5, 2), epsilon = 1e-7, device = None, retain_stats = True):
        self.gamma_range = gamma_range
        self.device = device
        self.epsilon = epsilon
        self.retain_stats = retain_stats

    def __call__(self, images):
        if len(images.shape) == 4:
            return images
        bs, n_modalities, nx, ny, nz = images.shape
        
        if np.random.random() < 0.5 and self.gamma_range[0] < 1:
            gamma = torch.FloatTensor(bs, n_modalities,1,1,1).uniform_(self.gamma_range[0], 1).to(self.device)
        else:
            gamma = torch.FloatTensor(bs, n_modalities,1,1,1).uniform_(max(self.gamma_range[0], 1), self.gamma_range[1]).to(self.device)
        
        if self.retain_stats:
            before_mmean = torch.mean(images.view(bs, n_modalities, -1), 2).view(bs, n_modalities, 1, 1, 1)
            before_mstd = torch.std(images.view(bs, n_modalities, -1), 2).view(bs, n_modalities, 1, 1, 1)
        mmax = torch.max(images.view(bs, n_modalities, -1), dim = 2)[0].view(bs, n_modalities, 1, 1, 1)
        mmin = torch.min(images.view(bs, n_modalities, -1), dim = 2)[0].view(bs, n_modalities, 1, 1, 1)
        
        mrange = mmax - mmin 
        temp = (mrange + self.epsilon)
        
        images = torch.pow(((images - mmin) / temp), gamma) * temp + mmin
        

        if self.retain_stats:
            after_mmean = torch.mean(images.view(bs, n_modalities, -1), 2).view(bs, n_modalities, 1, 1, 1)
            after_mstd = torch.std(images.view(bs, n_modalities, -1), 2).view(bs, n_modalities, 1, 1, 1)
            images = images - after_mmean
            images = images / (after_mstd + 1e-8) * before_mstd
            images = images + before_mmean
        
        return images

class Batch_Contrast(object):
  def __init__(self, device = None, contrast_range = [0.75, 1.25], preserve_range = True):
    self.device = device
    self.preserve_range = preserve_range
    self.contrast_range = contrast_range
    
  def __call__(self, images):
    if len(images.shape) == 4:
            return images
    bs, n_modalities, nx, ny, nz = images.shape
    
    if np.random.random() < 0.5 and self.contrast_range[0] < 1:
        factor = torch.FloatTensor(bs, n_modalities,1,1,1).uniform_(self.contrast_range[0], 1).to(self.device)
    else:
        factor = torch.FloatTensor(bs, n_modalities,1,1,1).uniform_(max(self.contrast_range[0], 1), self.contrast_range[1]).to(self.device)

    
    mmean = torch.mean(images.view(n_modalities, -1), 1).view(n_modalities, 1, 1, 1)
    mmax = torch.max(images.view(n_modalities, -1), dim = 1)[0].view(n_modalities, 1, 1, 1)
    mmin = torch.min(images.view(n_modalities, -1), dim = 1)[0].view(n_modalities, 1, 1, 1)
    images = (images - mmean ) * factor + mmean
    if self.preserve_range:
        images = torch.where(images > mmax, mmax, images)
        images = torch.where(images < mmin, mmin, images)

    return images

    
class Batch_Mirroring(object):
    def __init__(self, x_prob = 0.5, y_prob = 0.5, z_prob = 0.5):
        self.x_prob = x_prob
        self.y_prob = y_prob
        self.z_prob = z_prob

    def get_mirror_axis(self):
        do_mirror = np.random.uniform(0,1, size = 3)
        self.mirror_axis = []
        if do_mirror[0] < self.x_prob:
            self.mirror_axis.append(-3)
        
        if do_mirror[1] < self.y_prob:
            self.mirror_axis.append(-2)
        
        if do_mirror[2] < self.z_prob:
            self.mirror_axis.append(-1)


    def __call__(self, images):
        return torch.flip(images, self.mirror_axis)
        