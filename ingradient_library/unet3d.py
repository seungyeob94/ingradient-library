
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels = 4, output_channels = 1,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))



class UNet3DDeepsupervision(nn.Module):
    def __init__(self, n_modalities = 4, patch_size = (128, 128, 128), final_output_channels = 5):
        super().__init__()
        self.patch_size = np.array(patch_size)
        self.networks_depth = self.get_networks_depth()
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.upsample_operations = []
        self.final_output_channels = final_output_channels
        n_input_channels = n_modalities
        n_output_channels = 32
        
        conv_kwargs1 = {'kernel_size' : 3, 'padding' : 1, 'stride' : 1}
        self.encoder1 = nn.Sequential(
            ConvDropoutNormNonlin(4, 32, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(32, 32, conv_kwargs = conv_kwargs1, conv_op = nn.Conv3d)
        )
        conv_kwargs1 = {'kernel_size' : 3, 'padding' : 1, 'stride' : 2}
        conv_kwargs2 = {'kernel_size' : 3, 'padding' : 1, 'stride' : 1}
        self.encoder2 = nn.Sequential(
            ConvDropoutNormNonlin(32, 64, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(64, 64, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )
        self.encoder3 = nn.Sequential(
            ConvDropoutNormNonlin(64, 128, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(128, 128, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )
        self.encoder4 = nn.Sequential(
            ConvDropoutNormNonlin(128, 256, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(256, 256, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )
        self.encoder5 = nn.Sequential(
            ConvDropoutNormNonlin(256, 320, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(320, 320, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )

        conv_kwargs1 = {'kernel_size' : 3, 'padding' : 1,  'stride' : 1}
        conv_kwargs2 = {'kernel_size' : 3, 'stride' : 1, 'padding':1}
        transp_kwargs = {'kernel_size' : 2, 'stride' : 1, 'stride' : 2}


        self.decoder1_trans = ConvDropoutNormNonlin(320, 256, conv_kwargs= transp_kwargs, conv_op = nn.ConvTranspose3d)
        self.decoder1_conv = nn.Sequential(
            ConvDropoutNormNonlin(256*2, 256, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(256, 256, conv_kwargs = conv_kwargs2, conv_op = nn.Conv3d)
        )
        self.decoder2_trans = ConvDropoutNormNonlin(256, 128, conv_kwargs= transp_kwargs, conv_op = nn.ConvTranspose3d)
        self.decoder2_conv = nn.Sequential(
            ConvDropoutNormNonlin(128*2, 128 , conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(128, 128, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )
        self.decoder3_trans = ConvDropoutNormNonlin(128, 64, conv_kwargs= transp_kwargs, conv_op = nn.ConvTranspose3d)
        self.decoder3_conv = nn.Sequential(
            ConvDropoutNormNonlin(64*2, 64, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(64, 64, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )
        self.decoder4_trans = ConvDropoutNormNonlin(64, 32, conv_kwargs= transp_kwargs, conv_op = nn.ConvTranspose3d)
        self.decoder4_conv = nn.Sequential(
            ConvDropoutNormNonlin(32*2, 32, conv_kwargs= conv_kwargs1, conv_op = nn.Conv3d),
            ConvDropoutNormNonlin(32, 32, conv_kwargs= conv_kwargs2, conv_op = nn.Conv3d)
        )

        self.final_layer1 = nn.Conv3d(32, final_output_channels, 3, 1, 1)
        self.final_layer2 = nn.Conv3d(64, final_output_channels, 3, 1, 1)
        self.final_layer3 = nn.Conv3d(128, final_output_channels, 3, 1, 1)
        self.final_layer4 = nn.Conv3d(256, final_output_channels, 3, 1, 1)

        self.upsample4 = nn.Upsample(scale_factor=8, mode = 'trilinear')
        self.upsample3 = nn.Upsample(scale_factor=4, mode = 'trilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode = 'trilinear')
        
    
    def forward(self, x):
        bs, nc, nx, ny, nz  = x.shape
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x4_ = self.decoder1_conv(torch.cat((x4, self.decoder1_trans(x5)), dim = 1))
        x3_ = self.decoder2_conv(torch.cat((x3, self.decoder2_trans(x4_)), dim = 1))
        x2_ = self.decoder3_conv(torch.cat((x2, self.decoder3_trans(x3_)), dim = 1))
        x1_ = self.decoder4_conv(torch.cat((x1, self.decoder4_trans(x2_)), dim = 1))

        x1_ = F.softmax(self.final_layer1(x1_).view(bs, self.final_output_channels, -1), dim = 1).view(bs, self.final_output_channels, nx, ny, nz)
        x2_ = F.softmax(self.upsample2(self.final_layer2(x2_)).view(bs, self.final_output_channels, -1), dim = 1).view(bs, self.final_output_channels, nx, ny, nz)
        x3_ = F.softmax(self.upsample3(self.final_layer3(x3_)).view(bs, self.final_output_channels, -1), dim = 1).view(bs, self.final_output_channels, nx, ny, nz)
        x4_ = F.softmax(self.upsample4(self.final_layer4(x4_)).view(bs, self.final_output_channels, -1), dim = 1).view(bs, self.final_output_channels, nx, ny, nz)

        result = torch.stack([x1_, x2_, x3_, x4_]).permute(1,0,2,3,4,5)
        
        return result.contiguous()
                
    def get_networks_depth(self):
        min_patch_axis = np.min(self.patch_size)
        return np.clip(np.log2(min_patch_axis), 0, 6).astype(int)
    

