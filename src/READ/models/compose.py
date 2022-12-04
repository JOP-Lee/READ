import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import cv2
from PIL import Image

import pdb


class ModelAndLoss(nn.Module):
    def __init__(self, model, loss, use_mask=False):
        super().__init__()
        self.model = model
        self.loss = loss
        self.use_mask = use_mask

    def forward(self, *args, **kwargs):
        input = args[:-1]
        target = args[-1]
        # label =
        if not isinstance(input, (tuple, list)):
            input = [input]
        output = self.model(*input, **kwargs)

        im_out = output['im_out']


        loss = {}
        if self.use_mask and 'mask' in kwargs and kwargs['mask'] is not None:
            loss['vgg_loss'] = self.loss(im_out * kwargs['mask'], target)
            loss['huber_loss'] = F.huber_loss(im_out * kwargs['mask'], target)
        else:
            loss['vgg_loss'] = self.loss(im_out, target)
            loss['huber_loss'] = F.huber_loss(im_out, target)
        if 'seg_out' in output and 'label' in kwargs and kwargs['label'] is not None:
            loss['seg_loss'] = F.cross_entropy(output['seg_out'], kwargs['label'], ignore_index=0)

        return output, loss


class BoxFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None)
        )

        self.weights_init(kernel_size)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, kernel_size):
        kernel = torch.ones((kernel_size, kernel_size)) / kernel_size ** 2
        self.seq[1].weight.data.copy_(kernel)


class GaussianLayer(nn.Module):
    _instance = None

    def __init__(self, in_channels, out_channels, kernel_size=21, sigma=3):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=8)
        )

        self.weights_init(kernel_size, sigma)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, kernel_size, sigma):
        n= np.zeros((kernel_size, kernel_size))
        n[kernel_size//2, kernel_size//2] = 1
        k = gaussian_filter(n,sigma=sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

    @staticmethod
    def get_instance():
        if GaussianLayer._instance is None:
            GaussianLayer._instance = GaussianLayer(8, 8, kernel_size=13, sigma=6).cuda()

        return GaussianLayer._instance


class NetAndTexture(nn.Module):
    def __init__(self, net, textures, supersampling=1, temporal_average=False):
        super().__init__()
        self.net = net
        self.ss = supersampling

        try:
            textures = dict(textures)
        except TypeError:
            textures = {0: textures}

        self._textures = {k: v.cpu() for k, v in textures.items()}
        self._loaded_textures = []

        self.last_input = None
        self.temporal_average = temporal_average

    def load_textures(self, texture_ids):
        if torch.is_tensor(texture_ids):
            texture_ids = texture_ids.cpu().tolist()
        elif isinstance(texture_ids, int):
            texture_ids = [texture_ids]

        for tid in texture_ids:
            self._modules[str(tid)] = self._textures[tid]
        self._loaded_textures = texture_ids

    def unload_textures(self):
        for tid in self._loaded_textures:
            self._modules[str(tid)].cpu()
            del self._modules[str(tid)]

    def reg_loss(self):
        loss = 0
        for tid in self._loaded_textures:
            loss += self._modules[str(tid)].reg_loss()

        return loss

    def forward(self, inputs, **kwargs):
        outs = {'im_out':[]}
        # outs = {'x1':[],'x2':[],'x4':[],}
        texture_ids = inputs['id']
        del inputs['id']
        if torch.is_tensor(texture_ids):
            texture_ids = texture_ids.tolist()
        elif isinstance(texture_ids, int):
            texture_ids = [texture_ids]

        for i, tid in enumerate(texture_ids): # per item in batch
            input = {k: v[i][None] for k, v in inputs.items()}
            assert 'uv' in list(input)[0], 'first input must be uv'

            texture = self._modules[str(tid)]
            j = 0
            keys = list(input)
            input_multiscale = []
            while j < len(keys): # sample texture at multiple scales
                tex_sample = None
                input_ex = []
                if 'uv' in keys[j]:
                    tex_sample = texture(input[keys[j]])
                    j += 1
                    while j < len(keys) and 'uv' not in keys[j]:
                        input_ex.append(input[keys[j]])
                        j += 1
                assert tex_sample is not None
                input_cat = torch.cat(input_ex + [tex_sample], 1)


                if self.ss > 1:
                    input_cat = nn.functional.interpolate(input_cat, scale_factor=1./self.ss, mode='bilinear')

                input_multiscale.append(input_cat)
            
            if self.temporal_average:
                if self.last_input is not None:
                    for i in range(len(input_multiscale)):
                        input_multiscale[i] = (input_multiscale[i] + self.last_input[i]) / 2
                self.last_input = list(input_multiscale)

            out = self.net(*input_multiscale, **kwargs)
            outs['im_out'].append(out['im_out'])
            if 'seg_out' in out:
                if 'seg_out' not in outs:
                    outs['seg_out'] = []
                outs['seg_out'].append(out['seg_out'])


        if 'seg_out' in outs and len(outs['seg_out']) == len(outs['im_out']):
            outs['seg_out'] = torch.cat(outs['seg_out'], 0)
        outs['im_out'] = torch.cat(outs['im_out'], 0)


        
        if kwargs.get('return_input'):
            return outs, input_multiscale
        else:
            return outs


class MultiscaleNet(nn.Module):
    def __init__(self, net, input_modality, supersampling=1):
        super().__init__()
        
        self.net = net
        self.input_modality = input_modality
        self.ss = supersampling

    def forward(self, inputs, **kwargs):
        del inputs['id']

        modes = len(inputs)
        assert modes % self.input_modality == 0

        inputs_ms = []
        input_values = list(inputs.values())
        for i in range(modes // self.input_modality):
            i0 = i * self.input_modality
            i1 = (i + 1) * self.input_modality
            cat = torch.cat(input_values[i0:i1], 1)
            if self.ss > 1:
                cat = nn.functional.interpolate(cat, scale_factor=1./self.ss, mode='bilinear')
            inputs_ms.append(cat)

        out = self.net(*inputs_ms, **kwargs)

        if kwargs.get('return_input'):
            return out, inputs_ms
        else:
            return out


class RGBTexture(nn.Module):
    def __init__(self, texture, supersampling=1):
        super().__init__()

        self.texture = texture
        self.ss = supersampling

    def forward(self, inputs, **kwargs):
        del inputs['id']

        assert list(inputs) == ['uv_2d'], 'check input format'

        uv = inputs['uv_2d']
        out = self.texture(uv)

        if kwargs.get('return_input'):
            return out, uv
        else:
            return out
