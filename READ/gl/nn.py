import os, sys
try:
    import torch
except ImportError:
    print('torch is not available')
import numpy as np

from READ.gl.render import OffscreenRender
# from READ.gl.utils import get_net

from READ.pipelines import load_pipeline
from READ.datasets.dynamic import MultiscaleRender, default_input_transform

from scipy.ndimage import gaussian_filter
import torch.nn as nn


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


class BoxFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=8)
        )

        self.weights_init(kernel_size)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, kernel_size):
        kernel = torch.ones((kernel_size, kernel_size)) / kernel_size ** 2
        print(self.seq[0].named_parameters())


def to_gpu(data):
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].cuda()
        return data
    else:
        return data.cuda()


class OGL:
    def __init__(self, scene, scene_data, viewport_size, net_ckpt, texture_ckpt, out_buffer_location='numpy', supersampling=1, gpu=True, clear_color=None, temporal_average=False):
        self.gpu = gpu

        args_upd = {
            'inference': True,
        }

        if texture_ckpt:
            args_upd['texture_ckpt'] = texture_ckpt
            if 'pointcloud' in scene_data:
                args_upd['n_points'] = scene_data['pointcloud']['xyz'].shape[0]
        
        pipeline, args = load_pipeline(net_ckpt, args_to_update=args_upd)

        self.model = pipeline.model
        
        if args.pipeline == 'READ.pipelines.ogl.TexturePipeline':
            self.model.load_textures(0)

        if self.gpu:
            self.model.cuda()
        self.model.eval()

        if supersampling > 1:
            self.model.ss = supersampling

        self.model.temporal_average = temporal_average

        print(f"SUPERSAMPLING: {self.model.ss}")

        factor = 16
        assert viewport_size[0] % 16 == 0, f'set width {factor * (viewport_size[0] // factor)}'
        assert viewport_size[1] % 16 == 0, f'set height {factor * (viewport_size[1] // factor)}'

        self.renderer = MultiscaleRender(scene, args.input_format, viewport_size, out_buffer_location=out_buffer_location, supersampling=self.model.ss, clear_color=clear_color)

    def infer(self):
        input_dict = self.renderer.render()
        input_dict = {k: default_input_transform(v)[None] for k, v in input_dict.items()}
        if self.gpu:
            input_dict = to_gpu(input_dict)

        input_dict['id'] = 0
        with torch.set_grad_enabled(False):
            out, net_input = self.model(input_dict, return_input=True)

        out = out[0].detach().permute(1, 2, 0)
        out = torch.cat( [out, out[:, :, :1] * 0 + 1], 2 ).contiguous() # expected to have 4 channels

        return {
            'output': out,
            'net_input': net_input
        }
