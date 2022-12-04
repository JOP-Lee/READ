import os
import numpy as np
import random
import torch
import cv2
import gzip

import torchvision

import matplotlib
import matplotlib.cm

from READ.models.compose import ModelAndLoss


def to_device(data, device='cuda:0'):
    if isinstance(data, torch.Tensor): 
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = to_device(data[k], device)

        return data
    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = to_device(data[i], device)

        return data
    
    return data


def set_requires_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value


def freeze(model, b):
    set_requires_grad(model, not b)


def save_model(save_path, model, args=None, compress=False):
    model = unwrap_model(model)

    if not isinstance(args, dict):
        args = vars(args)

    dict_to_save = { 
        'state_dict': model.state_dict(),
        'args': args
    }

    if compress:
        with gzip.open(f'{save_path}.gz', 'wb') as f:
            torch.save(dict_to_save, f, pickle_protocol=-1)
    else:
        torch.save(dict_to_save, save_path, pickle_protocol=-1)
        

def load_model_checkpoint(path, model):
    ckpt = torch.load(path, map_location='cpu')
    
    model.load_state_dict(ckpt['state_dict'])
        
    return model


def unwrap_model(model):
    model_ = model
    while True: 
        if isinstance(model_, torch.nn.DataParallel):
            model_ = model_.module
        elif isinstance(model_, ModelAndLoss):
            model_ = model_.model
        else:
            return model_


def colorize(value, vmin=0, vmax=1, cmap='viridis'):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    return np.ascontiguousarray(value[:, :, :3].transpose(2, 0, 1))


def resize(imgs, sz=256):
    return torch.nn.functional.interpolate(imgs, size=sz)


def to_numpy(t, flipy=False, uint8=True, i=0):
    out = t[:]
    if len(out.shape) == 4:
        out = out[i]
    out = out.detach().permute(1, 2, 0)
    out = out.flip([0]) if flipy else out
    out = out.detach().cpu().numpy()
    out = (out.clip(0, 1)*255).astype(np.uint8) if uint8 else out
    return out


def image_grid(*args, sz = 256):
    num_img = min( min([len(x) for x in args]), 4)

    grid = []
    for a in args:
        b = a[:num_img].detach().cpu().float()
        if b.shape[1] == 1:
            grid.append(torch.cat( [ torch.from_numpy(colorize(bb)).float()[None, ...]/255 for bb in b ], dim=0 ))
            # grid.append(torch.cat( [b, b, b], dim=1 ) )
        else: 
            grid.append(b[:, :3])

    # print([x.shape for x in grid ])
    imgs = resize( torch.cat(grid), sz=sz)
    x = torchvision.utils.make_grid(imgs, nrow = num_img)
    
    return x


def get_module(path):
    import pydoc

    m = pydoc.locate(path)
    assert m is not None, f'{path} not found'

    return m


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass