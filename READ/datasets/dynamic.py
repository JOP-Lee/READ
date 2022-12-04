import os, sys
import yaml
import multiprocessing
from functools import partial

from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms

import cv2
import numpy as np

from glumpy import app, gl

from READ.gl.render import OffscreenRender
from READ.gl.programs import NNScene
from READ.gl.utils import get_proj_matrix, load_scene, load_scene_data, setup_scene, FastRand
from READ.gl.dataset import parse_input_string

from READ.datasets.common import ToTensor, load_image, get_dataset_config, split_lists
from READ.utils.perform import TicToc, AccumDict



def rescale_K(K_, sx, sy, keep_fov=True):
    K = K_.copy()
    K[0, 2] = sx * K[0, 2]
    K[1, 2] = sy * K[1, 2]
    if keep_fov:
        K[0, 0] = sx * K[0, 0]
        K[1, 1] = sy * K[1, 1]
    return K


def rand_(min_, max_, *args):
    return min_ + (max_ - min_) * np.random.rand(*args)


default_input_transform = transforms.Compose([
        ToTensor(),
])

default_target_transform = transforms.Compose([
        ToTensor(),
])


class MultiscaleRender:
    def __init__(self, scene, input_format, viewport_size, proj_matrix=None, out_buffer_location='numpy', gl_frame=False, supersampling=1, clear_color=None):
        self.scene = scene
        self.input_format = input_format
        self.proj_matrix = proj_matrix
        self.gl_frame = gl_frame
        self.viewport_size = viewport_size
        self.ss = supersampling
    
        self.ofr = []
        for i in range(5):
            vs = self.ss * viewport_size[0] // 2 ** i, self.ss * viewport_size[1] // 2 ** i
            self.ofr.append(
                OffscreenRender(viewport_size=vs, out_buffer_location=out_buffer_location, clear_color=clear_color)
            )

    def render(self, view_matrix=None, proj_matrix=None, input_format=None):
        if view_matrix is not None:
            self.scene.set_camera_view(view_matrix)
        
        proj_matrix = self.proj_matrix if proj_matrix is None else proj_matrix
        if proj_matrix is not None:
            self.scene.set_proj_matrix(proj_matrix)
        
        self.scene.set_use_light(False)

        out_dict = {}
        input_format = input_format if input_format else self.input_format
        for fmt in input_format.replace(' ', '').split(','):
            config = parse_input_string(fmt)

            iscale = config['downscale'] if 'downscale' in config else 0

            self.scene.set_params(**config)
            x = self.ofr[iscale].render(self.scene)

            if not self.gl_frame:
                if torch.is_tensor(x):
                    x = x.flip([0])
                else:
                    x = x[::-1].copy()

            if ('depth' in fmt and not 'depth3' in fmt) or 'label' in fmt:
                x = x[..., :1]
            else:
                x = x[..., :3]

            out_dict[fmt] = x

        return out_dict


def get_rnd_crop_center_v1(mask, factor=8):
    mask_down = mask[::factor, ::factor]
    foregr_i, foregr_j = np.nonzero(mask_down)
    pnt_idx = np.random.choice(len(foregr_i))
    pnt = (foregr_i[pnt_idx] * factor, foregr_j[pnt_idx] * factor)
    return pnt


class DynamicDataset:
    znear = 0.1
    zfar = 1000
    
    def __init__(self, scene_data, input_format, image_size,
                 view_list, target_list, mask_list, label_list,
                 keep_fov=False, gl_frame=False,
                 input_transform=None, target_transform=None,
                 num_samples=None,
                 random_zoom=None, random_shift=None,
                 drop_points=0., perturb_points=0.,
                 label_in_input=False,
                 crop_by_mask=False,
                 use_mesh=False,
                 supersampling=1):

        if isinstance(image_size, (int, float)):
            image_size = image_size, image_size
        
        # if render image size is different from camera image size, then shift principal point
        K_src = scene_data['intrinsic_matrix']
        old_size = scene_data['config']['viewport_size']
        sx = image_size[0] / old_size[0]
        sy = image_size[1] / old_size[1]
        K = rescale_K(K_src, sx, sy, keep_fov)
        
        assert len(view_list) == len(target_list)

        self.view_list = view_list
        self.target_list = target_list
        self.mask_list = mask_list
        self.label_list = label_list
        self.scene_data = scene_data
        self.input_format = input_format
        self.image_size = image_size
        self.renderer = None
        self.scene = None
        self.K = K
        self.K_src = K_src
        self.random_zoom = random_zoom
        self.random_shift = random_shift
        self.sx = sx
        self.sy = sy
        self.keep_fov = keep_fov
        self.gl_frame = gl_frame
        self.target_list = target_list
        self.input_transform = default_input_transform if input_transform is None else input_transform
        self.target_transform = default_target_transform if target_transform is None else target_transform
        self.num_samples = num_samples if num_samples else len(view_list)
        self.id = None
        self.name = None
        self.drop_points = drop_points
        self.perturb_points = perturb_points
        self.label_in_input = label_in_input
        self.crop_by_mask = crop_by_mask
        self.use_mesh = use_mesh
        self.ss = supersampling

        self.fastrand = None
        self.timing = None
        self.count = 0

    def load(self):
        self.scene = NNScene()
        setup_scene(self.scene, self.scene_data, use_mesh=self.use_mesh)

        if self.perturb_points and self.fastrand is None:
            print(f'SETTING PERTURB POINTS: {self.perturb_points}')
            tform = lambda p: self.perturb_points * (p - 0.5)
            self.fastrand = FastRand((self.scene_data['pointcloud']['xyz'].shape[0], 2), tform, 10) 

    def unload(self):
        self.scene.delete()
        self.scene = None
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # assert -1 < idx < len(self)
        idx = idx % len(self.view_list)
        
        # we want to make sure GL and CUDA Interop contexts are created in
        # the process calling __getitem__ method, otherwise rendering would not work
        if self.renderer is None:
            assert self.scene is not None, 'call load()'
            app.Window(visible=False) # creates GL context
            self.renderer = MultiscaleRender(self.scene, self.input_format, self.image_size, supersampling=self.ss)

        if self.timing is None:
            self.timing = AccumDict()
        
        tt = TicToc()
        tt.tic()

        mask = None
        mask_crop = None
        if self.mask_list[idx]:
            mask = load_image(self.mask_list[idx])

            if self.crop_by_mask:
                cnt = get_rnd_crop_center_v1(mask[..., 0])
                mask_crop = -1 + 2 * np.array(cnt) / mask.shape[:2]

        view_matrix = self.view_list[idx]
        K, proj_matrix = self._get_intrinsics(shift=mask_crop)

        target = load_image(self.target_list[idx])
        target = self._warp(target, K)

        if mask is None:
            mask = np.ones((target.shape[0], target.shape[1], 1), dtype=np.float32)
        else:
            mask = self._warp(mask, K)

        if self.label_list[idx]:
            label = load_image(self.label_list[idx])
            label = self._warp(label, K)
            label = label[..., :1]
        else:
            label = np.zeros((target.shape[0], target.shape[1], 1), dtype=np.uint8)
        
        self.timing.add('get_target', tt.toc())
        tt.tic()

        if self.drop_points:
            self.scene.set_point_discard(np.random.rand(self.scene_data['pointcloud']['xyz'].shape[0]) < self.drop_points)

        if self.perturb_points:
            self.scene.set_point_perturb(self.fastrand.toss())
        
        input_ = self.renderer.render(view_matrix=view_matrix, proj_matrix=proj_matrix)

        if self.label_in_input:
            for k in input_:
                if 'labels' in k:
                    m = input_[k].sum(2) > 1e-9
                    label_sz = cv2.resize(label, (input_[k].shape[1], input_[k].shape[0]), interpolation=cv2.INTER_NEAREST)
                    label_m = label_sz * m
                    input_[k] = label_m[..., None]
        
        self.timing.add('render', tt.toc())
        tt.tic()
        
        input_ = {k: self.input_transform(v) for k, v in input_.items()}
        target = self.target_transform(target)
        mask = ToTensor()(mask)
        label = ToTensor()(label)

        input_['id'] = self.id
        
        self.timing.add('transform', tt.toc())

        # if self.count and self.count % 100 == 0:
        #     print(self.timing)

        self.count += 1
        
        return {'input': input_,
                'view_matrix': view_matrix,
                'intrinsic_matrix': K,
                'target': target,
                'target_filename': self.target_list[idx],
                'mask': mask,
                'label': label
               }

    def _get_intrinsics(self, shift=None):
        K = self.K.copy()
        sx = 1. if self.keep_fov else self.sx
        sy = 1. if self.keep_fov else self.sy
        if self.random_zoom:
            z = rand_(*self.random_zoom)
            K[0, 0] *= z
            K[1, 1] *= z
            sx /= z
            sy /= z
        if self.random_shift:
            if shift is None:
                x, y = rand_(*self.random_shift, 2)
            else:
                x, y = shift
            w = self.image_size[0] * (1. - sx) / sx / 2.
            h = self.image_size[1] * (1. - sy) / sy / 2.


            K[0, 2] += x * w
            K[1, 2] += y * h
            
        return K, get_proj_matrix(K, self.image_size, self.znear, self.zfar)
    
    def _warp(self, image, K):
        H = K @ np.linalg.inv(self.K_src)
        image = cv2.warpPerspective(image, H, tuple(self.image_size))
        
        if self.gl_frame:
            image = image[::-1].copy()

        return image


def get_datasets(args):
    assert args.paths_file, 'set paths'
    # assert args.dataset_names, 'set dataset_names'

    with open(args.paths_file) as f:
        paths_data = yaml.load(f,Loader=yaml.FullLoader)

    if not args.dataset_names:
        print('Using all datasets')
        args.dataset_names = list(paths_data['datasets'])

    if args.exclude_datasets:
        args.dataset_names = list(set(args.dataset_names) - set(args.exclude_datasets))

    pool = multiprocessing.Pool(32)
    map_fn = partial(_load_dataset, paths_data=paths_data, args=args)
    pool_out = pool.map_async(map_fn, args.dataset_names)
    
    # pool_out = [_load_dataset(tasks[0])]

    ds_train_list, ds_val_list = [], []

    for ds_train, ds_val in pool_out.get():
        ds_train_list.append(ds_train)
        ds_val_list.append(ds_val)

        print(f'ds_train: {len(ds_train)}')
        print(f'ds_val: {len(ds_val)}')    

    # for name in args.dataset_names:
        # print(f'creating dataset {name}')
        
        # ds_train, ds_val = _get_splits(paths_data, name, args)

        # ds_train.name = ds_val.name = name
        # ds_train.id = ds_val.id = args.dataset_names.index(name)

        # ds_train_list.append(ds_train)
        # ds_val_list.append(ds_val)

        # print(f'ds_train: {len(ds_train)}')
        # print(f'ds_val: {len(ds_val)}')

    pool.close()

    return ds_train_list, ds_val_list


def _load_dataset(name, paths_data, args):
    ds_train, ds_val = _get_splits(paths_data, name, args)

    ds_train.name = ds_val.name = name
    ds_train.id = ds_val.id = args.dataset_names.index(name)

    return ds_train, ds_val


def _get_splits(paths_file, ds_name, args):
    config = get_dataset_config(paths_file, ds_name)

    scene_path = config['scene_path']
    assert args.input_format, 'specify input format'
    input_format = args.input_format

    scene_data = load_scene_data(scene_path)

    view_list = scene_data['view_matrix']
    camera_labels = scene_data['camera_labels']

    if 'target_name_func' in config:
        target_name_func = eval(config['target_name_func'])
    else:
        target_name_func = lambda i: f'{i:06}.png'
    
    target_list = [os.path.join(config['target_path'], target_name_func(i)) for i in camera_labels]


    if 'mask_path' in config:
        mask_name_func = eval(config['mask_name_func'])
        mask_list = [os.path.join(config['mask_path'], mask_name_func(i)) for i in camera_labels]

    else:
        mask_list = [None] * len(target_list)

    if 'label_path' in config:
        label_name_func = eval(config['label_name_func'])
        label_list = [os.path.join(config['label_path'], label_name_func(i)) for i in camera_labels]

    else:
        label_list = [None] * len(target_list)

    assert hasattr(args, 'splitter_module') and hasattr(args, 'splitter_args')

    splits = args.splitter_module([view_list, target_list, mask_list, label_list], **args.splitter_args)

    view_list_train, view_list_val = splits[0]
    target_list_train, target_list_val = splits[1]
    mask_list_train, mask_list_val = splits[2]
    label_list_train, label_list_val = splits[3]

    # num_samples_train = int(config['num_samples_train']) if 'num_samples_train' in config else None

    ds_train = DynamicDataset(scene_data, input_format, args.crop_size, view_list_train, target_list_train, mask_list_train, label_list_train,
        use_mesh=args.use_mesh, supersampling=args.supersampling, **args.train_dataset_args)

    ds_val = DynamicDataset(scene_data, input_format, args.crop_size, view_list_val, target_list_val, mask_list_val, label_list_val,
        use_mesh=args.use_mesh, supersampling=args.supersampling, **args.val_dataset_args)

    return ds_train, ds_val
