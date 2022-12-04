import os, sys
from cv2 import phase
import yaml
import multiprocessing
from functools import partial

from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.transforms import ToTensor

import cv2
import numpy as np


from READ.gl.render import OffscreenRender
from READ.gl.programs import NNScene
from READ.gl.utils import get_proj_matrix, load_scene, load_scene_data, setup_scene, FastRand
from READ.gl.dataset import parse_input_string
# ToTensor,
from READ.datasets.common import  load_image, get_dataset_config, split_lists
from READ.utils.perform import TicToc, AccumDict

import pdb


def rescale_K(K_, s, keep_fov=True):
    K = K_.copy()
    K[0, 2] = s[0] * K[0, 2]
    K[1, 2] = s[1] * K[1, 2]
    if keep_fov:
        K[0, 0] = s[0] * K[0, 0]
        K[1, 1] = s[1] * K[1, 1]
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
            # set config input_format uv1d with mode0=3, mode1=0
            
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
    
    def __init__(self, phase, scene_data, input_format, image_size,
                 view_list, target_list, mask_list, label_list,
                 keep_fov=False, gl_frame=False,
                 input_transform=None, target_transform=None,
                 num_samples=None,
                 inner_batch=None,
                 random_zoom=None, random_shift=None,
                 drop_points=0., perturb_points=0.,
                 label_in_input=False,
                 crop_by_mask=False,
                 use_mesh=False,
                 supersampling=1, headless=False):

        self.phase = phase
        
        if isinstance(image_size, (int, float)):
            image_size = image_size, image_size
        
        # if render image size is different from camera image size, then shift principal point
        K_src = scene_data['intrinsic_matrix']
        old_size = scene_data['config']['viewport_size']
        self.src_sh = np.array(old_size)
        self.tgt_sh = np.array(list(map(lambda x:x//2**4 * 2**4, self.src_sh)))
        # self.tgt_sh = np.array([512,512])
        if phase=='train':
            self.tgt_sh = np.array(image_size)
        assert len(view_list) == len(target_list)
        
        print(f'{phase}_tgt_size', self.tgt_sh)
        self.view_list = view_list
        self.target_list = target_list
        self.mask_list = mask_list
        self.label_list = label_list
        self.scene_data = scene_data
        self.input_format = input_format
        self.headless = headless
        # True
        self.renderer = None
        self.scene = None
        self.K_src = K_src
        self.random_zoom = random_zoom
        self.random_shift = random_shift
        self.keep_fov = keep_fov
        self.gl_frame = gl_frame
        self.target_list = target_list
        self.input_transform = default_input_transform if input_transform is None else input_transform
        self.target_transform = default_target_transform if target_transform is None else target_transform
        self.num_samples = len(view_list)
        if phase == 'train':
            self.num_samples *= num_samples
        self.inner_batch = inner_batch
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
        if not self.headless:
            self.scene = NNScene()
            setup_scene(self.scene, self.scene_data, use_mesh=self.use_mesh)

            if self.perturb_points and self.fastrand is None:
                print(f'SETTING PERTURB POINTS: {self.perturb_points}')
                tform = lambda p: self.perturb_points * (p - 0.5)
                self.fastrand = FastRand((self.scene_data['pointcloud']['xyz'].shape[0], 2), tform, 10) 

    def unload(self):
        if not self.headless:
            self.scene.delete()
            self.scene = None
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # assert -1 < idx < len(self)
        idx = idx % len(self.view_list)
        if self.timing is None:
            self.timing = AccumDict()
        
        tt = TicToc()
        tt.tic()

        mask = None
        mask_crop = None
        if self.mask_list[idx]:
            mask = (load_image(self.mask_list[idx])[...,0]/255).astype(np.float32)
            if self.crop_by_mask:
                cnt = get_rnd_crop_center_v1(mask[..., 0])
                mask_crop = -1 + 2 * np.array(cnt) / mask.shape[:2]

        label = None
        if self.label_list[idx]:
            label = load_image(self.label_list[idx])[...,0]

        view_matrix = self.view_list[idx].astype(np.float32)
        target = load_image(self.target_list[idx])

        
        if self.phase=='train':
            # innner_batch 
            Hs = self.get_transform_crop(self.inner_batch,8)
            Ks = [H @self.K_src for H in Hs]
            targets = [self._warp(target, H) for H in Hs]
            if mask is None:
                masks = [np.ones((self.tgt_sh[1], self.tgt_sh[0]), dtype=np.float32)]*len(Hs)
            else:
                masks = [self._warp(mask, H) for H in Hs]
            if label is None:
                labels = [np.zeros((target.shape[0], target.shape[1], 1), dtype=np.uint8)]*len(Hs)
            else:
                labels = [self._warp(label, H) for H in Hs]
            proj_matrixs = [get_proj_matrix(K, self.tgt_sh, self.znear, self.zfar).astype(np.float32) for K in Ks]
            
        else:
            K = rescale_K(self.K_src, self.tgt_sh/self.src_sh, False)
            H = K @ np.linalg.inv(self.K_src)
            target = self._warp(target, H)
            if mask is None:
                mask = np.ones((self.tgt_sh[1], self.tgt_sh[0]), dtype=np.float32)
            else:
                mask = self._warp(mask, H)
            if label is None:
                label = np.zeros((target.shape[0], target.shape[1], 1), dtype=np.uint8)
            else:
                label = self._warp(label, H) 
            proj_matrix = get_proj_matrix(K, self.tgt_sh, self.znear, self.zfar).astype(np.float32)
            # H = self.randomImageCrop()
            # K = H @self.K_src
            # target = self._warp(target, H)
            # proj_matrix = get_proj_matrix(K, self.tgt_sh, self.znear, self.zfar).astype(np.float32)

        # if mask is None:
        #     mask = np.ones((target.shape[0], target.shape[1], 1), dtype=np.float32)
        # elif self.phase=='train':
        #     mask = self._warp(mask, H)

        self.timing.add('get_target', tt.toc())
        tt.tic()
        
        # we want to make sure GL and CUDA Interop contexts are created in
        # the process calling __getitem__ method, otherwise rendering would not work
        if not self.headless:
            from glumpy import app, gl
            if self.renderer is None:
            # headless
                assert self.scene is not None, 'call load()'
                app.Window(visible=False) # creates GL context
                self.renderer = MultiscaleRender(self.scene, self.input_format, self.tgt_sh, supersampling=self.ss)
                
            if self.drop_points:
                self.scene.set_point_discard(np.random.rand(self.scene_data['pointcloud']['xyz'].shape[0]) < self.drop_points)
            if self.perturb_points:
                self.scene.set_point_perturb(self.fastrand.toss())
                    
            input_ = self.renderer.render(view_matrix=view_matrix, proj_matrix=proj_matrix)
            self.timing.add('render', tt.toc())
            tt.tic()
            input_ = {k: self.input_transform(v) for k, v in input_.items()} # np.array to tensor
            

                        
        else:
            input_ = {}
              


        self.count += 1
        
        if self.phase=='train':
            input_['id'] = np.array([self.id]* len(Hs))
            targets = torch.stack([self.target_transform(t) for t in targets],0)
            return {'input': input_,
                    'view_matrix': np.array([view_matrix]*len(Hs)),
                    'intrinsic_matrix': np.array(Ks),
                    'proj_matrix': np.array(proj_matrixs),
                    'target': targets,
                    'target_filename':[self.target_list[idx]]*len(Hs),
                    'mask': torch.stack([torch.from_numpy(t) for t in masks],0),
                    'label': torch.stack([torch.from_numpy(t) for t in labels],0)
                }
        else:
            input_['id'] = self.id                            
            target = self.target_transform(target)
            return {'input': input_,
                'view_matrix': view_matrix,
                'intrinsic_matrix': K,
                'proj_matrix': proj_matrix,
                'target': target,
                'target_filename': self.target_list[idx],
                'mask': torch.from_numpy(mask),
                'label': torch.from_numpy(label)
               }
            
        
 
    def get_transform_crop(self, inner_batch=8, inner_sample=8):
        centers, Hs = [], []
        c = self.tgt_sh * 0.5
        best_dis = -1
        for i in range(inner_batch):
            for j in range(inner_sample):
                H = self.randomImageCrop()
                H_inv = np.linalg.inv(H)
                c_trans = np.array([H_inv[0,0]*c[0]+H_inv[0,2], H_inv[1,1]*c[1]+H_inv[1,2]])
                # center
                if len(centers)==0: 
                    dis=0
                else:
                    dis = []
                    for c2 in centers:
                        dis.append(np.linalg.norm(c_trans-c2))
                    dis = min(dis)
                if(j==0 or dis>best_dis):
                    best = H
                    best_c = c_trans
                    best_dis = dis
            centers.append(best_c)
            Hs.append(best)
        return Hs                 
    
    def randomImageCrop(self):
        H=np.eye(3)
        min_zoom_xy = self.tgt_sh/self.src_sh
        z = max(min_zoom_xy[0], min_zoom_xy[1])
        if self.random_zoom:
            min_zoom = max(self.random_zoom[0], z)
            max_zoom = self.random_zoom[1]
            z = rand_(min_zoom, max_zoom)
        if self.random_shift:
            max_shift = self.src_sh*z-self.tgt_sh
            random_shift_x =  rand_(0,max_shift[0])
            random_shift_y =  rand_(0,max_shift[1])
            H[0,2], H[1,2] = -random_shift_x, -random_shift_y
        H[0,0], H[1,1] = z,z
        return H
   
    
    def _warp(self, image, H):
        image = cv2.warpPerspective(image, H, tuple(self.tgt_sh))
        
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


    ds_train_list, ds_val_list = [], []
    texture_ckpts = []
    

    for name in args.dataset_names:
        print(f'creating dataset {name}')
        
        ds_train, ds_val = _get_splits(paths_data, name, args)

        ds_train.name = ds_val.name = name
        ds_train.id = ds_val.id = args.dataset_names.index(name)

        ds_train_list.append(ds_train)
        ds_val_list.append(ds_val)
        if 'texture_ckpt' in ds_train.scene_data['config'].keys():
            texture_ckpts.append(ds_train.scene_data['config']['texture_ckpt'])
        else:
            texture_ckpts.append(None)

        print(f'ds_train: {len(ds_train)}')
        print(f'ds_val: {len(ds_val)}')


    return ds_train_list, ds_val_list, texture_ckpts


def _load_dataset(name, paths_data, args):
    ds_train, ds_val = _get_splits(paths_data, name, args)

    ds_train.name = ds_val.name = name
    ds_train.id = ds_val.id = args.dataset_names.index(name)

    return ds_train, ds_val


def _get_splits(paths_file, ds_name, args):
    print("paths_file",paths_file)
    print("ds_name",ds_name)
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
    if args.eval_all:
        from READ.datasets.splitter import eval_all
        splits = eval_all([view_list, target_list, mask_list, label_list], **args.splitter_args)
        
    view_list_train, view_list_val = splits[0]
    target_list_train, target_list_val = splits[1]
    mask_list_train, mask_list_val = splits[2]
    label_list_train, label_list_val = splits[3]

    # num_samples_train = int(config['num_samples_train']) if 'num_samples_train' in config else None

    ds_train = DynamicDataset('train', scene_data, input_format, args.crop_size, view_list_train, target_list_train, mask_list_train, label_list_train,
        use_mesh=args.use_mesh, supersampling=args.supersampling, headless=args.headless, **args.train_dataset_args)

    ds_val = DynamicDataset('val', scene_data, input_format, args.crop_size, view_list_val, target_list_val, mask_list_val, label_list_val,
        use_mesh=args.use_mesh, supersampling=args.supersampling, headless=args.headless, **args.val_dataset_args)

    return ds_train, ds_val


