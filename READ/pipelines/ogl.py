import os, sys
from pathlib import Path

from torch import autograd, optim

from READ.pipelines import Pipeline
from READ.datasets.dynamic import get_datasets
from READ.models.texture import PointTexture, MeshTexture
from READ.models.unet import UNet
from READ.models.compose import NetAndTexture, MultiscaleNet, RGBTexture
from READ.criterions.vgg_loss import VGGLoss
from READ.utils.train import to_device, set_requires_grad, save_model, unwrap_model, image_grid, to_numpy, load_model_checkpoint, freeze
from READ.utils.perform import TicToc, AccumDict, Tee


TextureOptimizerClass = optim.RMSprop


def get_net(input_channels, args):
    net = UNet(
        num_input_channels=8, 
        num_output_channels=3,
        feature_scale=4,
        num_res=4
        )

    return net


def get_texture(num_channels, size, args):
    if not hasattr(args, 'reg_weight'):
        args.reg_weight = 0.

    if args.use_mesh:
        texture = MeshTexture(num_channels, size, activation=args.texture_activation, reg_weight=args.reg_weight)
    else:
        texture = PointTexture(num_channels, size, activation=args.texture_activation, reg_weight=args.reg_weight)

    if args.texture_ckpt:
        texture = load_model_checkpoint(args.texture_ckpt, texture)

    return texture


def backward_compat(args):
    if not hasattr(args, 'input_channels'):
        args.input_channels = None
    if not hasattr(args, 'conv_block'):
        args.conv_block = 'gated'

    if args.pipeline == 'READ.pipelines.ogl.Pix2PixPipeline':
        if not hasattr(args, 'input_modality'):
            args.input_modality = 1

    return args


class TexturePipeline(Pipeline):
    def export_args(self, parser):
        parser.add_argument('--descriptor_size', type=int, default=8)
        parser.add_argument('--texture_size', type=int)
        parser.add_argument('--texture_ckpt', type=Path)
        parser.add('--texture_lr', type=float, default=1e-1)
        parser.add('--texture_activation', type=str, default='none')
        parser.add('--n_points', type=int, default=0, help='this is for inference')

    def create(self, args):
        args = backward_compat(args)

        if not args.input_channels:
            args.input_channels = [args.descriptor_size] * args.num_mipmap

        net = get_net(args.input_channels, args)

        textures = {}

        if args.inference:
            if args.use_mesh:
                size = args.texture_size
            else:
                size = args.n_points
            textures = {
                0: get_texture(args.descriptor_size, size, args)
                }
        else:
            self.ds_train, self.ds_val = get_datasets(args)

            for ds in self.ds_train:
                if args.use_mesh:
                    assert args.texture_size, 'set texture size'
                    size = args.texture_size
                else:
                    assert ds.scene_data['pointcloud'] is not None, 'set pointcloud'
                    size = ds.scene_data['pointcloud']['xyz'].shape[0]
                textures[ds.id] = get_texture(args.descriptor_size, size, args)

            self.optimizer = optim.Adam(net.parameters(), lr=args.lr)

            if len(textures) == 1:
                self._extra_optimizer = TextureOptimizerClass(textures[0].parameters(), lr=args.texture_lr)
            else:
                self._extra_optimizer = None

            self.criterion = args.criterion_module(**args.criterion_args).cuda()

        ss = args.supersampling if hasattr(args, 'supersampling') else 1

        self.net = net
        self.textures = textures
        self.model = NetAndTexture(net, textures, ss)

        self.args = args

    def state_objects(self):
        datasets = self.ds_train

        objs = {'net': self.net}
        objs.update({ds.name: self.textures[ds.id] for ds in datasets})

        return objs

    def dataset_load(self, dataset):
        self.model.load_textures([ds.id for ds in dataset])
        
        for ds in dataset:
            ds.load()


    def extra_optimizer(self, dataset):
        # if we have single dataset, don't recreate optimizer
        if self._extra_optimizer is not None:
            lr_drop = self.optimizer.param_groups[0]['lr'] / self.args.lr
            self._extra_optimizer.param_groups[0]['lr'] = self.args.texture_lr * lr_drop
            return self._extra_optimizer

        param_group = []
        for ds in dataset:
            param_group.append(
                {'params': self.textures[ds.id].parameters()}
            )

        lr_drop = self.optimizer.param_groups[0]['lr'] / self.args.lr

        return TextureOptimizerClass(param_group, lr=self.args.texture_lr * lr_drop)

    def dataset_unload(self, dataset):
        self.model.unload_textures()

        for ds in dataset:
            ds.unload()
            self.textures[ds.id].null_grad()

    def get_net(self):
        return self.net


class Pix2PixPipeline(Pipeline):
    def export_args(self, parser):
        parser.add('--input_modality', type=int, default=1)

    def create(self, args):
        args = backward_compat(args)

        if not args.input_channels:
            print('Assume input channels is 3')
            args.input_channels = [3] * args.num_mipmap

        net = get_net(args.input_channels, args)

        self.model = MultiscaleNet(net, args.input_modality)
        self.net = net

        if not args.inference:
            self.ds_train, self.ds_val = get_datasets(args)

            
            self.optimizer= optim.Adam(self.model.parameters(), lr=args.lr)
                

            self.criterion = args.criterion_module(**args.criterion_args).cuda()

    def state_objects(self):
        return {'net': self.net}

    def dataset_load(self, dataset):
        for ds in dataset:
            ds.load()


    def dataset_unload(self, dataset):
        for ds in dataset:
            ds.unload()
            

    def get_net(self):
        return self.net


class RGBTexturePipeline(Pipeline):
    def export_args(self, parser):
        parser.add('--texture_size', type=int, default=2048)
        parser.add('--texture_lr', type=float, default=1e-2)

    def create(self, args):
        self.texture = MeshTexture(3, args.texture_size, activation='none', levels=1, reg_weight=0)
        self.model = RGBTexture(self.texture)

        if not args.inference:
            self.ds_train, self.ds_val = get_datasets(args)

            self.optimizer = TextureOptimizerClass(self.texture.parameters(), lr=args.texture_lr)

            self.criterion = args.criterion_module(**args.criterion_args).cuda()

    def dataset_load(self, dataset):
        for ds in dataset:
            ds.load()

    def dataset_unload(self, dataset):
        for ds in dataset:
            ds.unload()

    def state_objects(self):
        return {'model': self.model}

    def get_net(self):
        return self.model
