import os, sys
import munch

import torch

from READ.utils.train import get_module, save_model, load_model_checkpoint
from READ.utils.arguments import deval_args


class Pipeline:
    def export_args(self, parser):
        # add arguments of this pipeline to the cmd parser
        raise NotImplementedError()

    def create(self, args):
        # return dictionary with pipeline components
        raise NotImplementedError()

    def dataset_load(self, *args, **kwargs):
        # called before running train/val
        pass

    def dataset_unload(self, *args, **kwargs):
        # called after running train/val
        pass

    def get_net(self):
        raise NotImplementedError()

    def extra_optimizer(self, *args):
        return None


def load_pipeline(checkpoint, args_to_update=None):
    ckpt = torch.load(checkpoint, map_location='cpu')

    assert 'args' in ckpt

    if args_to_update:
        ckpt['args'].update(args_to_update)
    ckpt['args']['pipeline']='READ.pipelines.ogl.TexturePipeline'
    ckpt['args']['splitter_module']= 'READ.datasets.splitter.split_by_step'
    ckpt['args']['criterion_module']= 'READ.criterions.vgg_loss.VGGLoss'

    try:
        args = munch.munchify(ckpt['args'])

        pipeline = get_module(args.pipeline)()
        pipeline.create(args)
    except AttributeError as err:
        print('\nERROR: Checkpoint args is incompatible with this version\n', file=sys.stderr)
        raise err

    load_model_checkpoint(checkpoint, pipeline.get_net())

    return pipeline, args
    

def save_pipeline(pipeline, save_dir, epoch, stage, args):
    objects = pipeline.state_objects()

    args = deval_args(args)

    for name, obj in objects.items():
        obj_class = obj.__class__.__name__
        filename = f'{obj_class}_stage_{stage}_epoch_{epoch}'
        if name:
            name = name.replace('/', '_')
            filename = f'{filename}_{name}'
        save_path = os.path.join(save_dir, filename + '.pth')
        save_model(save_path, obj, args=args)
