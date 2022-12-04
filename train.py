import numpy as np
import random
import cv2
import yaml
from time import time, sleep
from collections import defaultdict
from pprint import pprint
from pathlib import Path
import math
import os, sys
import datetime
import argparse

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data
from torchvision import transforms

from tensorboardX import SummaryWriter
import torch.nn.functional as F

from READ.utils.perform import TicToc, AccumDict, Tee
from READ.utils.arguments import MyArgumentParser, eval_args
from READ.models.compose import ModelAndLoss
from READ.utils.train import to_device, image_grid, to_numpy, get_module, freeze, load_model_checkpoint, unwrap_model
from READ.pipelines import save_pipeline

def psnr2(target, ref):
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
 
    if target.shape != ref.shape:
        raise ValueError('imgae size')
 
    diff = ref - target
    diff = diff.flatten('C')
 
    rmse = math.sqrt( np.mean(diff ** 2.) )
    psnr = 20 * math.log10(np.max(target) / rmse)
 
    return psnr

def setup_environment(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THREADS'] = '1'


def setup_logging(save_dir):
    tee = Tee(os.path.join(save_dir, 'log.txt'))
    sys.stdout, sys.stderr = tee, tee



def get_experiment_name(args, default_args, args_to_ignore, delimiter='__'):
    s = []

    args = vars(args)
    default_args = vars(default_args)

    def shorten_paths(args):
        args = dict(args)
        for arg, val in args.items():
            if isinstance(val, Path):
                args[arg] = val.name
        return args

    args = shorten_paths(args)
    default_args = shorten_paths(default_args)

    for arg in sorted(args.keys()):
        if arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}^{args[arg]}"]
    
    out = delimiter.join(s)
    out = out.replace('/', '+')
    out = out.replace("'", '')
    out = out.replace("[", '')
    out = out.replace("]", '')
    out = out.replace(" ", '')
    return out


def make_experiment_dir(base_dir, postfix='', use_time=True):
    time = datetime.datetime.now()

    if use_time:
        postfix = time.strftime(f"%m-%d_%H-%M-%S___{postfix}")

    save_dir = os.path.join(base_dir, postfix)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)

    return save_dir


def num_param(model):
    return sum([p.numel() for p in unwrap_model(model).parameters()])

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def run_epoch(pipeline, phase, epoch, args, iter_cb=None):
    ad = AccumDict()
    tt = TicToc()
    
    device = 'cuda:0'

    model = pipeline.model
    criterion = pipeline.criterion
    optimizer = pipeline.optimizer

    print(f'model parameters: {num_param(model)}')

    if args.merge_loss:
        model = ModelAndLoss(model, criterion, use_mask=args.use_mask)

    if args.multigpu:
        model = nn.DataParallel(model)

    def run_sub(dl, extra_optimizer):
        model.cuda()

        tt.tic()
        for it, data in enumerate(dl):
            input = to_device(data['input'], device)
            target = to_device(data['target'], device)
  
            if 'mask' in data and args.use_mask:
                mask = to_device(data['mask'], device)

                if mask.sum() < 1:
                    print(f'skip batch, mask is {mask.sum()}')
                    continue


                target *= mask
            else:
                mask = None

            #ad.add('data_time', tt.toc())

            tt.tic()
            if args.merge_loss:
                out, loss = model(input, target, mask=mask)
            else:
                out = model(input)

                if mask is not None and args.use_mask:
                    loss = criterion(out * mask, target)
                else:
                    loss = criterion(out, target)

            psnr_value = psnr2(out.cpu().detach().numpy(),target.cpu().detach().numpy())


            ad.add('psnr', psnr_value)


            if loss.numel() > 1:
                loss = loss.mean()

            if mask is not None:
                loss /= mask.mean() + 1e-6

                # TODO: parameterize
                bkg_color = torch.FloatTensor([1, 1, 1]).reshape(1, 3, 1, 1).to(loss.device)
                bkg_weight = 500

                n_mask = 1 - mask
                out_bkg = out * n_mask
                bkg = bkg_color * n_mask
                loss_bkg = bkg_weight * torch.abs((out_bkg - bkg)).mean() / (n_mask.mean() + 1e-6)

                loss += loss_bkg

                ad.add('loss_bkg', loss_bkg.item())

            if hasattr(pipeline.model, 'reg_loss'):
                reg_loss = pipeline.model.reg_loss()
                loss += reg_loss

                if torch.is_tensor(reg_loss):
                    reg_loss = reg_loss.item()
                #ad.add('reg_loss', reg_loss)

            ad.add('batch_time', tt.toc())
            if phase == 'train':
                tt.tic()
                loss.backward(create_graph=False)
                
                optimizer.step()
                optimizer.zero_grad()

                if extra_optimizer is not None:
                    extra_optimizer.step()
                    extra_optimizer.zero_grad()
                #ad.add('step_time', tt.toc())

            ad.add('loss', loss.item())

            if iter_cb:
                tt.tic()
                iter_cb.on_iter(it + it_before, max_it, input, out, target, data, ad, phase, epoch)
                # ad.add('iter_cb_time', tt.toc())
            
            tt.tic() # data_time

    ds_list = pipeline.__dict__[f'ds_{phase}']
    sub_size = args.max_ds

    if phase == 'train':
        random.shuffle(ds_list)

    it_before = 0
    max_it = np.sum([len(ds) for ds in ds_list]) // args.batch_size

    for i_sub in range(0, len(ds_list), sub_size):
        ds_sub = ds_list[i_sub:i_sub + sub_size]
        ds_ids = [d.id for d in ds_sub]
        print(f'running on datasets {ds_ids}')

        ds = ConcatDataset(ds_sub)
        if phase == 'train':
            dl = DataLoader(ds, args.batch_size, num_workers=args.dataloader_workers, drop_last=True, pin_memory=False, shuffle=True, worker_init_fn=ds_init_fn)
        else:
            batch_size_val = args.batch_size if args.batch_size_val is None else args.batch_size_val
            dl = DataLoader(ds, batch_size_val, num_workers=args.dataloader_workers, drop_last=True, pin_memory=False, shuffle=False, worker_init_fn=ds_init_fn)
        
        pipeline.dataset_load(ds_sub)
        print(f'total parameters: {num_param(model)}')

        extra_optimizer = pipeline.extra_optimizer(ds_sub)

        run_sub(dl, extra_optimizer)

        pipeline.dataset_unload(ds_sub)

        it_before += len(dl)

        torch.cuda.empty_cache()

    avg_loss = np.mean(ad['loss'])
    iter_cb.on_epoch(phase, avg_loss, epoch)

    return avg_loss


def run_train(epoch, pipeline, args, iter_cb):

    if args.eval_in_train or (args.eval_in_train_epoch >= 0 and epoch >= args.eval_in_train_epoch):
        print('EVAL MODE IN TRAIN')
        pipeline.model.eval()
        if hasattr(pipeline.model, 'ray_block') and pipeline.model.ray_block is not None:
            pipeline.model.ray_block.train()
    else:
        pipeline.model.train()

    with torch.set_grad_enabled(True):
        return run_epoch(pipeline, 'train', epoch, args, iter_cb=iter_cb)


def run_eval(epoch, pipeline, args, iter_cb):
    torch.cuda.empty_cache()

    if args.eval_in_test:
        pipeline.model.eval()
    else:
        print('TRAIN MODE IN EVAL')
        pipeline.model.train()

    with torch.set_grad_enabled(False):
        return run_epoch(pipeline, 'val', epoch, args, iter_cb=iter_cb)
    

class TrainIterCb:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.train_it = 0

    def on_iter(self, it, max_it, input, out, target, data_dict, ad, phase, epoch):    
        if it % self.args.log_freq == 0:
            s = f'{phase.capitalize()}: [{epoch}][{it}/{max_it-1}]\t'
            s += str(ad)
            print(s)

        if phase == 'train':
            self.writer.add_scalar(f'{phase}/loss', ad['loss'][-1], self.train_it)

            if 'reg_loss' in ad.__dict__():
                self.writer.add_scalar(f'{phase}/reg_loss', ad['reg_loss'][-1], self.train_it)

            self.train_it += 1

        if it % self.args.log_freq_images == 0:
            if isinstance(out, dict):
                inputs = out['input']
                scale = np.random.choice(len(inputs))
                keys = list(inputs.keys())
                out_img = inputs[keys[scale]]
                out = F.interpolate(out_img, size=target.shape[2:])
            
            out = out.clamp(0, 1)
            self.writer.add_image(f'{phase}', image_grid(out, target), self.train_it)

    def on_epoch(self, phase, loss, epoch):
        if phase != 'train':
            self.writer.add_scalar(f'{phase}/loss', loss, epoch)


class EvalIterCb:
    def __init__(self):
        pass

    def on_iter(self, it, max_it, input, out, target, data_dict, ad, phase, epoch):
        for fn in data_dict['target_filename']:
            name = fn.split('/')[-1]
            out_fn = os.path.join('data/eval', name)
            print(out_fn)
            cv2.imwrite(out_fn, to_numpy(out)[...,::-1])
            cv2.imwrite(out_fn+'.target.jpg', to_numpy(target)[...,::-1])

    def on_epoch(self, phase, loss, epoch):
        pass


def save_splits(exper_dir, ds_train, ds_val):
    def write_list(path, data):
        with open(path, 'w') as f:
            for l in data:
                f.write(str(l))
                f.write('\n')

    for ds in ds_train.datasets:
        np.savetxt(os.path.join(exper_dir, 'train_view.txt'), np.vstack(ds.view_list))
        write_list(os.path.join(exper_dir, 'train_target.txt'), ds.target_list)

    for ds in ds_val.datasets:
        np.savetxt(os.path.join(exper_dir, 'val_view.txt'), np.vstack(ds.view_list))
        write_list(os.path.join(exper_dir, 'val_target.txt'), ds.target_list)



def ds_init_fn(worker_id):
    np.random.seed(int(time()))



def parse_image_size(string):
    error_msg = 'size must have format WxH'
    tokens = string.split('x')
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError(error_msg)
    try:
        w = int(tokens[0])
        h = int(tokens[1])
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(error_msg)


def parse_args(parser):
    args, _ = parser.parse_known_args()
    assert args.pipeline, 'set pipeline module'
    pipeline = get_module(args.pipeline)()
    pipeline.export_args(parser)

    # override defaults
    if args.config:
        with open(args.config) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)

        parser.set_defaults(**config)

    return parser.parse_args(), parser.parse_args([])


def print_args(args, default_args):
    from huepy import bold, lightblue, orange, lightred, green, red

    args_v = vars(args)
    default_args_v = vars(default_args)
    
    print(bold(lightblue(' - ARGV: ')), '\n', ' '.join(sys.argv), '\n')
    # Get list of default params and changed ones    
    s_default = ''     
    s_changed = ''
    for arg in sorted(args_v.keys()):
        value = args_v[arg]
        if default_args_v[arg] == value:
            s_default += f"{lightblue(arg):>50}  :  {orange(value if value != '' else '<empty>')}\n"
        else:
            s_changed += f"{lightred(arg):>50}  :  {green(value)} (default {orange(default_args_v[arg] if default_args_v[arg] != '' else '<empty>')})\n"

    print(f'{bold(lightblue("Unchanged args")):>69}\n\n'
          f'{s_default[:-1]}\n\n'
          f'{bold(red("Changed args")):>68}\n\n'
          f'{s_changed[:-1]}\n')


def check_pipeline_attributes(pipeline, attributes):
    for attr in attributes:
        if not hasattr(pipeline, attr):
            raise AttributeError(f'pipeline missing attribute "{attr}"')


def try_save_dataset(save_dir, dataset, prefix):
    if hasattr(dataset[0], 'target_list'):
        with open(os.path.join(save_dir, f'{prefix}.txt'), 'w') as f:
            for ds in dataset:
                f.writelines('\n'.join(ds.target_list))
                f.write('\n')


def save_args(exper_dir, args, prefix):
    with open(os.path.join(exper_dir, f'{prefix}.yaml'), 'w') as f:
        yaml.dump(vars(args), f)


if __name__ == '__main__':
    parser = MyArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument
    parser.add('--eval', action='store_bool', default=False)
    parser.add('--crop_size', type=parse_image_size, default='512x512')
    parser.add('--batch_size', type=int, default=8)
    parser.add('--batch_size_val', type=int, default=None, help='if not set, use batch_size')
    parser.add('--lr', type=float, default=1e-4)
    parser.add('--freeze_net', action='store_bool', default=False)
    parser.add('--eval_in_train', action='store_bool', default=False)
    parser.add('--eval_in_train_epoch', default=-1, type=int)
    parser.add('--eval_in_test',  action='store_bool', default=True)
    parser.add('--merge_loss', action='store_bool', default=True)
    parser.add('--net_ckpt', type=Path, default=None, help='neural network checkpoint')
    parser.add('--save_dir', type=Path, default='data/experiments')
    parser.add('--epochs', type=int, default=100)
    parser.add('--seed', type=int, default=2019)
    parser.add('--save_freq', type=int, default=1, help='save checkpoint each save_freq epoch')
    parser.add('--log_freq', type=int, default=5, help='print log each log_freq iter')
    parser.add('--log_freq_images', type=int, default=100)
    parser.add('--comment', type=str, default='', help='comment to experiment')
    parser.add('--paths_file', type=str)
    parser.add('--dataset_names', type=str, nargs='+')
    parser.add('--exclude_datasets', type=str, nargs='+')
    parser.add('--config', type=Path)
    parser.add('--use_mask', action='store_bool')
    parser.add('--pipeline', type=str, help='path to pipeline module')
    parser.add('--inference', action='store_bool', default=False)
    parser.add('--ignore_changed_args', type=str, nargs='+', default=['ignore_changed_args', 'save_dir', 'dataloader_workers', 'epochs', 'max_ds', 'batch_size_val'])
    parser.add('--multigpu', action='store_bool', default=True)
    parser.add('--dataloader_workers', type=int, default=3)
    parser.add('--max_ds', type=int, default=3, help='maximum datasets in DataLoader at the same time')
    parser.add('--reg_weight', type=float, default=0.)
    parser.add('--input_format', type=str)
    parser.add('--num_mipmap', type=int, default=5)
    parser.add('--net_size', type=int, default=4)
    parser.add('--input_channels', type=int, nargs='+')
    parser.add('--conv_block', type=str, default='gated')
    parser.add('--supersampling', type=int, default=1)
    parser.add('--use_mesh', action='store_bool', default=False)
    parser.add('--simple_name', action='store_bool', default=False)

    args, default_args = parse_args(parser)

    setup_environment(args.seed)

    if args.eval:
        iter_cb = EvalIterCb()
    else:
        if args.simple_name:
            args.ignore_changed_args += ['config', 'pipeline']
        exper_name = get_experiment_name(args, default_args, args.ignore_changed_args)
        exper_dir = make_experiment_dir(args.save_dir, postfix=exper_name)

        writer = SummaryWriter(log_dir=exper_dir, flush_secs=10)
        iter_cb = TrainIterCb(args, writer)

        setup_logging(exper_dir)

        print(f'experiment dir: {exper_dir}')

    print_args(args, default_args)

    args = eval_args(args)

    pipeline = get_module(args.pipeline)()
    pipeline.create(args)

    required_attributes = ['model', 'ds_train', 'ds_val', 'optimizer', 'criterion']
    check_pipeline_attributes(pipeline, required_attributes)

    # lr_scheduler = [torch.optim.lr_scheduler.ReduceLROnPlateau(o, patience=3, factor=0.5, verbose=True) for o in pipeline.optimizer]
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer, patience=3, factor=0.5, verbose=True)

    if args.net_ckpt:
        print(f'LOAD NET CHECKPOINT {args.net_ckpt}')
        load_model_checkpoint(args.net_ckpt, pipeline.get_net())
    
    if hasattr(pipeline.model, 'ray_block') and pipeline.model.ray_block is not None:
        if hasattr(args, 'ray_block_ckpt') and args.ray_block_ckpt:
            print(f'LOAD RAY BLOCK CHECKPOINT {args.ray_block_ckpt}')
            load_model_checkpoint(args.ray_block_ckpt, pipeline.model.ray_block)
    # torch.backends.cudnn.enabled = False 

    if args.freeze_net:
        print('FREEZE NET')
        freeze(pipeline.get_net(), True)

    if args.eval:
        loss = run_eval(0, pipeline, args, iter_cb)
        print('VAL LOSS', loss)
    else:
        try_save_dataset(exper_dir, pipeline.ds_train, 'train')
        try_save_dataset(exper_dir, pipeline.ds_val, 'val')

        save_args(exper_dir, args, 'args')
        save_args(exper_dir, default_args, 'default_args')

        for epoch in range(args.epochs):
            print('### EPOCH', epoch)

            print('> TRAIN')

            train_loss = run_train(epoch, pipeline, args, iter_cb)
            
            print('TRAIN LOSS', train_loss)

            print('> EVAL')

            val_loss = run_eval(epoch, pipeline, args, iter_cb)

            # for sched in lr_scheduler:
            #     sched.step(val_loss)
            if val_loss is not None:
                lr_scheduler.step(val_loss)

            print('VAL LOSS', val_loss)

            if (epoch + 1) % args.save_freq == 0:
                save_pipeline(pipeline, os.path.join(exper_dir, 'checkpoints'), epoch, 0, args)
