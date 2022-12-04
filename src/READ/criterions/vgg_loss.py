import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
from collections import OrderedDict
from os.path import expanduser, join
import os

from READ.models.conv import PartialConv2d


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1) 

        
class VGGLoss(nn.Module):
    def __init__(self, net='caffe', partialconv=False, optimized=False, save_dir='.cache/torch/models'):
        super().__init__()
        
        self.partialconv = partialconv
        
        if net == 'pytorch':
            vgg19 = torchvision.models.vgg19(pretrained=True).features

            self.register_buffer('mean_', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer('std_', torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

        elif net == 'caffe':
            if not os.path.exists(join(save_dir, 'vgg_caffe_features.pth')):
                vgg_weights = torch.utils.model_zoo.load_url('https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth', model_dir=save_dir) 
                
                map = {'classifier.6.weight':u'classifier.7.weight', 'classifier.6.bias':u'classifier.7.bias'}
                vgg_weights = OrderedDict([(map[k] if k in map else k,v) for k,v in vgg_weights.items()])
                
                model = torchvision.models.vgg19()
                model.classifier = nn.Sequential(View(), *model.classifier._modules.values())

                model.load_state_dict(vgg_weights)
                
                vgg19 = model.features
                os.makedirs(save_dir, exist_ok=True)
                torch.save(vgg19, join(save_dir, 'vgg_caffe_features.pth'))

                self.register_buffer('mean_', torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.)
                self.register_buffer('std_', torch.FloatTensor([1./255, 1./255, 1./255])[None, :, None, None])

            else:
                vgg19 = torch.load(join(save_dir, 'vgg_caffe_features.pth'))
                self.register_buffer('mean_', torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.)
                self.register_buffer('std_', torch.FloatTensor([1./255, 1./255, 1./255])[None, :, None, None])
        else:
            assert False
                                   
        if self.partialconv:
            part_conv = PartialConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            part_conv.weight = vgg19[0].weight
            part_conv.bias = vgg19[0].bias
            vgg19[0] = part_conv

        vgg19_avg_pooling = []

        
        for weights in vgg19.parameters():
            weights.requires_grad = False

        for module in vgg19.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg19_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg19_avg_pooling.append(module)
        
        if optimized:
            self.layers = [3, 8, 17, 26, 35]            
        else:
            self.layers = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29]

        self.vgg19 = nn.Sequential(*vgg19_avg_pooling)

        # print(self.vgg19)

    def normalize_inputs(self, x):
        return (x - self.mean_) / self.std_

    def forward(self, input, target):
        loss = 0

        if self.partialconv:
            eps = 1e-9
            mask = target.sum(1, True) > eps
            mask = mask.float()

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)
        for i, layer in enumerate(self.vgg19):
            if isinstance(layer, PartialConv2d):
                features_input  = layer(features_input, mask)
                features_target = layer(features_target, mask)
            else:
                features_input  = layer(features_input)
                features_target = layer(features_target)

            if i in self.layers:
                loss = loss + F.l1_loss(features_input, features_target)

        return loss


class VGGLossMix(nn.Module):
    def __init__(self, weight=0.5):
        super(VGGLossMix, self).__init__()
        self.l1 = VGGLoss()
        self.l2 = VGGLoss(net='caffe')
        self.weight = weight

    def forward(self, input, target):
        return self.l1(input, target)*self.weight + self.l2(input, target) * (1-self.weight)
