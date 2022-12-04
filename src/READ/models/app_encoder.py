import torch
import torch.nn as nn

# E_attr, implicit_mask
class E_attr(nn.Module):
  def __init__(self, input_dim_a=3, output_nc=32):
    super(E_attr, self).__init__()
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))  ## 1*1

  def forward(self, x):
    out = self.model(x)
    # out = x.view(x.size(0), -1)
    return out

if __name__ == '__main__':
    data = torch.randn((1,3,255,255))
    model = E_attr(3)
    out = model(data)
    import pdb; pdb.set_trace()