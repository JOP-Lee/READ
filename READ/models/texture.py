import torch
import torch.nn as nn



class Texture(nn.Module):
    def null_grad(self):
        raise NotImplementedError()

    def reg_loss(self):
        return 0.


class PointTexture(Texture):
    def __init__(self, num_channels, size, activation='none', checkpoint=None, init_method='zeros', reg_weight=0.):
        super().__init__()

        assert isinstance(size, int), 'size must be int'

        shape = 1, num_channels, size

        if checkpoint:
            self.texture_ = torch.load(checkpoint, map_location='cpu')['texture'].texture_
        else:
            if init_method == 'rand':
                texture = torch.rand(shape)
            elif init_method == 'zeros':
                texture = torch.zeros(shape)
            else:
                raise ValueError(init_method)
            self.texture_ = nn.Parameter(texture.float())

        self.activation = activation
        self.reg_weight = reg_weight

    def null_grad(self):
        self.texture_.grad = None

    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.texture_, 2))

    def forward(self, inputs):
        if isinstance(inputs, dict):
            ids = None
            for f, x in inputs.items():
                if 'uv' in f:
                    ids = x[:, 0].long()
            assert ids is not None, 'Input format does not have uv'
        else:
            ids = inputs[:, 0] # BxHxW

        sh = ids.shape
        n_pts = self.texture_.shape[-1]

        ind = ids.contiguous().view(-1).long()

        texture = self.texture_.permute(1, 0, 2) # Cx1xN
        texture = texture.expand(texture.shape[0], sh[0], texture.shape[2]) # CxBxN
        texture = texture.contiguous().view(texture.shape[0], -1) # CxB*N

        sample = torch.index_select(texture, 1, ind) # CxB*H*W
        sample = sample.contiguous().view(sample.shape[0], sh[0], sh[1], sh[2]) # CxBxHxW
        sample = sample.permute(1, 0, 2, 3) # BxCxHxW

        if self.activation == 'sigmoid':
            return torch.sigmoid(sample)
        elif self.activation == 'tanh':
            return torch.tanh(sample)
        #print('sample,',sample.shape)
        return sample


class MeshTexture(Texture):
    def __init__(self, num_channels, size, activation='none', init_method='zeros', levels=4, reg_weight=0.):
        super().__init__()

        assert isinstance(size, int), f'size must be int not {size}'

        if init_method == 'rand':
            init = lambda shape: torch.rand(shape)
        elif init_method == 'zeros':
            init = lambda shape: torch.zeros(shape)
        elif init_method == '0.5':
            init = lambda shape: torch.zeros(shape) + 0.5
        else:
            raise ValueError(init_method)

        assert levels > 0
        self.levels = levels

        for i in range(self.levels):
            shape = 1, num_channels, size // 2 ** i, size // 2 ** i
            tex = nn.Parameter(init(shape)).float()
            self.__setattr__(f'texture_{i}', tex)

        self.activation = activation
        self.reg_weight = reg_weight

    def null_grad(self):
        for i in range(self.levels):
            self.__getattr__(f'texture_{i}').grad = None

    def reg_loss(self):
        loss = 0.
        tex_weight = [8., 2., 1., 0.]
        for i in range(self.levels):
            tex = self.__getattr__(f'texture_{i}')
            loss += self.reg_weight * tex_weight[i] * torch.mean(torch.pow(tex, 2))

        return loss

    def forward(self, inputs):
        uv = (inputs[:, :2] * 2 - 1.0).transpose(1, 3).transpose(1, 2).contiguous()

        samples = []
        for i in range(self.levels):
            tex = self.__getattr__(f'texture_{i}')
            sample = nn.functional.grid_sample(tex.expand(uv.shape[0], -1, -1, -1), uv)
            samples.append(sample)

        out = samples[0]
        for i in range(1, self.levels):
            out += samples[i]

        if self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'tanh':
            return torch.tanh(out)

        return out
        