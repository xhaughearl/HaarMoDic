import torch
from torch import nn
from einops.layers.torch import Rearrange

# down_dwt
from pytorch_wavelets import DWT2D, IDWT2D
dwt = DWT2D(wave='haar', J=1).cuda()
idwt = IDWT2D(wave='haar').cuda()


def down_dwt(x):
    inp = x.clone()
    inp = inp[None,]
    inp = dwt(inp)
    inp = torch.concat(
        [
            inp[0][0],
            inp[1][0][0][:, 0],
            inp[1][0][0][:, 1],
            inp[1][0][0][:, 2]
        ], 1
    )
    return inp
# up_dwt


def up_dwt(x):
    out = x.clone()
    b, n, d = out.shape
    # print(n)
    sec = n // 4
    # print(sec)
    yl = out[:, 0:sec][None,]
    yh = [out[:, (i+1)*sec:(i+2)*sec].unsqueeze(1) for i in range(3)]
    yh = torch.concat(yh, 1)[None, :]
    yl.shape
    out = idwt((yl, [yh]))[0]
    return out


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x


class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPblock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

        self.dropout = nn.Dropout(0.005)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):

        x_ = self.fc0(x)
        x_ = self.norm0(x_)

        x = self.dropout(x)

        x = x + x_

        return x


class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x


class HR_MLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        sec = num_layers // 4
        self.mlp_initial = nn.Sequential(
            *[
                MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
                for _ in range(sec)
            ]
        )
        self.mlp_final = nn.Sequential(
            *[
                MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
                for _ in range(sec)
            ]
        )
        self.mlp_stem_0 = nn.Sequential(
            *[
                MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
                for _ in range(sec * 2)
            ]
        )
        self.mlp_stem_1_1 = nn.Sequential(
            *[
                MLPblock(dim * 2, seq // 2, use_norm,
                         use_spatial_fc, layernorm_axis)
                for _ in range(sec)
            ]
        )
        self.mlp_stem_1_2 = nn.Sequential(
            *[
                MLPblock(dim * 2, seq // 2, use_norm,
                         use_spatial_fc, layernorm_axis)
                for _ in range(sec)
            ]
        )
        self.mlp_stem_2 = nn.Sequential(
            *[
                MLPblock(dim*4, seq//4+1, use_norm,
                         use_spatial_fc, layernorm_axis)
                for _ in range(sec)
            ]
        )

    def forward(self, x):
        x = self.mlp_initial(x)
        t1 = self.mlp_stem_0(x)
        p1 = down_dwt(x)
        p2 = self.mlp_stem_1_1(p1)

        p3 = down_dwt(p2)

        p3 = self.mlp_stem_2(p3)

        p3 = up_dwt(p3)[:, :, :-1]
        p3 = up_dwt(p3)

        p4 = self.mlp_stem_1_2(p2)
        p4 = up_dwt(p4)

        x = p4 + t1 + p3
        x = self.mlp_final(x)
        return x


class initial_block(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        J0 = x
        J1 = down_dwt(J0)
        J2 = down_dwt(J1)
        return [J0, J1, J2]


class fuse_block(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        J0, J1, J2 = x

        J0_J1 = down_dwt(J0)
        J0_J2 = down_dwt(J0_J1)

        J1_J0 = up_dwt(J1)
        J1_J2 = down_dwt(J1)

        J2_J1 = up_dwt(J2)[:, :, :-1]
        J2_J0 = up_dwt(J2_J1)

        J0 = J0 + J1_J0 + J2_J0
        J1 = J1 + J0_J1 + J2_J1
        J2 = J2 + J0_J2 + J1_J2

        return [J0, J1, J2]


class MLP_block(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis) -> None:
        super().__init__()
        sec = num_layers
        self.mlp_list = nn.ModuleList([
            nn.Sequential(
                *[
                    MLPblock(dim, seq, use_norm,
                             use_spatial_fc, layernorm_axis)
                    for _ in range(sec)
                ]
            ),
            nn.Sequential(
                *[
                    MLPblock(dim*2, seq//2, use_norm,
                             use_spatial_fc, layernorm_axis)
                    for _ in range(sec)
                ]
            ),
            nn.Sequential(
                *[
                    MLPblock(dim*4, seq//4+1, use_norm,
                             use_spatial_fc, layernorm_axis)
                    for _ in range(sec)
                ]
            )
        ])

    def forward(self, x):
        J0, J1, J2 = x
        J0 = self.mlp_list[0](J0)
        J1 = self.mlp_list[1](J1)
        J2 = self.mlp_list[2](J2)
        return [J0, J1, J2]


class final_block(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis) -> None:
        super().__init__()
        # sec = num_layers 
        self.merge_mlp_0 = nn.Sequential(
            *[
                MLPblock(dim, seq, use_norm,
                         use_spatial_fc, layernorm_axis)
                for _ in range(3)
            ]
        )
        self.merge_mlp_1 = nn.Sequential(
            *[
                MLPblock(dim*2, seq//2, use_norm,
                         use_spatial_fc, layernorm_axis)
                for _ in range(3)
            ]
        )

    def forward(self, x):
        J0, J1, J2 = x
        J2 = up_dwt(J2)[:, :, :-1]
        J1 = J1 + J2
        J1 = self.merge_mlp_1(J1)
        J1 = up_dwt(J1)
        J0 = J0 + J1
        J0 = self.merge_mlp_0(J0)
        return J0


class HR_Motion(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        if 'seq_len' in args:
            seq_len = args.seq_len
        else:
            seq_len = None
        self.num_block = 4
        block_length = 12
        dim=args.hidden_dim
        seq=seq_len
        use_norm=args.with_normalization
        use_spatial_fc=args.spatial_fc_only
        num_layers=block_length
        layernorm_axis=args.norm_axis
        # print(dim,seq)
        self.initial = initial_block()
        self.final = final_block(dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis)
        self.fuse0 = fuse_block()
        self.fuse1 = fuse_block()
        self.stem_0 = MLP_block(dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis)
        self.stem_1 = MLP_block(dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis)
        
    def forward(self, x):
        x = self.initial(x)

        x = self.fuse0(x)
        x = self.stem_0(x)
        x = self.fuse1(x)
        x = self.stem_1(x)
        x = self.final(x)
        return x


class intermedia(nn.Module):
    def __init__(self, args):
        super(intermedia, self).__init__()
        if 'seq_len' in args:
            seq_len = args.seq_len
        else:
            seq_len = None
        self.num_block = 4
        block_length = 12
        
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    HR_MLP(
                        dim=args.hidden_dim,
                        seq=seq_len,
                        use_norm=args.with_normalization,
                        use_spatial_fc=args.spatial_fc_only,
                        num_layers=block_length,
                        layernorm_axis=args.norm_axis,
                    )
                )
                for b in range(self.num_block)
            ]
        )

    def forward(self, x):
        for i in range(self.num_block):
            if i < self.num_block - 1:
                x = self.stages[i](x) + x
            else:
                x = self.stages[i](x)
        return x


def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    # return HR_MLP(
    #     dim=args.hidden_dim,
    #     seq=seq_len,
    #     use_norm=args.with_normalization,
    #     use_spatial_fc=args.spatial_fc_only,
    #     num_layers=args.num_layers,
    #     layernorm_axis=args.norm_axis,
    # )
    return intermedia(args)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    # if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")
