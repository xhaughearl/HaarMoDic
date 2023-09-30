import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange

from pytorch_wavelets import DWT2D, IDWT2D

dwt = DWT2D(wave='haar', J=1).cuda()
idwt = IDWT2D(wave='haar').cuda()


class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(
                self.config.motion.dim, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(
                self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):

        inp = motion_input.clone()
        # inp = inp[None,]
        # inp = dwt(inp)
        # # print(inp[1][0].shape)
        # inp = torch.concat(
        #     [
        #         inp[0][0],
        #         inp[1][0][0][:, 0],
        #         inp[1][0][0][:, 1],
        #         inp[1][0][0][:, 2]
        #     ], 1
        # )

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(inp)
            motion_feats = self.arr0(motion_feats)
        
        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        # out = motion_feats
        # yl = out[:, 0:25][None,]
        # yh = [out[:, (i+1)*25:(i+2)*25].unsqueeze(1) for i in range(3)]
        # yh = torch.concat(yh, 1)[None, :]
        # out = idwt((yl, [yh]))[0]

        return motion_feats
