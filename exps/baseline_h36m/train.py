from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from test import test
import sys

sys.path.append("/media/store1/ll-ben/simple_base/redo_seed/seed/siMLPe")

from lib.datasets.h36m_eval import H36MEval
from lib.utils.pyt_utils import link_file, ensure_dir
from lib.utils.logger import get_logger, print_and_log_info
from lib.datasets.h36m import H36MDataset
from model import siMLPe as Model
from config import config
import copy
import numpy as np
import math
import json
import argparse
import os
import sys
from os.path import dirname

sys.path.append(dirname(dirname(os.getcwd())))
print(dirname(dirname(os.getcwd())))


dwt = DWT1DForward(wave=config.wavelet, J=1).cuda()
idwt = DWT1DInverse(wave=config.wavelet).cuda()

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=114514, help='=seed')
parser.add_argument('--temporal-only', action='store_true',
                    help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str,
                    default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization',
                    action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true',
                    help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(f"logs/{args.exp_name}", 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
# config.motion_mlp.num_layers = args.num

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)

    """Unable DCT hook"""
    # dct_m = np.eye(N)
    
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer):
    if nb_iter > 30000:
        current_lr = 6e-5 * (0.85 ** ((nb_iter - 30000) // 3000))
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr):

    if config.deriv_input:
        b, n, c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(
            dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
        # h36m_motion_input_unfold = h36m_motion_input.unfold(1,10,5)
        # hurdle_sequence = torch.movedim(h36m_motion_input_unfold,-2,-1)
        # hurdle_varience = torch.var(hurdle_sequence,dim=(2,3))
        # hurdle_factor = torch.sum(torch.abs(torch.diff(hurdle_varience)),1)

    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred = model(h36m_motion_input_.cuda())
    # print(type(motion_pred))
    # motion_pred = torch.mean(motion_pred,1,False)
    motion_pred = torch.matmul(
        idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:,
                                  :config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

 

    # hurdle =
    b, n, c = h36m_motion_target.shape

    motion_pred = motion_pred.reshape(b, n, 22, 3).reshape(-1, 3)

    h36m_motion_target = h36m_motion_target.cuda().reshape(b, n, 22, 3).reshape(-1, 3)

    """
    tweak the loss here, try to enhance or decrease the influence by hard poses
    """
    norm = torch.norm(motion_pred - h36m_motion_target, 2, 1)
    # print(norm.shape)
    norm = norm.reshape(b, -1)
    norm = torch.mean(norm, 1)
    # print(norm.shape)

    # norm = norm * torch.clamp((1 - hurdle_factor * 0 * (1/torch.log10(nb_iter/1500))).cuda(),0)
    norm.reshape(-1)

    loss = torch.mean(norm)
    # original loss ↓
    # loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss


        # motion_pred = motion_pred.reshape(b,n,22,3)
        # motion_pred = motion_pred.movedim(-1,-2)
        # motion_pred = motion_pred.reshape(b,n,-1)
        # pred_yl, pred_yh = dwt(motion_pred)

        # motion_gt = h36m_motion_target.reshape(b,n,22,3).movedim(-1,-2).reshape(b,n,-1)
        # gt_yl, gt_yh = dwt(motion_gt)

        # pred = torch.concat([pred_yl, *pred_yh], -1)
        # gt =  torch.concat([gt_yl, *gt_yh], -1)
        # # print(pred.shape)
        # haar_loss = torch.mean(torch.norm(pred - gt, 2, 1))
        # loss = loss + haar_loss

    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(
        nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


model = Model(config)
model.train()
model.cuda()

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = H36MDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = H36MEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                             num_workers=1, drop_last=False,
                             sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None:
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(
        logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

print(__file__)

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (h36m_motion_input, h36m_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model,
                                                 optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every == 0:
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(
                logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every == 0:
            torch.save(model.state_dict(), config.snapshot_dir +
                       '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(eval_config, model, eval_dataloader)
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters:
            break
        nb_iter += 1

writer.close()
