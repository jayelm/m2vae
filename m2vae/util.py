"""
Various training/serialization utils
"""

import os
import json
import shutil

import torch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, exp_dir, filename='checkpoint.pth'):
    torch.save(state, os.path.join(exp_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(exp_dir, filename),
                        os.path.join(exp_dir, 'model_best.pth'))


def load_checkpoint(exp_dir, filename='checkpoint.pth',
                    cuda=False):
    device = 'gpu' if cuda else 'cpu'
    return torch.load(os.path.join(exp_dir, filename),
                      map_location=device)


def restore_checkpoint(model, optimizer, exp_dir, filename='checkpoint.pth',
                       cuda=False):
    ckpt = load_checkpoint(exp_dir, filename=filename, cuda=cuda)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])


def restore_args(args, exp_dir):
    if not os.path.exists(args.exp_dir):
        raise RuntimeError("Can't find {}".format(args.exp_Dir))

    exp_args = load_args(args.exp_dir)
    for arg, val in exp_args.items():
        if not arg in args:
            args.__setattr__(arg, val)


def load_args(exp_dir, filename='args.json'):
    with open(os.path.join(exp_dir, filename), 'r') as f:
        return json.load(f)


def save_args(args, exp_dir, filename='args.json'):
    with open(os.path.join(exp_dir, filename), 'w') as f:
        json.dump(vars(args), f, indent=4, separators=(',', ': '), sort_keys=True)


def load_metrics(exp_dir, filename='metrics.json'):
    with open(os.path.join(exp_dir, filename), 'r') as f:
        return json.load(f)


def save_metrics(metrics, exp_dir, filename='metrics.json'):
    with open(os.path.join(exp_dir, filename), 'w') as f:
        json.dump(dict(metrics), f)


def is_resumable(exp_dir):
    return (
        os.path.exists(os.path.join(exp_dir, 'metrics.json')) and
        os.path.exists(os.path.join(exp_dir, 'args.json')) and
        os.path.exists(os.path.join(exp_dir, 'checkpoint.pth')))
