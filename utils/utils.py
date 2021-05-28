import numpy as np
import logging
import os
import random
import os
import torch

Abnormal_type=['Abuse','Arrest','Arson','Assault','Burglary',
               'Explosion','Fighting','RoadAccidents','Robbery',
               'Shooting','Shoplifting','Stealing','Vandalism','Normal']


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.vals=[]

    def __format__(self, format_spec):
        f=0
        if len(self.vals)!=0:
            f=(sum(self.vals)/len(self.vals))
        return ('{:'+format_spec+'}').format(f)

    def val(self):
        if len(self.vals) != 0:
            f = sum(self.vals) / len(self.vals)
        else:
            f=0
        return f

    def update(self,val):
        if isinstance(val,np.ndarray):
            self.vals.append(val[0])
        elif isinstance(val,np.float64):
            self.vals.append(val)
        else:
            self.vals.append(val.detach().cpu().item())

def show_params(args):
    params=vars(args)
    keys=sorted(params.keys())

    for k in keys:
        print(k,'\t',params[k])

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def log_param(logger,args):
    params=vars(args)
    keys=sorted(params.keys())
    for k in keys:
        logger.info('{}\t{}'.format(k,params[k]))

import time
def get_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def mkdir(dir):
    if not os.path.exists(dir):
        try:os.mkdir(dir)
        except:pass

def set_seeds(seed):
    print('set seed {}'.format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def random_perturb(v_len, num_segments):
    """
    Given the length of video and sampling number, which segments should I choose?
    Random sampling is used.
    :param v_len: length of video
    :param num_segments: expected number of segments
    :return: a list of indices to sample
    """
    random_p = np.arange(num_segments) * v_len / num_segments
    for i in range(num_segments):
        if i < num_segments - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)

def get_epoch_idx(epoch,milestones):
    count=0
    for milestone in milestones:
        if epoch>milestone:
            count+=1
    return count

def get_lr_scheduler(args,optimizer):
    lr_policy = lambda epoch: (epoch + 0.5) / (args.warmup_epochs) \
        if epoch < args.warmup_epochs else 1
    lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_policy)
    return lr_scheduler

def weights_normal_init(model, dev=0.01):
    import torch
    from torch import nn

    # torch.manual_seed(2020)
    # torch.cuda.manual_seed_all(2020)
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias!=None:
                    m.bias.data.fill_(0)