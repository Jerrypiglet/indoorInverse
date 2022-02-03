import shutil

import torch
torch_version = torch.__version__
# import apex
import torch.distributed as dist
import numpy as np
import os, sys
import glob

from utils.utils_misc import *
import ntpath
import logging
import os
from tqdm import tqdm
from utils.comm import get_world_size
import nvidia_smi

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_optimizer(parameters, cfg):
    if cfg.method == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                    lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters),
                                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, parameters),
                                         lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


class Logger(object):
    def __init__(self, filename="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def save_checkpoint(state, is_best, task_name, filename='checkpoint.pth.tar', save_folder=None, logger=None):
    assert save_folder is not None
    save_path = os.path.join(os.path.join(save_folder, task_name), filename)
    best_path = os.path.join(os.path.join(save_folder, task_name), 'best_'+filename)
    latest_path = os.path.join(os.path.join(save_folder, task_name), 'latest.pth.tar')
    
    if not os.path.isdir(os.path.join(save_folder, task_name)):
        os.mkdir(os.path.join(save_folder, task_name))
    
    save_path = os.path.join(os.path.join(save_folder, task_name), filename)
    torch.save(state, save_path)
    logger.info(colored("Saved to " + save_path, 'white', 'on_magenta'))

    if is_best:
        print("best", state["eval_loss"])
        shutil.copyfile(save_path, best_path)
    else:
        print("NOT best", state["eval_loss"])

    shutil.copyfile(save_path, latest_path)  


def printensor(msg):
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}")
            print(f"min/max/mean: {tensor.max()}, {tensor.min()}, {tensor.mean()}")
    return printer

def reduce_loss_dict(loss_dict, mark='', if_print=False, logger=None):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size() # NUM of GPUs
    if world_size < 2 or len(loss_dict.keys())==0:
        logger.debug('[train_utils] world_size==%d; not reduced!'%world_size)
        return loss_dict
    
    # print(loss_dict.keys())
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
            # print(k, loss_dict[k].shape, loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        if if_print:
            print(mark, '-0-all_losses', all_losses)
        dist.reduce(all_losses, dst=0)
        if if_print:
            print(mark, '-1-all_losses', all_losses)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
        if if_print:
            print(mark, '-2-reduced_losses', reduced_losses)
    return reduced_losses



def copy_py_files(root_path, dest_path, exclude_paths=[]):
    from multiprocessing import Pool
    origin_path_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                origin_path = os.path.join(root, file)
                # print(os.path.join(root, file))
                exclude_flag = False
                for exclude_path in exclude_paths:
                    if exclude_path != '' and exclude_path in origin_path:
                        exclude_flag = True
                        break
                else:
                    origin_path_list.append([origin_path, dest_path])
                    # os.system('cp %s %s/'%(origin_path, dest_path))
                    # print('Copied ' + origin_path)

    with Pool(processes=12, initializer=np.random.seed(123456)) as pool:
        for _ in list(tqdm(pool.imap_unordered(copy_file, origin_path_list), total=len(origin_path_list))):
            pass

def copy_file(origin_dest):
    os.system('cp %s %s/'%(origin_dest[0], origin_dest[1]))

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def print_gpu_usage(handle, logger):
    # print GPU usage
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    usage_str = f'mem: {mem_res.used / (1024**2)} (GiB) - '
    usage_percent_str = f'{100 * (mem_res.used / mem_res.total):.3f}%'
    logger.info(red(usage_str + usage_percent_str))
    # print(f'mem: {mem_res.used / (1024**2)} (GiB)') # usage in GiB
    # print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%') # percentage usage
    return mem_res.used / mem_res.total

def time_meters_to_string(time_meters):
    # for key in time_meters:
    #     if key != 'ts':
    #         meter = time_meters[key]
    #         print(key, meter.avg)
    return ', '.join(['%s - %.2f'%(key, time_meters[key].avg) for key in time_meters if key != 'ts'])

def clean_up_checkpoints(checkpoint_folder, leave_N, start_with='checkpoint_', logger=None):
    # checkpoint_folder = '/home/ruizhu/Documents/Projects/adobe_rui_camera-calibration-redux/checkpoint/test'
    # list_checkpoints = glob.glob(checkpoint_folder+'/checkpoint*.pth.tar')
    list_checkpoints = list(filter(os.path.isfile, glob.glob(str(checkpoint_folder)+'/%s*.*'%start_with)))
    list_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # print([ntpath.basename(filename) for filename in list_checkpoints])


    last_checkpoint_file = os.path.join(checkpoint_folder, "last_checkpoint")
    try:
        with open(last_checkpoint_file, "r") as f:
            last_saved = f.read()
            last_saved = last_saved.strip()
    except IOError:
        last_saved = None
        pass

    if logger is None:
        logger = logging.getLogger('clean_up_checkpoints')

    if len(list_checkpoints) > leave_N:
        for checkpoint_path in list_checkpoints[leave_N:]:
            # last_saved = '/home/ruizhu/Documents/Projects/adobe_rui_camera-calibration-redux/checkpoint/tmp2/checkpointer_epoch0010_iter0000100.pth'
            # print(ntpath.basename(last_saved), ntpath.basename(checkpoint_path), last_saved)
            if last_saved is not None and ntpath.basename(last_saved) == ntpath.basename(checkpoint_path):
                logger.info(magenta('Skipping latest at '+last_saved))
                continue
            os.system('rm %s'%checkpoint_path)
            logger.info(white_blue('removed checkpoint at '+checkpoint_path))

def check_save(opt, tid, epoch_save, epoch_total, checkpointer, epochs_saved, checkpoints_folder, logger=None, is_better=False):
    arguments = {"iteration": tid, 'epoch': epoch_total,}
    # if rank == 0 and epoch != 0 and (epoch < opt.save_every_epoch or epoch % opt.save_every_epoch == 0) and epoch not in epochs_saved:
    # if opt.is_master and (epoch_save < opt.save_every_epoch or epoch_save % opt.save_every_epoch == 0) and epoch_save not in epochs_saved:
    if opt.is_master and epoch_save not in epochs_saved:

        saved_filename = checkpointer.save('checkpointer_epoch%04d_iter%07d'%(epoch_total, tid), **arguments)
        if opt.cfg.TRAINING.MAX_CKPT_KEEP != -1:
            clean_up_checkpoints(checkpoints_folder, leave_N=opt.cfg.TRAINING.MAX_CKPT_KEEP, start_with='checkpointer_', logger=logger)

        epochs_saved.append(epoch_save)
    
    if opt.is_master and is_better:
        ckpt_filepath = 'best_checkpointer_epoch%04d_iter%07d'%(epoch_total, tid)
        saved_filename = checkpointer.save(ckpt_filepath, **arguments)
        logger.info(green('Saved BEST checkpoint to '+saved_filename))

def freeze_bn_in_module(module, if_print=True):
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        if if_print:
            print(red('-- turning off BN in '), module)
        mod.eval()
        # mod.param.requires_grad = False
        mod.track_running_stats = False

    # if isinstance(module, torch.nn.modules.groupnorm._GroupNorm):
    #     # print('--convert_syncbn_model_hvd converting...', module)
    #     mod.eval()
        # print(mod)
        # mod.weight.requires_grad = False
        # mod.bias.requires_grad = False
        # mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        # mod.running_mean = module.running_mean
        # mod.running_var = module.running_var
        # if module.affine:
        #     mod.weight.data = module.weight.data.clone().detach()
        #     mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        freeze_bn_in_module(child, if_print=if_print)
        # mod.add_module(name, convert_syncbn_model_hvd(child))
    # # TODO(jie) should I delete model explicitly?
    # del module
    # return mod

def unfreeze_bn_in_module(module, if_print=True):
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        #  or isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if if_print:
            print(red('-- turning ON BN in '), module)
        mod.train()
        mod.track_running_stats = True

    # if isinstance(module, torch.nn.modules.groupnorm._GroupNorm):
    #     # print('--convert_syncbn_model_hvd converting...', module)
    #     mod.eval()
        # print(mod)
        # mod.weight.requires_grad = False
        # mod.bias.requires_grad = False
        # mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        # mod.running_mean = module.running_mean
        # mod.running_var = module.running_var
        # if module.affine:
        #     mod.weight.data = module.weight.data.clone().detach()
        #     mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        unfreeze_bn_in_module(child, if_print=if_print)
        # mod.add_module(name, convert_syncbn_model_hvd(child))
    # # TODO(jie) should I delete model explicitly?
    # del module
    # return mod
