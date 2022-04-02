import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
# import random
from tqdm import tqdm
import time
import os, sys, inspect
from icecream import ic

pwdpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from pathlib import Path
os.system('touch %s/models_def/__init__.py'%pwdpath)
os.system('touch %s/utils/__init__.py'%pwdpath)
os.system('touch %s/__init__.py'%pwdpath)
print('started.' + pwdpath)
print(sys.path)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# from dataset_openrooms_OR_BRDFLight_RAW import openrooms, collate_fn_OR
from dataset_openrooms_OR_BRDFLight_pickles import openrooms_pickle, collate_fn_OR


from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import cfg
from utils.comm import synchronize
from utils.utils_misc import *
from utils.utils_dataloader import make_data_loader
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string, find_free_port
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR

import utils.utils_config as utils_config
from utils.utils_envs import set_up_envs

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--data_root', default=None, help='path to input images')
parser.add_argument('--task_name', type=str, default='tmp', help='task name (e.g. N1: disp ref)')
parser.add_argument('--task_split', type=str, default='train', help='train, val, test', choices={"train", "val", "test"})
# Fine tune the model
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the model')
parser.add_argument("--if_train", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_val", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_vis", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_overfit_val", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--if_overfit_train", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--epochIdFineTune', type=int, default = 0, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')   
parser.add_argument('--reconstWeight', type=float, default=10, help='the weight for reconstruction error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for the rendering' )
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# Rui
# Device
parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--master_port", type=str, default='8914')

# DEBUG
parser.add_argument('--debug', action='store_true', help='Debug eval')
parser.add_argument('--batch_size_override_vis', type=int, default=-1, help='')
parser.add_argument('--if_cluster', action='store_true', help='if using cluster')
parser.add_argument('--cluster', type=str, default='kubectl', help='cluster name if if_cluster is True', choices={"kubectl", "nvidia", "ngc"})
parser.add_argument('--eval_every_iter', type=int, default=2000, help='')
parser.add_argument('--save_every_iter', type=int, default=5000, help='')
parser.add_argument('--debug_every_iter', type=int, default=2000, help='')
parser.add_argument('--max_iter', type=int, default=-1, help='')

# Pre-training
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument('--resumes_extra', type=str, help='list of extra resumed checkpoints; strings concat by #', default='NoCkpt')
parser.add_argument('--reset_latest_ckpt', action='store_true', help='remove latest_checkpoint file')
parser.add_argument('--reset_scheduler', action='store_true', help='')
parser.add_argument('--reset_lr', action='store_true', help='')
parser.add_argument('--reset_tid', action='store_true', help='')
parser.add_argument('--tid_start', type=int, default=-1)
parser.add_argument('--epoch_start', type=int, default=-1)
parser.add_argument('--test_real', action='store_true', help='')

parser.add_argument('--skip_keys', nargs='+', help='Skip those keys in the model', required=False)
parser.add_argument('--replaced_keys', nargs='+', help='Replace those keys in the model', required=False)
parser.add_argument('--replacedby', nargs='+', help='... to match those keys from ckpt. Must be in the same length as ``replace_leys``', required=False)
parser.add_argument("--if_save_pickles", type=str2bool, nargs='?', const=True, default=False)

parser.add_argument('--meta_splits_skip', nargs='+', help='Skip those keys in the model', required=False)

 # Learning rate schedule parameters
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


parser.add_argument(
    "--config-file",
    default=os.path.join(pwdpath, "configs/config.yaml"),
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "params",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

# >>>>>>>>>>>>> A bunch of modularised set-ups
opt = parser.parse_args()
os.environ['MASETER_PORT'] = str(find_free_port())
cfg.merge_from_file(opt.config_file)
cfg = utils_config.merge_cfg_from_list(cfg, opt.params)
opt.cfg = cfg
opt.pwdpath = pwdpath
opt.if_plotted = False

from utils.utils_envs import set_up_dist
handle = set_up_dist(opt)
from utils.utils_envs import set_up_folders
set_up_folders(opt)
from utils.utils_envs import set_up_logger
logger, writer = set_up_logger(opt)
opt.logger = logger
set_up_envs(opt)
opt.cfg.freeze()

if opt.is_master:
    ic(opt.cfg)
# <<<<<<<<<<<<< A bunch of modularised set-ups

# >>>>>>>>>>>>> MODEL AND OPTIMIZER
# build model
from models_def.model_joint_all import Model_Joint as the_model
model = the_model(opt, logger)

if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(opt.device)

model.freeze_BN()

if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    model.load_pretrained_MODEL_BRDF(if_load_Bs=opt.cfg.MODEL_BRDF.if_bilateral)

model.print_net()

# set up optimizers
lr_scale = 1.
optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.lr * lr_scale, betas=(0.5, 0.999) )

if opt.distributed:
    model = DDP(model, device_ids=[opt.rank], output_device=opt.rank, find_unused_parameters=True)
logger.info(red('Optimizer: '+type(optimizer).__name__))

scheduler = None
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, cooldown=0, verbose=True, threshold_mode='rel', threshold=0.01)
# <<<<<<<<<<<<< MODEL AND OPTIMIZER


# >>>>>>>>>>>>> DATASET
from utils.utils_transforms import get_transform_BRDF

transforms_train_BRDF = get_transform_BRDF('train', opt)
transforms_val_BRDF = get_transform_BRDF('val', opt)

openrooms_to_use = openrooms_pickle
make_data_loader_to_use = make_data_loader
    

if opt.if_train:
    brdf_dataset_train = openrooms_to_use(opt, 
        transforms_BRDF = transforms_train_BRDF, 
        cascadeLevel = opt.cascadeLevel, split = 'train', if_for_training=True, logger=logger)
    brdf_loader_train, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_train,
        is_train=True,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
)

if opt.if_val:
    brdf_dataset_val = openrooms_to_use(opt, 
        transforms_BRDF = transforms_val_BRDF, 
        cascadeLevel = opt.cascadeLevel, split = 'val', if_for_training=False, load_first = -1, logger=logger)
    brdf_loader_val, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_val,
        is_train=False,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
        if_distributed_override=opt.cfg.DATASET.if_val_dist and opt.distributed # default: True; -> should use gather from all GPUs if need all batches
    )

if opt.if_overfit_val and opt.if_train:
    brdf_dataset_train = openrooms_to_use(opt, 
        transforms_BRDF = transforms_val_BRDF, 
        cascadeLevel = opt.cascadeLevel, split = 'val', if_for_training=True, load_first = -1, logger=logger)

    brdf_loader_train, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_train,
        is_train=True,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
    )

if opt.if_overfit_train and opt.if_val:
    brdf_dataset_val = openrooms_to_use(opt, 
        transforms_BRDF = transforms_val_BRDF, 
        cascadeLevel = opt.cascadeLevel, split = 'train', if_for_training=False, load_first = -1, logger=logger)
    brdf_loader_val, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_val,
        is_train=False,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
        if_distributed_override=opt.cfg.DATASET.if_val_dist and opt.distributed # default: True; -> should use gather from all GPUs if need all batches
    )

if opt.if_vis:
    brdf_dataset_val_vis = openrooms_to_use(opt, 
        transforms_BRDF = transforms_val_BRDF, 
        cascadeLevel = opt.cascadeLevel, split = 'val', task='vis', if_for_training=False, load_first = opt.cfg.TEST.vis_max_samples, logger=logger)
    brdf_loader_val_vis, batch_size_val_vis = make_data_loader(
        opt,
        brdf_dataset_val_vis,
        is_train=False,
        start_iter=0,
        logger=logger,
        workers=2,
        batch_size_override=opt.batch_size_override_vis, 
        collate_fn=collate_fn_OR, 
        if_distributed_override=False
    )
    if opt.if_overfit_train:
        brdf_dataset_val_vis = openrooms_to_use(opt, 
            transforms_BRDF = transforms_val_BRDF, 
            cascadeLevel = opt.cascadeLevel, split = 'train', task='vis', if_for_training=False, load_first = opt.cfg.TEST.vis_max_samples, logger=logger)
        brdf_loader_val_vis, batch_size_val_vis = make_data_loader(
            opt,
            brdf_dataset_val_vis,
            is_train=False,
            start_iter=0,
            logger=logger,
            workers=2,
            batch_size_override=opt.batch_size_override_vis, 
            collate_fn=collate_fn_OR, 
            if_distributed_override=False
        )


# <<<<<<<<<<<<< DATASET

from utils.utils_envs import set_up_checkpointing
checkpointer, tid_start, epoch_start = set_up_checkpointing(opt, model, optimizer, scheduler, logger)


# >>>>>>>>>>>>> TRANING
from train_funcs_joint_all import get_labels_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint

tid = tid_start
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

ts_iter_end_start_list = []
ts_iter_start_end_list = []
num_mat_masks_MAX = 0

model.train()
synchronize()


if not opt.if_train:
    val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
    if opt.if_vis:
        val_params.update({'batch_size_val_vis': batch_size_val_vis})
        with torch.no_grad():
            vis_val_epoch_joint(brdf_loader_val_vis, model, val_params)
        synchronize()
    if opt.if_val:
        val_params.update({'brdf_dataset_val': brdf_dataset_val})
        with torch.no_grad():
            val_epoch_joint(brdf_loader_val, model, val_params)
else:
    for epoch_0 in list(range(opt.cfg.SOLVER.max_epoch)):
        epoch = epoch_0 + epoch_start

        time_meters = get_time_meters_joint()

        epochs_saved = []

        ts_epoch_start = time.time()
        ts_iter_end = ts_epoch_start
        
        print('=======NEW EPOCH', opt.rank)
        synchronize()

        if tid >= opt.max_iter and opt.max_iter != -1:
            break
        
        start_iter = tid_start + len(brdf_loader_train) * epoch_0
        logger.info("Starting training from iteration {}".format(start_iter))

        if cfg.SOLVER.if_test_dataloader:
            tic = time.time()
            tic_list = []

        count_samples_this_rank = 0

        for i, data_batch in tqdm(enumerate(brdf_loader_train)):

            if cfg.SOLVER.if_test_dataloader:
                if i % 100 == 0:
                    print(data_batch.keys())
                    print(opt.task_name, 'On average: %.4f iter/s'%((len(tic_list)+1e-6)/(sum(tic_list)+1e-6)))
                tic_list.append(time.time()-tic)
                tic = time.time()
                continue
            reset_tictoc = False
            # Evaluation for an epoch```

            # synchronize()
            print((tid - tid_start) % opt.eval_every_iter, opt.eval_every_iter)
            if opt.eval_every_iter != -1 and (tid - tid_start) % opt.eval_every_iter == 0:
                val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
                if opt.if_vis:
                    val_params.update({'batch_size_val_vis': batch_size_val_vis})
                    with torch.no_grad():
                        vis_val_epoch_joint(brdf_loader_val_vis, model, val_params)
                    synchronize()                
                if opt.if_val:
                    val_params.update({'brdf_dataset_val': brdf_dataset_val})
                    with torch.no_grad():
                        val_epoch_joint(brdf_loader_val, model, val_params)
                model.train()
                reset_tictoc = True
                
                synchronize()

            # Save checkpoint
            if opt.save_every_iter != -1 and (tid - tid_start) % opt.save_every_iter == 0 and 'tmp' not in opt.task_name:
                check_save(opt, tid, tid, epoch, checkpointer, epochs_saved, opt.checkpoints_path_task, logger)
                reset_tictoc = True

            if reset_tictoc:
                ts_iter_end = time.time()
            ts_iter_start = time.time()
            if tid > 5:
                ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

            if tid % opt.debug_every_iter == 0:
                opt.if_vis_debug_pac = True

            # ======= Load data from cpu to gpu
            labels_dict = get_labels_dict_joint(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            optimizer.zero_grad()
            output_dict, loss_dict = forward_joint(True, labels_dict, model, opt, time_meters, tid=tid)
            
            # print('=======loss_dict', loss_dict)
            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()

            # ======= Backward
            loss = 0.
            loss_keys_backward = []
            loss_keys_print = []

            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                if not (opt.cfg.MODEL_BRDF.if_freeze or opt.cfg.MODEL_LIGHT.freeze_BRDF_Net):
                    loss_keys_backward.append('loss_brdf-ALL')
                    loss_keys_print.append('loss_brdf-ALL')
                if 'al' in opt.cfg.MODEL_BRDF.enable_list and 'al' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-albedo') 
                    if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_albedo:
                        loss_keys_print.append('loss_brdf-albedo-reg') 
                if 'no' in opt.cfg.MODEL_BRDF.enable_list and 'no' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-normal') 
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list and 'ro' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-rough') 
                if 'de' in opt.cfg.MODEL_BRDF.enable_list and 'de' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-depth') 
                    if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_depth:
                        loss_keys_print.append('loss_brdf-depth-reg') 

            if opt.cfg.MODEL_LIGHT.enable:
                if not opt.cfg.MODEL_LIGHT.if_freeze:
                    loss_keys_backward.append('loss_light-ALL')
                    loss_keys_print.append('loss_light-ALL')

            for loss_key in loss_keys_backward:
                if loss_key in opt.loss_weight_dict:
                    loss_dict[loss_key] = loss_dict[loss_key] * opt.loss_weight_dict[loss_key]
                    print('Multiply loss %s by weight %.3f'%(loss_key, opt.loss_weight_dict[loss_key]))
            loss = sum([loss_dict[loss_key] for loss_key in loss_keys_backward])

            if opt.is_master and tid % 20 == 0:
                print('----loss_dict', loss_dict.keys())
                print('----loss_keys_backward', loss_keys_backward)

            loss.backward()

            if opt.is_master and tid % 100 == 0:
                params_train_total = 0
                params_not_train_total = 0
                for name, param in model.named_parameters():
                    if param.grad is None:
                        if param.requires_grad==True:
                            print(name, '------!!!!!!!!!!!!')
                            params_not_train_total += 1
                    else:
                        print(name, '☑')
                        params_train_total += 1
                logger.info('%d params received grad; %d params require grads but not received'%(params_train_total, params_not_train_total))

            optimizer.step()
            time_meters['backward'].update(time.time() - time_meters['ts'])
            time_meters['ts'] = time.time()
            # synchronize()

            if opt.is_master:
                loss_keys_print = [x for x in loss_keys_print if 'ALL' in x] + [x for x in loss_keys_print if 'ALL' not in x]
                logger_str = 'Epoch %d - Tid %d -'%(epoch, tid) + ', '.join(['%s %.3f'%(loss_key, loss_dict_reduced[loss_key]) for loss_key in loss_keys_print])
                logger.info(white_blue(logger_str))

                for loss_key in loss_dict_reduced:
                    if loss_dict_reduced[loss_key] != 0.:
                        writer.add_scalar('loss_train/%s'%loss_key, loss_dict_reduced[loss_key].item(), tid)
                writer.add_scalar('training/epoch', epoch, tid)

            # End of iteration logging
            ts_iter_end = time.time()
            if opt.is_master and (tid - tid_start) > 5:
                ts_iter_start_end_list.append(ts_iter_end - ts_iter_start)
                if (tid - tid_start) % 10 == 0:
                    logger.info(green('Rolling end-to-start %.2f, Rolling start-to-end %.2f'%(sum(ts_iter_end_start_list)/len(ts_iter_end_start_list), sum(ts_iter_start_end_list)/len(ts_iter_start_end_list))))
                    logger.info(green('Training timings: ' + time_meters_to_string(time_meters)))
                # if len(ts_iter_end_start_list) > 100:
                #     ts_iter_end_start_list = []
                #     ts_iter_start_end_list = []
                if opt.is_master and tid % 100 == 0:
                    usage_ratio = print_gpu_usage(handle, logger)
                    writer.add_scalar('training/GPU_usage_ratio', usage_ratio, tid)
                    writer.add_scalar('training/batch_size_per_gpu', len(data_batch['image_path']), tid)
                    writer.add_scalar('training/gpus', opt.num_gpus, tid)
                    current_lr = optimizer.param_groups[0]['lr']
                    # if opt.cfg.SOLVER.if_warm_up:
                    writer.add_scalar('training/lr', current_lr, tid)

            if tid % opt.debug_every_iter == 0:
                opt.if_vis_debug_pac = False

            tid += 1
            if tid >= opt.max_iter and opt.max_iter != -1:
                break
    
        if opt.cfg.SOLVER.if_warm_up:
            # scheduler.step(epoch)
            pass