from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from utils.utils_misc import *
from utils.comm import synchronize, get_rank
import os, sys
from utils.utils_misc import only1true
import os
from utils import transform

def set_up_root(opt, cfg):
    assert opt.cluster in cfg.PATH.cluster_names
    opt.CLUSTER_ID = cfg.PATH.cluster_names.index(opt.cluster)
    opt.if_pad = False

    cfg.PATH.root = cfg.PATH.root_cluster[opt.CLUSTER_ID] if opt.if_cluster else cfg.PATH.root_local

def set_up_envs(opt):
    if opt.if_cluster:
        # opt.cfg.TRAINING.MAX_CKPT_KEEP = -1
        opt.if_save_pickles = False

    if opt.cfg.DEBUG.if_test_real:
        opt.cfg.DEBUG.if_dump_perframe_BRDF = True
        opt.cfg.TEST.vis_max_samples = 20000

    if opt.if_cluster:
        opt.cfg.DEBUG.if_fast_BRDF_labels = False
        opt.cfg.DEBUG.if_fast_light_labels = False

    if opt.cfg.DEBUG.if_fast_BRDF_labels:
        opt.cfg.DATASET.dataset_path_local = opt.cfg.DATASET.dataset_path_local_fast_BRDF

    if opt.cfg.DATASET.if_quarter and not opt.if_cluster:
        opt.cfg.DATASET.dataset_path_local = opt.cfg.DATASET.dataset_path_local_quarter
    opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_local
    opt.cfg.DATASET.dataset_path_pickle = opt.cfg.DATASET.dataset_path_pickle_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_pickle_local

    opt.cfg.DATASET.png_path = opt.cfg.DATASET.png_path_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.png_path_local
    opt.cfg.DATASET.dataset_path_mini = opt.cfg.DATASET.dataset_path_mini_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_mini_local
    opt.cfg.DATASET.png_path_mini = opt.cfg.DATASET.png_path_mini_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.png_path_mini_local
    opt.cfg.DATASET.envmap_path = opt.cfg.DATASET.envmap_path_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.envmap_path_local

    opt.cfg.DATASET.iiw_path = opt.cfg.DATASET.iiw_path_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.iiw_path_local
    opt.cfg.DATASET.nyud_path = opt.cfg.DATASET.nyud_path_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.nyud_path_local

    if opt.data_root is not None:
        opt.cfg.DATASET.dataset_path = opt.data_root

    if opt.cfg.PATH.OR_lists_path_if_zhengqinCVPR:
        # assert False, 'paths not correctly configured! (we use Zhengqins test set as val set, but they are in a different path (/eccv20dataset/DatasetNew_test) than the main dataset'
        opt.cfg.PATH.OR_lists_path = opt.cfg.PATH.OR_lists_path_zhengqinCVPR
    opt.cfg.DATASET.dataset_list = os.path.join(opt.cfg.PATH.OR_lists_path, 'list')

    if opt.cfg.DATASET.mini:
        opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_mini
        opt.cfg.DATASET.dataset_list = opt.cfg.DATASET.dataset_list_mini
        opt.cfg.DATASET.png_path = opt.cfg.DATASET.png_path_mini
    opt.cfg.DATASET.dataset_list = os.path.join(opt.cfg.PATH.root, opt.cfg.DATASET.dataset_list)

    print('======= DATASET.dataset_path ', opt.cfg.DATASET.dataset_path)

    opt.cfg.PATH.matcls_matIdG1_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.matcls_matIdG1_path)
    opt.cfg.PATH.matcls_matIdG2_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.matcls_matIdG2_path)

    # ===== data =====
    opt.cfg.DATA.data_read_list = [x for x in list(set(opt.cfg.DATA.data_read_list.split('_'))) if x != '']

    opt.if_pad = False
    opt.if_resize = False

    if opt.cfg.DEBUG.if_nyud: # if True, should pad indeptly for each sample
        opt.pad_op_nyud = None

    if opt.cfg.DEBUG.if_test_real:
        opt.cfg.DATA.load_light_gt = False
        if not opt.cfg.DEBUG.if_load_dump_BRDF_offline:
            opt.cfg.DATA.data_read_list = ''
            opt.cfg.DATASET.if_no_gt_BRDF = True
        opt.cfg.DATASET.if_no_gt_light = True

    # if opt.cfg.DEBUG.if_iiw:
    #     opt.cfg.DATASET.if_no_gt_BRDF = True

    # ====== BRDF =====
    if isinstance(opt.cfg.MODEL_BRDF.enable_list, str):
        opt.cfg.MODEL_BRDF.enable_list = [x for x in opt.cfg.MODEL_BRDF.enable_list.split('_') if x != '']
    opt.cfg.MODEL_BRDF.loss_list = [x for x in opt.cfg.MODEL_BRDF.loss_list.split('_') if x != '']

    assert opt.cfg.MODEL_BRDF.depth_activation in ['sigmoid', 'relu', 'tanh', 'midas']

    # ====== per-pixel lighting =====
    if opt.cfg.MODEL_LIGHT.enable:
        # opt.cfg.DATA.load_brdf_gt = True
        # if not opt.cfg.DEBUG.if_test_real:
            # opt.cfg.DATA.load_light_gt = True
            # if opt.cfg.DATA.load_light_gt:
            #     opt.cfg.DATA.data_read_list += 'al_no_de_ro'.split('_')
        if opt.cfg.MODEL_LIGHT.use_GT_brdf and opt.cfg.MODEL_BRDF.enable:
            opt.cfg.DATA.load_brdf_gt = True
            opt.cfg.MODEL_LIGHT.freeze_BRDF_Net = True
            opt.cfg.MODEL_BRDF.if_freeze = True

        if opt.cfg.MODEL_LIGHT.freeze_BRDF_Net:
            opt.cfg.MODEL_BRDF.if_freeze = True
        #     opt.cfg.MODEL_BRDF.enable = False
        #     opt.cfg.MODEL_BRDF.enable_list = ''
        #     opt.cfg.MODEL_BRDF.loss_list = ''
        # else:
        #     opt.cfg.MODEL_BRDF.enable = True
        #     opt.cfg.MODEL_BRDF.enable_list += 'al_no_de_ro'.split('_')
        #     opt.cfg.MODEL_BRDF.enable_BRDF_decoders = True
        #     if opt.cfg.MODEL_LIGHT.freeze_BRDF_Net:
        #         opt.cfg.MODEL_BRDF.if_freeze = True

        opt.cfg.MODEL_LIGHT.enable_list = opt.cfg.MODEL_LIGHT.enable_list.split('_')

    # ====== BRDF, cont. =====
    opt.cfg.MODEL_BRDF.enable_BRDF_decoders = len(opt.cfg.MODEL_BRDF.enable_list) > 0

    # ic(opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders)
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        # if not opt.cfg.DEBUG.if_test_real:
        # opt.cfg.DATA.load_brdf_gt = True
        opt.depth_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        if not opt.cfg.MODEL_LIGHT.freeze_BRDF_Net:
            opt.cfg.MODEL_BRDF.loss_list += opt.cfg.MODEL_BRDF.enable_list

    # ===== check if flags are legal =====
    check_if_in_list(opt.cfg.DATA.data_read_list, opt.cfg.DATA.data_read_list_allowed)
    check_if_in_list(opt.cfg.MODEL_BRDF.enable_list, opt.cfg.MODEL_BRDF.enable_list_allowed)
    check_if_in_list(opt.cfg.MODEL_BRDF.loss_list, opt.cfg.MODEL_BRDF.enable_list_allowed)

    # extra BRDF net params
    opt.cfg.MODEL_BRDF.encoder_exclude = opt.cfg.MODEL_BRDF.encoder_exclude.split('_')


    # export
    opt.cfg.PATH.torch_home_path = opt.cfg.PATH.torch_home_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.torch_home_local
    os.system('export TORCH_HOME=%s'%opt.cfg.PATH.torch_home_path)

    opt.cfg.PATH.pretrained_path = opt.cfg.PATH.pretrained_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.pretrained_local
    opt.cfg.PATH.models_ckpt_path = opt.cfg.PATH.models_ckpt_cluster[opt.CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.models_ckpt_local

    # dump
    if opt.cfg.DEBUG.if_dump_shadow_renderer:
        opt.cfg.DEBUG.if_dump_anything = True
        opt.if_vis = True

        opt.cfg.MODEL_LIGHT.load_pretrained_MODEL_BRDF = False
        opt.cfg.MODEL_LIGHT.load_pretrained_MODEL_LIGHT = False
        opt.cfg.MODEL_BRDF.use_scale_aware_depth = True
        opt.cfg.MODEL_BRDF.depth_activation = 'relu'
        opt.cfg.DATA.data_read_list += ['mesh', 'de']

    # extra loss weights
    opt.loss_weight_dict = {}

def check_if_in_list(list_to_check, list_allowed, module_name='Unknown Module'):
    if len(list_to_check) == 0:
        return
    if isinstance(list_to_check, str):
        list_to_check = list_to_check.split('_')
    list_to_check = [x for x in list_to_check if x != '']
    if not all(e in list_allowed for e in list_to_check):
        print(list_to_check, list_allowed)
        error_str = red('Illegal %s of length %d: %s'%(module_name, len(list_to_check), '_'.join(list_to_check)))
        raise ValueError(error_str)



def set_up_logger(opt):
    from utils.logger import setup_logger, Logger, printer
    import sys

    # === LOGGING
    sys.stdout = Logger(Path(opt.summary_path_task) / 'log.txt')
    # sys.stdout = Logger(opt.summary_path_task / 'log.txt')
    logger = setup_logger("logger:train", opt.summary_path_task, opt.rank, filename="logger_maskrcn-style.txt")
    logger.info(red("==[config]== opt"))
    logger.info(opt)
    logger.info(red("==[config]== cfg"))
    logger.info(opt.cfg)
    # logger.info(red("==[config]== Loaded configuration file {}".format(opt.config)))
    # logger.info(red("==[opt.semseg_configs]=="))
    # logger.info(opt.semseg_configs)

    # with open(opt.config, "r") as cf:
    #     config_str = "\n" + cf.read()
    #     # logger.info(config_str)
    printer = printer(opt.rank, debug=opt.debug)

    if opt.is_master and 'tmp' not in opt.task_name and not opt.cfg.DEBUG.if_test_real:
        exclude_list = ['apex', 'logs_bkg', 'archive', 'train_cifar10_py', 'train_mnist_tf', 'utils_external', 'build/'] + \
            ['Summary', 'summary_vis', 'Checkpoint', 'logs', '__pycache__', 'snapshots', '.vscode', '.ipynb_checkpoints', 'azureml-setup', 'azureml_compute_logs']
        # if opt.if_cluster:
        copy_py_files(opt.pwd_path, opt.summary_vis_path_task_py, exclude_paths=[str(opt.SUMMARY_PATH), str(opt.CKPT_PATH), str(opt.summary_vis_PATH)]+exclude_list)
        os.system('cp -r %s %s'%(opt.pwd_path, opt.summary_vis_path_task_py / 'train'))
        logger.info(green('Copied source files %s -> %s'%(opt.pwd_path, opt.summary_vis_path_task_py)))
        # folders = [f for f in Path('./').iterdir() if f.is_dir()]
        # for folder in folders:
        #     folder_dest = opt.summary_vis_path_task_py / folder.name
        #     if not folder_dest.exists() and folder.name not in exclude_list:
        #         os.system('cp -r %s %s'%(folder, folder_dest))
    synchronize()

    if opt.is_master:
        writer = SummaryWriter(opt.summary_path_task, flush_secs=opt.cfg.flush_secs)
        print(green('=====>Summary writing to %s'%opt.summary_path_task))
    else:
        writer = None
    # <<<< SUMMARY WRITERS

    return logger, writer



def set_up_folders(opt):
    from utils.global_paths import SUMMARY_PATH, summary_vis_PATH, CKPT_PATH
    opt.SUMMARY_PATH, opt.summary_vis_PATH, opt.CKPT_PATH = SUMMARY_PATH, summary_vis_PATH, CKPT_PATH

    # >>>> SUMMARY WRITERS
    if opt.if_cluster:
        if opt.cluster == 'kubectl':
            opt.home_path = Path('/ruidata/indoorInverse/') 
        elif opt.cluster == 'nvidia':
            opt.home_path = Path('/home/ruzhu/Documents/Projects/')
        elif opt.cluster == 'ngc':
            opt.home_path = Path('/newfoundland/indoorInverse/')
            # opt.SUMMARY_PATH_ALL = opt.home_path / SUMMARY_PATH
            # opt.home_path_tmp = Path('/result/')

        opt.CKPT_PATH = opt.home_path / CKPT_PATH
        opt.SUMMARY_PATH = opt.home_path / SUMMARY_PATH
        # if opt.cluster == 'ngc':
        #     opt.SUMMARY_PATH = opt.home_path_tmp / SUMMARY_PATH
        #     opt.SUMMARY_PATH.mkdir(exist_ok=True)
        opt.summary_vis_PATH = opt.home_path / summary_vis_PATH

    if not opt.if_cluster or 'DATE' in opt.task_name:
        if opt.resume != 'resume':
            opt.task_name = get_datetime() + '-' + opt.task_name
        # else:
        #     opt.task_name = opt.resume
        # print(opt.cfg)
    #     opt.root = opt.cfg.PATH.root_local
    # else:
    #     opt.root = opt.cfg.PATH.root_cluster
    opt.summary_path_task = opt.SUMMARY_PATH / opt.task_name
    # if opt.cluster == 'ngc':
    #     opt.summary_path_all_task = opt.SUMMARY_PATH_ALL / opt.task_name
    opt.checkpoints_path_task = opt.CKPT_PATH / opt.task_name
    opt.summary_vis_path_task = opt.summary_vis_PATH / opt.task_name
    opt.summary_vis_path_task_py = opt.summary_vis_path_task / 'py_files'

    save_folders = [opt.summary_path_task, opt.summary_vis_path_task, opt.summary_vis_path_task_py, opt.checkpoints_path_task, ]
    print('====%d/%d'%(opt.rank, opt.num_gpus), opt.checkpoints_path_task)

    if opt.is_master:
        for root_folder in [opt.SUMMARY_PATH, opt.CKPT_PATH, opt.summary_vis_PATH]:
            if not root_folder.exists():
                root_folder.mkdir(exist_ok=True)
        if_delete = 'n'
        print(green(opt.summary_path_task), os.path.isdir(opt.summary_path_task))
        if os.path.isdir(opt.summary_path_task):
            if 'POD' in opt.task_name:
                print('====opt.summary_path_task exists! %s'%opt.summary_path_task)
                if opt.resume != 'resume':
                    raise RuntimeError('====opt.summary_path_task exists! %s; opt.resume: %s'%(opt.summary_path_task, opt.resume))
                if_delete = 'n'
                # opt.resume = opt.task_name
                # print(green('Resuming task %s'%opt.resume))

                if opt.reset_latest_ckpt:
                    os.system('rm %s'%(os.path.join(opt.checkpoints_path_task, 'last_checkpoint')))
                    print(green('Removed last_checkpoint shortcut for %s'%opt.resume))
            else:
                if opt.resume == 'NoCkpt':
                    if_delete = 'y'
                elif opt.resume == 'resume':
                    if_delete = 'n'
                else:
                    if_delete = input(colored('Summary path %s already exists. Delete? [y/n] '%opt.summary_path_task, 'white', 'on_blue'))
                    # if_delete = 'y'
            if if_delete == 'y':
                for save_folder in save_folders:
                    os.system('rm -rf %s'%save_folder)
                    print(green('Deleted summary path %s'%save_folder))
        for save_folder in save_folders:
            if not Path(save_folder).is_dir() and opt.rank == 0:
                Path(save_folder).mkdir(exist_ok=True)

    synchronize()


def set_up_dist(opt):
    import nvidia_smi

    # >>>> DISTRIBUTED TRAINING
    torch.manual_seed(opt.cfg.seed)
    np.random.seed(opt.cfg.seed)
    random.seed(opt.cfg.seed)

    opt.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = opt.num_gpus > 1
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        opt.process_group = torch.distributed.init_process_group(
            backend="nccl", world_size=opt.num_gpus, init_method="env://"
        )
        # synchronize()
    # device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    opt.device = 'cuda'
    opt.if_cuda = opt.device == 'cuda'
    opt.rank = get_rank()
    opt.is_master = opt.rank == 0
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(opt.rank)
    # <<<< DISTRIBUTED TRAINING
    return handle

def set_up_checkpointing(opt, model, optimizer, scheduler, logger):
    from utils.checkpointer import DetectronCheckpointer

    # >>>> CHECKPOINTING
    save_to_disk = opt.is_master
    checkpointer = DetectronCheckpointer(
        opt, model, optimizer, scheduler, opt.CKPT_PATH, opt.checkpoints_path_task, save_to_disk, logger=logger, if_reset_scheduler=opt.reset_scheduler
    )
    tid_start = 0
    epoch_start = 0

    if opt.resume != 'NoCkpt':
        print('=+++++=opt.resume', opt.resume)
        if opt.resume == 'resume':
            opt.resume = opt.task_name
        replace_kws = []
        replace_with_kws = []
        if opt.replaced_keys is not None and opt.replacedby is not None:
            assert len(opt.replaced_keys) == len(opt.replacedby)
            replace_kws += opt.replaced_keys
            replace_with_kws += opt.replacedby
        checkpoint_restored, _, _ = checkpointer.load(task_name=opt.resume, skip_kws=opt.skip_keys if opt.skip_keys is not None else [], replace_kws=replace_kws, replace_with_kws=replace_with_kws)
    
        if opt.resumes_extra != 'NoCkpt':
            resumes_extra_list = opt.resumes_extra.split('#')
            for resume_extra in resumes_extra_list:
                _, _, _ = checkpointer.load(task_name=resume_extra, skip_kws=opt.skip_keys if opt.skip_keys is not None else [], replace_kws=replace_kws, replace_with_kws=replace_with_kws, prefix='[RESUME EXTRA] ')

        if 'iteration' in checkpoint_restored and not opt.reset_tid:
            tid_start = checkpoint_restored['iteration']
        if 'epoch' in checkpoint_restored and not opt.reset_tid:
            epoch_start = checkpoint_restored['epoch']
        if opt.tid_start != -1 and opt.epoch_start != -1:
            tid_start = opt.tid_start
            epoch_start = opt.epoch_start
        print(checkpoint_restored.keys())
        logger.info(colored('Restoring from epoch %d - iter %d'%(epoch_start, tid_start), 'white', 'on_blue'))

    if opt.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.cfg.SOLVER.lr
            
    # <<<< CHECKPOINTING
    return checkpointer, tid_start, epoch_start