# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DTYPE = "float32"

_C.MM1_DEBUG = False

_C.PATH = CN()
_C.PATH.cluster_names = ['kubectl']
_C.PATH.root = ''
_C.PATH.root_local = '/home/ruizhu/Documents/Projects/indoorInverse/train'
_C.PATH.root_cluster = ['.', '.', '.']

_C.PATH.OR_lists_path = 'data/openrooms/list_OR_V4full'
_C.PATH.OR_lists_path_if_zhengqinCVPR = False
_C.PATH.OR_lists_path_zhengqinCVPR = 'data/openrooms/list_OR_V4full_zhengqinCVPR'

_C.PATH.matcls_matIdG1_path = 'data/openrooms/matIdGlobal1.txt'
_C.PATH.matcls_matIdG2_path = 'data/openrooms/matIdGlobal2.txt'
_C.PATH.torch_home_path = ''
_C.PATH.torch_home_local = '/home/ruizhu/Documents/Projects/indoorInverse/'
_C.PATH.torch_home_cluster = ['/ruidata/indoorInverse/']
_C.PATH.pretrained_path = ''
_C.PATH.pretrained_local = '/home/ruizhu/Documents/Projects/semanticInverse/pretrained'
_C.PATH.pretrained_cluster = ['/ruidata/indoorInverse/pretrained']
_C.PATH.models_ckpt_path = ''
_C.PATH.models_ckpt_local = '/home/ruizhu/Documents/Projects/semanticInverse/models_ckpt'
_C.PATH.models_ckpt_cluster = ['/ruidata/indoorInverse/models_ckpt', '', '']

# ===== debug

_C.DEBUG = CN()
_C.DEBUG.if_fast_BRDF_labels = True
_C.DEBUG.if_fast_light_labels = True
_C.DEBUG.if_fast_val_labels = False
_C.DEBUG.if_dump_anything = False
_C.DEBUG.if_dump_full_envmap = False
_C.DEBUG.if_test_real = False
_C.DEBUG.if_iiw = False
_C.DEBUG.if_nyud = False
_C.DEBUG.if_dump_shadow_renderer = False
_C.DEBUG.if_dump_perframe_BRDF = False

# ===== dataset

_C.DATASET = CN()
_C.DATASET.mini = False # load mini OR from SSD to enable faster dataloading for debugging purposes etc.
_C.DATASET.tmp = False # load tmp OR list from DATASET.dataset_list_tmp
_C.DATASET.first_scenes = -1 # laod first # of the entire dataset: train/val
_C.DATASET.dataset_name = 'openrooms'
_C.DATASET.dataset_path = ''
_C.DATASET.dataset_path_local = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms'
_C.DATASET.dataset_path_local_quarter = ''
_C.DATASET.dataset_path_cluster = ['/openroomsindept']
# _C.DATASET.dataset_path_binary = ''
_C.DATASET.dataset_path_local_fast_BRDF = '/ruidata/openrooms_raw_BRDF'

_C.DATASET.dataset_path_pickle = ''
_C.DATASET.dataset_path_pickle_local = '/home/ruizhu/Documents/Projects/DPTSSN/dataset/ORfull-perFramePickles-240x320-fullBRDF-semseg'
_C.DATASET.dataset_path_pickle_cluster = ['/ruidata/ORfull-perFramePickles-240x320-fullBRDF-semseg']

_C.DATASET.real_images_root_path = '/home/ruizhu/Documents/Projects/indoorInverse'
_C.DATASET.real_images_list_path = 'data/list_real_20.txt'

_C.DATASET.iiw_path = ''
_C.DATASET.iiw_path_local = '/ruidata/iiw-dataset/data'
_C.DATASET.iiw_path_cluster = ['', '', '']
_C.DATASET.iiw_list_path = 'data/iiw/list'

_C.DATASET.nyud_path = ''
_C.DATASET.nyud_path_local = '/data/ruizhu/NYU'
_C.DATASET.nyud_path_cluster = ['', '', '']
_C.DATASET.nyud_list_path = 'data/nyud/list'

_C.DATASET.dataset_path_test = ''
_C.DATASET.dataset_path_test_local = '/home/ruizhu/Documents/Projects/indoorInverse/dataset/openrooms_test'
_C.DATASET.dataset_path_test_cluster = ['/openroomsindept']
_C.DATASET.png_path = ''
_C.DATASET.png_path_local = '/data/ruizhu/OR-pngs'
_C.DATASET.png_path_mini_local = '/data/ruizhu/ORmini-pngs'
_C.DATASET.png_path_mini_cluster = ['']
_C.DATASET.png_path_cluster = ['/siggraphasia20dataset/pngs']

_C.DATASET.envmap_path = ''
_C.DATASET.envmap_path_local = '/home/ruizhu/Documents/data/EnvDataset/'
_C.DATASET.envmap_path_cluster = ['/siggraphasia20dataset/EnvDataset/']

_C.DATASET.dataset_list = ''
_C.DATASET.dataset_path_mini = ''
_C.DATASET.dataset_path_mini_local = '/data/ruizhu/openrooms_mini'
_C.DATASET.dataset_path_mini_cluster = ['']
_C.DATASET.dataset_list_mini = 'data/openrooms/list_ORmini/list'
_C.DATASET.dataset_if_save_space = True # e.g. only same one depth for main_xml, diffMat, diffLight
_C.DATASET.dataset_list_sequence = False # convert #idx of the val list into sequential inputs
_C.DATASET.dataset_list_sequence_idx = -1

_C.DATASET.num_workers = 8
_C.DATASET.if_val_dist = True
# _C.DATASET.if_no_gt_semantics = False
_C.DATASET.if_quarter = False

_C.DATASET.if_no_gt_BRDF = False
_C.DATASET.if_no_gt_light = False


# ===== data loading configs

_C.DATA = CN()
_C.DATA.if_load_png_not_hdr = True # load png as input image instead of hdr image
_C.DATA.if_augment_train = False
_C.DATA.im_height = 240
_C.DATA.im_width = 320
_C.DATA.im_height_ori = 240
_C.DATA.im_width_ori = 320
_C.DATA.load_brdf_gt = True
_C.DATA.load_light_gt = False
_C.DATA.load_semseg_gt = False
_C.DATA.load_masks = False
_C.DATA.data_read_list = ''
_C.DATA.data_read_list_allowed = ['al', 'no', 'de', 'ro', 'li']

_C.DATA.iiw = CN()
_C.DATA.iiw.im_height = 341
_C.DATA.iiw.im_width = 512
# _C.DATA.iiw.im_height_padded_to = 256
# _C.DATA.iiw.im_width_padded_to = 320

_C.DATA.nyud = CN()
_C.DATA.nyud.im_height = 480
_C.DATA.nyud.im_width = 640
# _C.DATA.nyud.im_height_padded_to = 256
# _C.DATA.nyud.im_width_padded_to = 320

# ===== BRDF
_C.MODEL_BRDF = CN()
_C.MODEL_BRDF.enable = False
_C.MODEL_BRDF.if_bilateral = False
_C.MODEL_BRDF.if_bilateral_albedo_only = False
_C.MODEL_BRDF.if_freeze = False
_C.MODEL_BRDF.enable_list = '' # `al_no_de_ro`
_C.MODEL_BRDF.enable_list_allowed = ['al', 'no', 'de', 'ro']
_C.MODEL_BRDF.load_pretrained_pth = False
_C.MODEL_BRDF.loss_list = ''
_C.MODEL_BRDF.channel_multi = 1
_C.MODEL_BRDF.albedoWeight = 1.5
_C.MODEL_BRDF.normalWeight = 1.0
_C.MODEL_BRDF.roughWeight = 0.5
_C.MODEL_BRDF.depthWeight = 0.5
_C.MODEL_BRDF.if_debug_arch = False
_C.MODEL_BRDF.enable_BRDF_decoders = False

_C.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0 = 'check_cascade0_w320_h240/%s0_13.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits

_C.MODEL_BRDF.pretrained_pth_name_Bs_cascade0 = 'checkBs_cascade0_w320_h240/%s0_14_1000.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
_C.MODEL_BRDF.pretrained_if_load_encoder = True
_C.MODEL_BRDF.pretrained_if_load_decoder = True
_C.MODEL_BRDF.pretrained_if_load_Bs = False

_C.MODEL_BRDF.encoder_exclude = '' # e.g. 'x4_x5
_C.MODEL_BRDF.use_scale_aware_albedo = False # [default: False] set to False to use **scale-invariant** loss for albedo

_C.MODEL_BRDF.albedo = CN()
_C.MODEL_BRDF.albedo.if_HDR = False # compute albedo in SDR instead of pseudo-HDR

_C.MODEL_BRDF.loss = CN()
_C.MODEL_BRDF.loss.if_use_reg_loss_depth = False
_C.MODEL_BRDF.loss.reg_loss_depth_weight = 0.5
_C.MODEL_BRDF.loss.if_use_reg_loss_albedo = False
_C.MODEL_BRDF.loss.reg_loss_albedo_weight = 0.5

_C.MODEL_BRDF.use_scale_aware_depth = False
_C.MODEL_BRDF.depth_activation = 'tanh'
_C.MODEL_BRDF.loss.depth = CN() # ONLY works for MODEL_ALL (DPT) for now
_C.MODEL_BRDF.loss.depth.if_use_paper_loss = False # log(depth+0.001) instead of log(depth+1)
_C.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss = True # tanh + loss on depth

# ===== per-pixel lighting
_C.MODEL_LIGHT = CN()
_C.MODEL_LIGHT.enable = False
_C.MODEL_LIGHT.if_freeze = False
_C.MODEL_LIGHT.envRow = 120
_C.MODEL_LIGHT.envCol = 160
_C.MODEL_LIGHT.envHeight = 8
_C.MODEL_LIGHT.envWidth = 16
_C.MODEL_LIGHT.SGNum = 12
_C.MODEL_LIGHT.envmapWidth = 1024
_C.MODEL_LIGHT.envmapHeight = 512
_C.MODEL_LIGHT.offset = 1. # 'the offset for log error'
_C.MODEL_LIGHT.use_GT_brdf = False
_C.MODEL_LIGHT.use_offline_brdf = False
_C.MODEL_LIGHT.use_GT_light_envmap = False
_C.MODEL_LIGHT.load_GT_light_sg = False
_C.MODEL_LIGHT.use_GT_light_sg = False
_C.MODEL_LIGHT.load_pretrained_MODEL_BRDF = False
_C.MODEL_LIGHT.load_pretrained_MODEL_LIGHT = False
_C.MODEL_LIGHT.freeze_BRDF_Net = False
_C.MODEL_LIGHT.pretrained_pth_name_cascade0 = 'check_cascadeLight0_sg12_offset1.0/%s0_9.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
_C.MODEL_LIGHT.use_scale_aware_loss = False
_C.MODEL_LIGHT.if_transform_to_LightNet_coords = False # if transform pred lighting to global LightNet coords
_C.MODEL_LIGHT.enable_list = 'axis_lamb_weight'
_C.MODEL_LIGHT.if_align_log_envmap = True # instead of align raw envmap_pred and envmap_gt
_C.MODEL_LIGHT.if_align_rerendering_envmap = False
_C.MODEL_LIGHT.if_clamp_coeff = True
_C.MODEL_LIGHT.depth_thres = 50.
_C.MODEL_LIGHT.if_image_only_input = False
_C.MODEL_LIGHT.if_est_log_weight = False

# ===== solver

_C.SOLVER = CN()
_C.SOLVER.method = 'adam'
_C.SOLVER.lr = 1e-4
_C.SOLVER.if_warm_up = False
_C.SOLVER.weight_decay = 0.00001
_C.SOLVER.max_iter = 10000000
_C.SOLVER.max_epoch = 1000
_C.SOLVER.ims_per_batch = 16
_C.SOLVER.if_test_dataloader = False


_C.TRAINING = CN()
_C.TRAINING.MAX_CKPT_KEEP = 5

_C.TEST = CN()
_C.TEST.ims_per_batch = 16
_C.TEST.vis_max_samples = 20

_C.seed = 123
_C.resume_dir = None
_C.print_interval = 10
_C.flush_secs = 10