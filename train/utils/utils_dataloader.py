import torch
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import logging
from utils.maskrcnn_rui.utils.comm import get_world_size
from utils.maskrcnn_rui.data.build import make_data_sampler
from utils.maskrcnn_rui.data import samplers
from utils.utils_training import cycle
from utils.utils_misc import white_blue, basic_logger

def make_batch_data_sampler(
    dataset, sampler, images_per_batch, num_iters=None, start_iter=0, drop_last=True
):
    # if aspect_grouping:
    #     if not isinstance(aspect_grouping, (list, tuple)):
    #         aspect_grouping = [aspect_grouping]
    #     aspect_ratios = _compute_aspect_ratios(dataset)
    #     group_ids = _quantize(aspect_ratios, aspect_grouping)
    #     batch_sampler = samplers.GroupedBatchSampler(
    #         sampler, group_ids, images_per_batch, drop_uneven=False
    #     )
    # else:
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=drop_last
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_data_loader(opt, dataset, is_train=True, start_iter=0, is_for_period=False, logger=None, override_shuffle=None, collate_fn=None, batch_size_override=-1, workers=-1, pin_memory = True, if_distributed_override=True):
    cfg = opt.cfg
    num_gpus = opt.num_gpus
    if logger is None:
        logger = basic_logger()

    # print('==============', cfg.TEST.IMS_PER_BATCH, num_gpus, '=====')
    is_distributed=opt.distributed and if_distributed_override
    num_workers = cfg.DATASET.num_workers if workers==-1 else workers
    if is_train:
        # images_per_batch = cfg.SOLVER.IMS_PER_BATCH`
        # assert (
        #     images_per_batch % num_gpus == 0
        # ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
        #     images_per_batch, num_gpus)
        # images_per_gpu = images_per_batch // num_gpus if batch_size_override==-1 else batch_size_override
        images_per_gpu = cfg.SOLVER.ims_per_batch if batch_size_override==-1 else batch_size_override
        shuffle = True
        # num_iters = cfg.SOLVER.max_iter
        drop_last = False
        # num_workers = num_workers
        # num_workers = 4
    else:
        # images_per_batch = cfg.TEST.IMS_PER_BATCH
        # assert (
        #     images_per_batch % num_gpus == 0
        # ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
        #     images_per_batch, num_gpus)
        # images_per_gpu = images_per_batch // num_gpus if batch_size_override==-1 else batch_size_override
        images_per_gpu = cfg.TEST.ims_per_batch if batch_size_override==-1 else batch_size_override
        # shuffle = False if not is_distributed else True
        shuffle = False
        # num_iters = None
        start_iter = 0
        drop_last = False
        # num_workers = 0 if opt.AML else cfg.DATASET.num_workers
        # num_workers = num_workers

    if override_shuffle is not None:
        shuffle = override_shuffle
    sampler = make_data_sampler(dataset, shuffle, opt, distributed=is_distributed, if_distributed_override=if_distributed_override)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, images_per_gpu, None, start_iter, drop_last=drop_last
    )
    # collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
    #     BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    logger.info(white_blue('[utils_dataloader] %s-%s with bs %d*%d: len(dataset) %d, len(sampler) %d, len(batch_sampler) %d, len(data_loader) %d, is_train %s, is_distributed %s:' % \
                (dataset.dataset_name, dataset.split, images_per_gpu, num_gpus, len(dataset), len(sampler), len(batch_sampler), len(data_loader), is_train, is_distributed)))
    return data_loader, images_per_gpu

def iterator_coco_combine_alternate(iterator_A, iterator_B):
    flag = True
    # if len(iterator_A) > len(iterator_B):
    #     iterator_B = cycle(iterator_B)
    # else:
    #     iterator_A = cycle(iterator_A)
    iterator_A = cycle(iterator_A)
    iterator_B = cycle(iterator_B)
    iterator_A = iter(iterator_A)
    iterator_B = iter(iterator_B)

    # result = 0
    
    # while result is not None:
    while True:
        if flag:
            flag = not flag
            yield(next(iterator_A))
        else:
            flag = not flag
            yield(next(iterator_B))