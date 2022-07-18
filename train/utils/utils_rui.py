import torch.distributed as dist

def gather_lists(list0, num_gpus, process_group=None):
    list0_allgather = [None for _ in range(num_gpus)]
    if process_group is None:
        # print('======', list0[:10],len(list0))
        dist.all_gather_object(list0_allgather, list0)
        # print('======<')
    else:
        dist.all_gather_object(list0_allgather, list0, group=process_group)
    # print(len(list0_allgather), len(list0_allgather[0]), '<<<<<<<<<<-------', opt.rank)
    list0_allgather = [item for sublist in list0_allgather for item in sublist]
    return list0_allgather
