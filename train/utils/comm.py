"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time

import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data, if_cat_at_0 = False, if_same_device = True):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable objecqqt
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    if if_same_device and torch.is_tensor(data_list[0]):
        if data_list[0].is_cuda:
            device = data_list[0].get_device()
            data_list = [x.cuda(device) for x in data_list]

    if if_cat_at_0:
        return torch.cat(data_list)

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        print('[comm] world_size==%d; not reduced!'%world_size)
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, return_dict=False, only_gather=False):
    if _dict_to_list is None:
        return
    if get_world_size()==1:
        return predictions_per_gpu
    all_predictions = all_gather(predictions_per_gpu)
    if only_gather:
        return all_predictions
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if return_dict:
        return predictions

    return _dict_to_list(predictions)

def _dict_to_list(predictions):
    if predictions is None:
        return
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # if len(image_ids) != image_ids[-1] + 1:
    #     logger = logging.getLogger("maskrcnn_benchmark.inference")
    #     logger.warning(
    #         "Number of images that were gathered from multiple processes is not "
    #         "a contiguous set. Some images might be missing from the evaluation"
    #     )
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions