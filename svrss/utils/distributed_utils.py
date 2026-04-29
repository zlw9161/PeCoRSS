from collections import defaultdict, deque
import datetime
import pickle
import time
import errno
import os
import argparse
import torch
import torch.distributed as dist

# os.environ['MASTER_ADDR'] = '192.168.8.119'
# os.environ['MASTER_PORT'] = '4164'
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '2'

# def all_gather(data):
#     """
#     Run all_gather on arbitrary picklable data (not necessarily tensors)
#     Args:
#         data: any picklable object
#     Returns:
#         list[data]: list of data gathered from each rank
#     """
#     world_size = get_world_size()
#     if world_size == 1:
#         return [data]

#     # serialized to a Tensor
#     buffer = pickle.dumps(data)
#     storage = torch.ByteStorage.from_buffer(buffer)
#     tensor = torch.ByteTensor(storage).to("cuda")

#     # obtain Tensor size of each rank
#     local_size = torch.tensor([tensor.numel()], device="cuda")
#     size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
#     dist.all_gather(size_list, local_size)
#     size_list = [int(size.item()) for size in size_list]
#     max_size = max(size_list)

#     # receiving Tensor from all ranks
#     # we pad the tensor because torch all_gather does not support
#     # gathering tensors of different shapes
#     tensor_list = []
#     for _ in size_list:
#         tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
#     if local_size != max_size:
#         padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
#         tensor = torch.cat((tensor, padding), dim=0)
#     dist.all_gather(tensor_list, tensor)

#     data_list = []
#     for size, tensor in zip(size_list, tensor_list):
#         buffer = tensor.cpu().numpy().tobytes()[:size]
#         data_list.append(pickle.loads(buffer))

#     return data_list

def all_gather(input_value):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_value
    with torch.no_grad():  # 多GPU的情况
        output_value = [torch.zeros(size=input_value.shape, dtype=input_value.dtype) for _ in range(world_size)]
        dist.all_gather(output_value, input_value)
    return torch.concat(output_value, dim=0)
        
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_dict
    with torch.no_grad():  # 多GPU的情况
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

def reduce_value(input_value, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_value
    with torch.no_grad():  # 多GPU的情况
        if isinstance(input_value, list):
            output_value = torch.stack(input_value, dim=0)
        else:
            output_value = input_value
        dist.all_reduce(output_value)
        if average:
            output_value = output_value / world_size
        
        if isinstance(input_value, list):
            return [v for v in output_value]
        else:
            return output_value

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        #os.environ['MASTER_ADDR'] = '192.168.8.119'
        # print(os.environ['MASTER_ADDR'])
        args.rank = int(os.environ["RANK"])
        # args.rank = 1
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'gloo'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=2, rank=0)
    # 使用torch1.9或以上时建议加上device_ids=[args.rank]
    torch.distributed.barrier()
    # 仅让主进程进行print
    setup_for_distributed(args.rank == 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    args = parser.parse_args()
    #print(os.environ['LOCAL_RANK'])

    init_distributed_mode(args)
