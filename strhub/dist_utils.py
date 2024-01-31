# coding=utf-8
import os
import logging
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_only


def is_dist_avail_and_initialized():
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


def setup_logger(log_file=None, include_host=False):
    """
    Ref: https://github.com/mlfoundations/open_clip/blob/db338b0bb36c15ae12fcd37e86120414903df1ef/src/training/logger.
    """
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    level = logging.INFO if is_main_process() else logging.WARN

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


@rank_zero_only
def copy_remote(local_dir, remote_dir=None):
    if remote_dir is not None:
        print('=' * 50)
        os.chdir(local_dir)
        os.chdir('..')
        base_name = os.path.basename(local_dir)
        tar_filename = '{}.tar'.format(base_name)
        if os.path.exists(os.path.join(os.getcwd(), tar_filename)):
            print('remove existing tarfile and create a new tarfile')
            os.system("rm {}".format(tar_filename))

        os.system("tar -zcvf {} {}".format(tar_filename, base_name))
        os.system("rsync -rvP {} {}".format(tar_filename, remote_dir))
        os.system("ls -lah {}".format(remote_dir))
        print("Copy success!")
