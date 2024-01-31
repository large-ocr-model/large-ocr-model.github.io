from pathlib import PurePath
from typing import Sequence

import torch
from torch import nn

import yaml
from torchvision.ops.misc import FrozenBatchNorm2d


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
    'abinet': 'https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
    'trba': 'https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
    'vitstr': 'https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
    'crnn': 'https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
}


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parents[2]
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / f'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    return config


def _get_model_class(key):
    exp = 'parseq'
    if 'abinet' in key:
        from .abinet.system import ABINet as ModelClass
        exp = 'abinet'
    elif 'crnn' in key:
        from .crnn.system import CRNN as ModelClass
        exp = 'crnn'
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
        exp = 'parseq'
    elif 'trba' in key:
        from .trba.system import TRBA as ModelClass
    elif 'trbc' in key:
        from .trba.system import TRBC as ModelClass
    elif 'vitstr' in key:
        from .vitstr.system import ViTSTR as ModelClass
        exp = 'vitstr'
    elif 'vl4str' in key or 'clip4str' in key:
        from .vl_str.system import VL4STR as ModelClass
        exp = 'vl4str'
        if 'large' in key:
            exp = 'vl4str-large'
        if 'base32x32' in key:
            exp = 'vl4str-base32'

    elif 'str_adapter' in key:
        from .str_adapter.system import STRAdapter as ModelClass
        exp = 'str_adapter'
    else:
        raise InvalidModelError("Unable to find model class for '{}'".format(key))

    return ModelClass, exp


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError("No configuration found for '{}'".format(experiment)) from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        try:
            url = _WEIGHTS_URL[experiment]
        except KeyError:
            raise InvalidModelError("No pretrained weights found for '{}'".format(experiment)) from None
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    # if checkpoint_path.startswith('pretrained='):
    #     model_id = checkpoint_path.split('=', maxsplit=1)[1]
    #     model = create_model(model_id, True, **kwargs)
    # else:
    #     ModelClass, _ = _get_model_class(checkpoint_path)
    #     model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    try:
        ModelClass, _ = _get_model_class(checkpoint_path)
        model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    except:
        ModelClass, experiment = _get_model_class(checkpoint_path)
        try:
            config = _get_config(experiment, **kwargs)
        except FileNotFoundError:
            raise InvalidModelError("No configuration found for '{}'".format(experiment)) from None
        model = ModelClass(**config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)

    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def freeze_batch_norm_2d(module, module_match={}, name=""):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.
    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)
    Returns:
        torch.nn.Module: Resulting module
    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


if __name__ == "__main__":
    import numpy as np

    # define a sequence of 10 words over a vocab of 5 words
    data = [[0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1]]
    data = np.array(data)
    # # decode sequence
    # result = beam_search_decoder(data, 3)
    # # print result
    # for seq in result:
    #     print(seq)
