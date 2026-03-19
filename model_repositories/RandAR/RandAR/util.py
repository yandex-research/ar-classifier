import importlib
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ObjectParamConfig:
    target: str  # name of the Class
    params: dict  # parameters to pass to the Class


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def set_nested_key(data, keys, value):
    """Sets value in nested dictionary"""
    key = keys.pop(0)

    if keys:
        if key not in data:
            data[key] = {}
        set_nested_key(data[key], keys, value)
    else:
        data[key] = value_type(value)


def value_type(value):
    """Convert str to bool/int/float if possible"""
    try:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
    except AttributeError:
        return value


def load_safetensors(ckpt_path):
    tensors = dict()
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def save_model_safetensors(model, ckpt_path):
    tensors = {k: v for k, v in model.state_dict().items()}
    save_file(tensors, ckpt_path)
