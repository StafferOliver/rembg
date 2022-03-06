import click
import filetype
import glob
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import yaml

from src.rembg.bg import remove, alpha_layer_remove, video_remove
from src.rembg.bg import portrait as p, video_portrait as vp

from portrait import portrait as portrait_CLI
from cutout import cutout as cutout_CLI
from PIL import Image, ImageFile

from mlhub.utils import get_package_dir
from mlhub.pkg import get_cmd_cwd

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

parameters_dict = {
    "cutout": ["input", "output", "model", "compare", "alpha_matting",
               "alpha_matting_foreground_threshold",
               "alpha_matting_background_threshold",
               "alpha_matting_erode_size",
               "alpha_matting_base_size"],
    "portrait": ["input", "output", "composite", "composite_sigma", "composite_alpha"]
}


def load_config(config_path):
    f = open(config_path, 'r')
    saved_config = yaml.load(f)
    method = saved_config["method"]
    config = {}
    if method == "cutout":
        default_values = getattr(cutout_CLI, "func_defaults")[1:]     # Reflections
    elif method == "portrait":
        default_values = getattr(portrait_CLI, "func_defaults")[1:]
    else:
        raise RuntimeError("The method is undefined!")
    for i in range(len(parameters_dict[method])):
        if saved_config[parameters_dict[method][i]] is None:
            config[parameters_dict[method][i]] = default_values[i]
        else:
            config[parameters_dict[method][i]] = saved_config[parameters_dict[method][i]]
    return method, config


def batch(input_folder, config_path):
    # TODO: Refactoring to load the model only once
    method, config = load_config(config_path)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if method == "cutout":
                cutout_CLI(os.path.join(root, file), **config)
            elif method == "portrait":
                portrait_CLI(os.path.join(root, file), **config)
    print("Batch process completed")
    return True



# def save_config():
#     pass
