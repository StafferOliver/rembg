import functools
import io
import numpy as np
import torch.nn as nn
import yaml
from PIL import Image
from skimage import transform
from skimage.filters import gaussian
from .u2net import detect
from pickle import UnpicklingError

params_default_dict = {
    "cutout": {"input": "required",
               "output": None,
               "model": "u2net",
               "compare": "True",
               "alpha_matting": "False",
               "alpha_matting_foreground_threshold": "240",
               "alpha_matting_background_threshold": "10",
               "alpha_matting_erode_size": "10",
               "alpha_matting_base_size": "1000"},
    "portrait": {"input": "required",
                 "output": None,
                 "model": "u2net_portrait",
                 "composite": True,
                 "composite_sigma": 2,
                 "composite_alpha": 0.5}
}

params_dict = {"cutout": params_default_dict["cutout"].keys(),
               "protrain": params_default_dict["portrait"].keys()}


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold=240,
    background_threshold=10,
    erode_structure_size=10,
    base_size=1000,
):
    try:
        from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
        from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
        from pymatting.util.util import stack_images
        from scipy.ndimage.morphology import binary_erosion
    except:
        print("PyMatting seems unavailable currently.\nCheck your environment or disable alpha-matting")
        return None
    else:
        size = img.size

        img.thumbnail((base_size, base_size), Image.LANCZOS)
        mask = mask.resize(img.size, Image.LANCZOS)

        img = np.asarray(img)
        mask = np.asarray(mask)

        # guess likely foreground/background
        is_foreground = mask > foreground_threshold
        is_background = mask < background_threshold

        # erode foreground/background
        structure = None
        if erode_structure_size > 0:
            structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        # build trimap
        # 0   = background
        # 128 = unknown
        # 255 = foreground
        trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0

        # build the cutout image
        img_normalized = img / 255.0
        trimap_normalized = trimap / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = stack_images(foreground, alpha)

        cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
        cutout = Image.fromarray(cutout)
        cutout = cutout.resize(size, Image.LANCZOS)

        return cutout


def naive_cutout(img, mask):
    img_shape = img.size if type(img) == Image.Image else img.shape[0:2]
    empty = Image.new("RGBA", img_shape, 0)
    cutout = Image.composite(img,
                             empty,
                             mask.resize(img_shape, Image.LANCZOS))
    return cutout


@functools.lru_cache(maxsize=None)
def get_model(model_name):
    if model_name == "u2netp":
        return_model = detect.load_model(model_name="u2netp")
    elif model_name == "u2net_human_seg":
        return_model = detect.load_model(model_name="u2net_human_seg")
    elif model_name == "u2net_portrait":
        return_model = detect.load_model(model_name="u2net_portrait")
    else:
        return_model = detect.load_model(model_name="u2net")
    return return_model


def remove(
    data,
    input_model="u2net",
    alpha_matting=False,
    *args, **kwargs
):
    """
    Background removal on a single image

    :param data: the input image (PIL.Image or NumPy.ndarray)
    :param input_model: the model used in the background removal
    :param alpha_matting: if True, alpha matting will be applied with the rest of parameters
    :return: return True if all the steps are correctly finished
    """

    # Initialize model
    if isinstance(input_model, str):
        model = get_model(input_model)
    elif isinstance(input_model, nn.Module):
        model = input_model
    else:
        raise TypeError("The expected type for the argument input_model is either string or nn.Module.")

    # Transform data
    if isinstance(data, np.ndarray):
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = data

    # Predicting mask using u2net
    mask = Image.fromarray(detect.predict(model, np.array(img)) * 255).convert("L")
    cutout = None

    # Generate output image
    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            *args, **kwargs
        )

    if cutout is None:
        cutout = naive_cutout(img, mask)

    # Save the image
    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    return Image.open(io.BytesIO(bio.getbuffer())).convert("RGBA")


def portrait(
    data,
    input_model='u2net_portrait',
    composite=False,
    sigma=2,
    alpha=0.5
):
    """
    Portrait generation with single given picture

    :param data: the input image (PIL.Image or NumPy.ndarray)
    :param input_model: the model used in the portrait generation
    :param composite: if True, the original portrait will be blurred and composite into the output
    :param sigma: the blur parameters used for Gaussian filter (Used when composite=True)
    :param alpha: the blur parameters used for Gaussian filter (Used when composite=True)
    :return: generated portrait (In PIL.Image)
    """

    # Initialize model
    if isinstance(input_model, str):
        model = get_model(input_model)
    elif isinstance(input_model, nn.Module):
        model = input_model
    else:
        raise TypeError("The expected type for the argument input_model is either string or nn.Module.")

    # Transform data
    if isinstance(data, np.ndarray):
        img = data
    else:
        img = np.array(data)

    # Size alert. If the picture is far too small, the result might be unexpected
    if img.shape[0] < 512 and img.shape[1] < 512:
        print("The size of the input picture is too small to generate. The result may be unexpected.")
        print("To obtain a good portrait, use a picture larger than 512*512 with a clear face.")

    # Predicting portrait using u2net
    output = detect.predict(model, img, True)

    # Generate output image
    if composite:
        output = transform.resize(output, img.shape[0:2], order=2)
        output = output / (np.amax(output) + 1e-8) * 255
        output = output[:, :, np.newaxis]
        img_blurred = gaussian(img, sigma=sigma, preserve_range=True)
        output = img_blurred * alpha + output * (1 - alpha)
        output = Image.fromarray(output.astype(np.uint8)).convert('RGB')
    else:
        output = Image.fromarray(output * 255).convert('RGB')
    return output


def alpha_layer_remove(input_image, bg_color=np.array([255, 255, 255])):
    """
    Remove alpha layer in a PNG (RGBA color format) file/array

    :param input_image: the input image (in PIL.Image or NumPy.ndarray)
    :param bg_color: the color used to fill the blank left by alpha layer, default color is white
    :return: processed picture(in NumPy.ndarray)
    """
    if isinstance(input_image, np.ndarray):
        img = input_image
    else:
        img = np.array(input_image)
    alpha = (img[:, :, 3] / 255).reshape(img.shape[:2] + (1,))
    output = bg_color * (1 - alpha) + (img[:, :, :3] * alpha)
    return output.astype(np.uint8)


def video_remove(
    input_path,
    output_path,
    input_model="u2net",
    alpha_matting=False,
    *args, **kwargs
):
    """
    Video background removal

    :param input_path: the path to the input video file
    :param output_path: the path to the output video file
    :param input_model: the model used in the background removal
    :param alpha_matting: if True, alpha matting will be applied with the rest of parameters
    :return: return True if all the steps are correctly finished
    """

    # Library detection
    try:
        import ffmpeg
    except ModuleNotFoundError:
        print("ffmpeg library is not currently installed, which is required for this functionality")
        print("Please run 'pip install opencv-python' and 'apt install ffmpeg' in command-line to install dependency")
        return False

    # Obtain basic video infos
    probe = ffmpeg.probe(input_path)
    width = probe['streams'][0]['width']
    height = probe['streams'][0]['height']
    frame_rate = probe['streams'][0]['avg_frame_rate']
    data = ffmpeg.input(input_path)

    # Extract frames
    video_frames = np.frombuffer(data
                                 .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                                 .run(quiet=True)[0], np.uint8).reshape([-1, height, width, 3])

    # Initailize output flow
    output = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
        .output(output_path, r=frame_rate).overwrite_output() \
        .run_async(pipe_stdin=True, quiet=True)

    # Load model
    if isinstance(input_model, str):
        model = get_model(input_model)
    elif isinstance(input_model, nn.Module):
        model = input_model
    else:
        raise TypeError("The expected type for the argument input_model is either string or nn.Module.")

    # Removal process
    for i in range(video_frames.shape[0]):
        img = Image.fromarray(video_frames[i, :, :, :]).convert('RGB')
        mask = Image.fromarray(detect.predict(model, np.array(img)) * 255).convert("L")
        cutout = None
        if alpha_matting:
            cutout = alpha_matting_cutout(
                img,
                mask,
                *args, **kwargs
            )
        if cutout is None:
            cutout = naive_cutout(img, mask)
        output.stdin.write(alpha_layer_remove(cutout).tobytes())

    # Write & close flow
    output.stdin.close()
    output.wait()
    return True


def video_portrait(
    input_path,
    output_path,
    input_model='u2net_portrait',
    composite=False,
    sigma=2,
    alpha=0.5):
    """
    Video portrait generation (Experimental)

    :param input_path: the path to the input video file
    :param output_path: the path to the output video file
    :param input_model: the model used in the portrait generation
    :param composite: if True, the original portrait will be blurred and composite into the output
    :param sigma: the blur parameters used for Gaussian filter (Used when composite=True)
    :param alpha: the blur parameters used for Gaussian filter (Used when composite=True)
    :return: return True if all the steps are correctly finished
    """

    # Library detection
    try:
        import ffmpeg
    except ModuleNotFoundError:
        print("ffmpeg library is not currently installed, which is required for this functionality")
        print("Please run 'pip install opencv-python' and 'apt install ffmpeg' in command-line to install dependency")
        return False

    # Obtain basic video infos
    probe = ffmpeg.probe(input_path)
    width = probe['streams'][0]['width']
    height = probe['streams'][0]['height']
    frame_rate = probe['streams'][0]['avg_frame_rate']
    data = ffmpeg.input(input_path)

    # Extract frames
    video_frames = np.frombuffer(data
                                 .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                                 .run(quiet=True)[0], np.uint8).reshape([-1, height, width, 3])

    # Initialize output flow
    output = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
        .output(output_path, r=frame_rate).overwrite_output() \
        .run_async(pipe_stdin=True, quiet=True)

    # Load model
    if isinstance(input_model, str):
        model = get_model(input_model)
    elif isinstance(input_model, nn.Module):
        model = input_model
    else:
        raise TypeError("The expected type for the argument input_model is either string or nn.Module.")

    # Portrait generation process
    for i in range(video_frames.shape[0]):
        img = Image.fromarray(video_frames[i, :, :, :]).convert('RGB')
        result = detect.predict(model, img, True)
        if composite:
            result = transform.resize(result, img.shape[0:2], order=2)
            result = result / (np.amax(result) + 1e-8) * 255
            result = result[:, :, np.newaxis]
            img_blurred = gaussian(img, sigma=sigma, preserve_range=True)
            result = img_blurred * alpha + result * (1 - alpha)
            result = result.astype(np.uint8)
        else:
            result = (result * 255).astype(np.uint8)
        output.stdin.write(result.tobytes())

    # Write & close flow
    output.stdin.close()
    output.wait()
    return True


def load_config(config_path):
    """
    Load saved configurations from existing YAML file

    :param config_path: The path to the input file
    :return: The first element of the tuple indicate the method is going to be used
             The second element indicate the detailed configs
    """
    f = open(config_path, 'r')
    saved_config = yaml.load(f)
    method = saved_config["method"]
    config = {}
    for i in range(len(params_dict[method])):
        if saved_config[params_dict[method][i]] is None:
            config[params_dict[method][i]] = params_default_dict[method].values()[i]
        else:
            config[params_dict[method][i]] = saved_config[params_dict[method][i]]
    return method, config


# Deprecated
def _save_config(config_path, method, config_list):
    """
    Save configurations to YAML file

    :param config_path: the path to the YAML file
    :param method: the method is going to be saved
    :param config_list: the detailed configurations
    """
    f = open(config_path, 'w')
    output_dict = {}
    for i in range(len(params_default_dict[method].keys())):
        if i == 0:
            output_dict[params_default_dict[method].keys()[i]] = "required"
        elif config_list[i] is None and params_default_dict[method].keys()[i] is not None:
            output_dict[params_default_dict[method].keys()[i]] = config_list[i]
        else:
            output_dict[params_default_dict[method].keys()[i]] = params_default_dict[method].values()[i]
    yaml.dump({method: output_dict}, f)


def save_config():
    method = input("Please specify the method (portrait/cutout) you want to setup: ")
    assert method in params_default_dict.keys()
    config_values = []
    for k,v in params_default_dict[method].items():
        custom_val = input("Please specify the value for {} or leave a blank for default value: ".format(k))
        if len(custom_val) == 0:
            config_values.append(v)
        else:
            config_values.append(custom_val)
    config_path = input("Please specify the path to the config file: ")
    _save_config(config_path, method, config_values)
    return True
