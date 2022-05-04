import click
import numpy as np
import io
import glob
import os
import warnings
import filetype
from src.rembg.bg import remove, get_model, alpha_layer_remove, video_remove, params_default_dict, load_config
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from mlhub.utils import get_package_dir
from mlhub.pkg import get_cmd_cwd
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

model_path = os.environ.get(
    "U2NET_PATH",
    os.path.expanduser(os.path.join(get_package_dir(), "model")),
)
model_choices = [os.path.splitext(os.path.basename(x))[0] for x in set(glob.glob(model_path + "/*"))]
if len(model_choices) == 0:
    model_choices = ["u2net", "u2netp", "u2net_human_seg"]


@click.command()
@click.argument("input", type=click.Path())
@click.option('--output',
              '-o',
              type=str,
              default=params_default_dict['cutout']['output'],
              help='Path to the output file')
@click.option('--model',
              '-m',
              type=click.Choice(model_choices),
              default=params_default_dict['cutout']['model'],
              help='The model name')
@click.option('--compare',
              '-c',
              is_flag=bool(params_default_dict['cutout']['compare']),
              help="Display both original and result picture")
@click.option('--alpha-matting',
              '-a',
              is_flag=bool(params_default_dict['cutout']['alpha_matting']),
              help='When true use alpha matting cutout')
@click.option('--alpha-matting-foreground-threshold',
              '-af',
              type=int,
              default=int(params_default_dict['cutout']['alpha_matting_foreground_threshold']),
              help='The trimap foreground threshold')
@click.option('--alpha-matting-background-threshold',
              '-ab',
              type=int,
              default=int(params_default_dict['cutout']['alpha_matting_background_threshold']),
              help='The trimap background threshold')
@click.option('--alpha-matting-erode-size',
              '-ae',
              type=int,
              default=int(params_default_dict['cutout']['alpha_matting_erode_size']),
              help='Size of element used for the erosion')
@click.option('--alpha-matting-base-size',
              '-ab',
              type=int,
              default=int(params_default_dict['cutout']['alpha_matting_base_size']),
              help='The image base size')
def cutout(input, output, model, compare, alpha_matting,
           alpha_matting_foreground_threshold,
           alpha_matting_background_threshold,
           alpha_matting_erode_size,
           alpha_matting_base_size):
    if os.path.isabs(input):
        input_path = input
    else:
        input_path = os.path.join(get_cmd_cwd(), input)

    if output is not None:
        if os.path.isabs(output):
            output_path = output
        else:
            output_path = os.path.join(get_cmd_cwd(), output)

    if os.path.exists(input_path) \
       and filetype.guess(input_path).mime.find('image') >= 0:
        f = np.fromfile(input_path)
        result = remove(
                f,
                input_model=model,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=alpha_matting_erode_size,
                alpha_matting_base_size=alpha_matting_base_size
            )

        if filetype.guess(input_path).mime.find('jpeg') >= 0:
            result = alpha_layer_remove(np.array(result))

        if compare:
            f = Image.open(io.BytesIO(f)).convert("RGBA")
            fig, plot = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
            plot[0].imshow(f)
            plot[0].set_title('Original Image')
            plot[0].axis('off')
            plot[1].imshow(result)
            plot[1].set_title('Removal Result')
            plot[1].axis('off')
            fig.suptitle('Removal Result')
        else:
            plt.axis('off')
            plt.imshow(result)

        if output is None:
            output_path, output_file = os.path.split(input_path)
            output_file = output_file.split('.')
            if filetype.guess(input_path).mime.find('jpeg') >= 0:
                plt.savefig(os.path.join(output_path, output_file[0]+'_out.jpg'))
            else:
                plt.savefig(os.path.join(output_path, output_file[0]+'_out.png'))
        else:
            output_dir, _ = os.path.split(output_path)
            if output_dir != '' and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(output_path)

    elif os.path.exists(input_path) \
         and filetype.guess(input_path).mime.find('video') >= 0:
        if not os.path.exists(os.path.split(output_path)[0]):
            raise FileNotFoundError("You have to specific a valid output path for a video input")
        else:
            flag = video_remove(input_path, output_path, input_model=model)

    else:
        raise FileNotFoundError("The input " + input_path + " is not a valid path to a image file")


if __name__ == '__main__':
    cutout()
