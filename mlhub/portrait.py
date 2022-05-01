import click
import os
import warnings
import filetype
from src.rembg.bg import params_default_dict, load_config, get_model
from src.rembg.bg import portrait as p, video_portrait as vp
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


@click.command()
@click.argument("input", type=click.Path())
@click.option('--output',
              '-o',
              type=str,
              default=params_default_dict["portrait"]["output"],
              help='Path to the output file')
@click.option('--model',
              '-m',
              type=str,
              default=params_default_dict["portrait"]["model"],
              help='The model name')
@click.option('--composite',
              '-c',
              default=params_default_dict["portrait"]["composite"],
              help='Generate the composition of portrait and original photo')
@click.option('--composite-sigma',
              '-cs',
              type=float,
              default=params_default_dict["portrait"]["composite_sigma"],
              help='Sigma value used for Gaussian filters when compositing.')
@click.option('--composite-alpha',
              '-ca',
              type=float,
              default=params_default_dict["portrait"]["composite_alpha"],
              help='Alpha value used for Gaussian filters when compositing.')
def portrait(input, output, model, composite, composite_sigma, composite_alpha):
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
        f = Image.open(input_path).convert("RGB")
        result = p(
            f,
            input_model=model,
            composite=composite,
            sigma=composite_sigma,
            alpha=composite_alpha
        )
        plt.axis('off')
        plt.imshow(result)

        if output is None:
            output_path, output_file = os.path.split(input_path)
            output_file = output_file.split('.')
            plt.savefig(os.path.join(output_path, output_file[0] + '_portrait.jpg'))
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
            flag = vp(input_path, output_path, input_model=model)

    else:
        raise FileNotFoundError("The input " + input_path + " is not a valid path to a image file")


def batch(input_folder, config_path):
    method, config = load_config(config_path)
    if method != "cutout":
        raise Exception("The method indicated in the file is not 'cutout', please check your config file.")
    for root, _, files in os.walk(input_folder):
        for file in files:
            model = get_model(config['model'])
            portrait(file,
                     output=config["output"],
                     model=model,
                     composite=config["composite"],
                     composite_sigma=config["composite_sigma"],
                     composite_alpha=config["composite_alpha"])
    print("Batch process completed")
    return True


if __name__ == '__main__':
    portrait()
