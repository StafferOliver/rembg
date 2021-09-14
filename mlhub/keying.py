import argparse
import numpy as np
import io
import glob
import os
from distutils.util import strtobool
import filetype
from tqdm import tqdm
from src.rembg.bg import remove
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from mlhub.utils import get_package_dir
from mlhub.pkg import get_cmd_cwd
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_path = os.environ.get(
    "U2NET_PATH",
    os.path.expanduser(os.path.join(get_package_dir(), "model")),
)
model_choices = [os.path.splitext(os.path.basename(x))[0] for x in set(glob.glob(model_path + "/*"))]
if len(model_choices) == 0:
    model_choices = ["u2net", "u2netp", "u2net_human_seg"]

ap = argparse.ArgumentParser()

ap.add_argument(
    "input",
    nargs="?",
    default="-",
    type=str,
    help="Path to the input image.",
)

ap.add_argument(
    "-o",
    "--output",
    nargs="?",
    default="-",
    type=str,
    help="Path to the output png image.",
)

ap.add_argument(
    "-m",
    "--model",
    default="u2net",
    type=str,
    choices=model_choices,
    help="The model name.",
)

ap.add_argument(
    '-c',
    '--compare',
    action='store_true',
    help="Display both original and result picture"
)

ap.add_argument(
    "-a",
    "--alpha-matting",
    nargs="?",
    const=True,
    default=False,
    type=lambda x: bool(strtobool(x)),
    help="When true use alpha matting cutout.",
)

ap.add_argument(
    "-af",
    "--alpha-matting-foreground-threshold",
    default=240,
    type=int,
    help="The trimap foreground threshold.",
)

ap.add_argument(
    "-ab",
    "--alpha-matting-background-threshold",
    default=10,
    type=int,
    help="The trimap background threshold.",
)

ap.add_argument(
    "-ae",
    "--alpha-matting-erode-size",
    default=10,
    type=int,
    help="Size of element used for the erosion.",
)

ap.add_argument(
    "-az",
    "--alpha-matting-base-size",
    default=1000,
    type=int,
    help="The image base size.",
)

args = ap.parse_args()

if os.path.isabs(args.input):
    input_path = args.input
else:
    input_path = os.path.join(get_cmd_cwd(), args.input)

if os.path.isabs(args.output):
    output_path = args.output
else:
    output_path = os.path.join(get_cmd_cwd(), args.output)

if os.path.exists(input_path) and filetype.guess(input_path).mime.find('image') >= 0:
    f = np.fromfile(input_path)
    result = remove(
            f,
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=args.alpha_matting_background_threshold,
            alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
            alpha_matting_base_size=args.alpha_matting_base_size,
        )

    if args.compare:
        f = Image.open(io.BytesIO(f)).convert("RGBA")
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        fig, plot = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        plot[0].imshow(f)
        plot[0].set_title('Original Image')
        plot[0].axis('off')
        plot[1].imshow(img)
        plot[1].set_title('Removal Result')
        plot[1].axis('off')
        fig.suptitle('Removal Result')
    else:
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        plt.imshow(img)

    if args.output is None:
        plt.show()
    else:
        output_dir, _ = os.path.split(output_path)
        if output_dir != '' and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path)
else:
    raise FileNotFoundError("The input " + input_path + " is not a valid path to a image file")

