import click
import os
from src.rembg.bg import load_config, get_model, save_config
from cutout import cutout
from portrait import portrait


@click.command()
@click.argument("input_folder", type=click.Path())
@click.argument("config_path", type=click.Path())
@click.option('--setup',
              '-s',
              is_flag=False,
              help='Whether setting up the config or not')
def batch(input_folder, config_path, setup):
    if setup:
        setup_path, _, _ = save_config()
    if setup_path == config_path:
        method, config = load_config(config_path)
    else:
        raise FileNotFoundError("The setup path and given path the config are not the same.")
    if method == "cutout":
        for root, _, files in os.walk(input_folder):
            model = get_model(config['model'])
            for file in files:
                cutout(file,
                       output=config['output'],
                       model=model,
                       compare=config['compare'],
                       alpha_matting=config['alpha_matting'],
                       alpha_matting_foreground_threshold=config["alpha_matting_foreground_threshold"],
                       alpha_matting_background_threshold=config["alpha_matting_background_threshold"],
                       alpha_matting_erode_structure_size=config["alpha_matting_erode_size"],
                       alpha_matting_base_size=config["alpha_matting_base_size"])
    elif method == "portrait":
        for root, _, files in os.walk(input_folder):
            model = get_model(config['model'])
            for file in files:
                portrait(file,
                         output=config["output"],
                         model=model,
                         composite=config["composite"],
                         composite_sigma=config["composite_sigma"],
                         composite_alpha=config["composite_alpha"])
    else:
        raise Exception("The method indicated in the file is not 'cutout' nor 'portrait'. \n"
                        "please check your config file.")
    print("Batch process completed")
    return True
