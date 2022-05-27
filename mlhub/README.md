# rembg

## Introduction
rembg is a Python library for image/video background removal or portrait generation, which is based on u2net as backbone network.

To use this library, extra pretrained model for u2net will be downloaded. If the models failed to download for several times due to the security policies of Google Drive, please refer to https://github.com/xuebinqin/U-2-Net for downloading.

This library is developed with MLHub library. Directly downloading zip files might not working.

See https://arxiv.org/pdf/2005.09007.pdf for paper or https://github.com/xuebinqin/U-2-Net for detailed model descriptions.

This repo is adapted from the [source library](https://github.com/danielgatis/rembg). Some unused codes and dependencies have been removed

## Usage
- To install MLHub on Ubuntu
```shell
$ pip3 install mlhub
$ ml configure
```

- To install rembg using MLHub
```shell
$ ml install rembg
```

- To run rembg demo using provided examples
```shell
$ ml demo rembg
```
Pretrained U-2-Net model will be downloaded on first use

- To perform removal on a custom image/video
```shell
$ ml cutout rembg PATH_TO_INPUT_FILE [-o PATH_TO_OUTPUT_FILE]
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path.

The output path is optional for image input. If unspecified, the output file will be generated in the input folder.

For a video input file, output path is required.

- To perform removal on a custom image with comparison to its original image
```shell
$ ml cutout rembg PATH_TO_INPUT_FILE -c
```
Replace PATH_TO_INPUT_FILE with your corresponding path

- To perform removal on a custom image using alpha-matting
```shell
$ ml cutout rembg PATH_TO_INPUT_FILE -o PATH_TO_OUTPUT_FILE -a True
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path

- To generate portrait on a given photo
```shell
$ ml portrait rembg PATH_TO_INPUT_FILE [-o PATH_TO_OUTPUT_FILE]
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path.

The output path is optional for image input. If unspecified, the output file will be generated in the input folder.

For a video input file, output path is required.

- Full usage of cutout.py
```shell
Usage: cutout.py [OPTIONS] INPUT

Positional Arguments:
  INPUT                 Path to the input file.

Optional Arguments:
  -o, --output TEXT               Path to the output file
  -m, --model [u2net|u2netp|u2net_human_seg]
                                  The model name
  -c, --compare                   Display both original and result picture
  -a, --alpha-matting TEXT        When true use alpha matting cutout
  -af, --alpha-matting-foreground-threshold INTEGER
                                  The trimap foreground threshold
  -ab, --alpha-matting-background-threshold INTEGER
                                  The trimap background threshold
  -ae, --alpha-matting-erode-size INTEGER
                                  Size of element used for the erosion
  -ab, --alpha-matting-base-size INTEGER
                                  The image base size
  --help                          Show this message and exit.
```

- Full usage of portrait.py
```shell
Usage: portrait.py [OPTIONS] INPUT

Positional Arguments:
  INPUT                 Path to the input file.

Optional Arguments:
  -o, --output TEXT             Path to the output file
  -c, --composite TEXT          Generate the composition of portrait and
                                original photo
  -cs, --composite-sigma FLOAT  Sigma value used for Gaussian filters when
                                compositing.
  -ca, --composite-alpha FLOAT  Alpha value used for Gaussian filters when
                                compositing.
  --help                        Show this message and exit.
```
