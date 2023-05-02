"""
preprocessing main:
* slicing large images into smaller ones (patches)
    e.g. of size 512
* 'cropping' or up-sampling with interleave stride
    e.g. of stride 4
"""
from pathlib import Path

import cv2
from loguru import logger

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tqdm import tqdm

from config import PATCHES_TEMPLATE, CROPPED_TEMPLATE, BASELINE_TEMPLATE, STRIDE, PATCH_SIZE


def patchify(input_dir, output_dir, patch_size: int = 512, drop_empty: bool = True):
    input_images = input_dir.glob("*.jpg")
    output_dir.mkdir(exist_ok=True, parents=True)

    log_empties = 0
    for image_path in tqdm(input_images, desc="patchify"):
        image = io.imread(image_path)

        # add (0,0,0) padding
        updated_size = (((image.shape[0] // patch_size) + 1) * patch_size - image.shape[0]) / 2
        top, bottom, left, right = [int(updated_size)] * 4
        image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        for row in range(0, image_with_border.shape[0], patch_size):
            for column in range(0, image_with_border.shape[1], patch_size):
                out_path = output_dir / f"{image_path.stem}_patch_{row}_{column}.jpg"
                write_image = image_with_border[row:row + patch_size, column:column + patch_size, :]
                if drop_empty and np.unique(write_image).shape[0] == 1:
                    log_empties += 1
                    continue
                if (write_image.shape[0] != patch_size) or (write_image.shape[1] != patch_size):
                    log_empties += 1
                    continue
                cv2.imwrite(
                    out_path.as_posix(),
                    np.flip(write_image, 2),  # (rgb) to (bgr)
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
    if log_empties > 0:
        logger.warning(f"Skipping n empty images: {log_empties}")


# %%
def crop_image(image: np.array, interpolation: bool, channels: int = None, stride: int = 4, ):
    """ either with linear interpolation (serves as a baseline)
     or zero values inbetween strides (the training data later on)
     """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if not channels:
        channels = image.shape[-1]
    # interpolation = cv2.INTER_LINEAR if interpolation else "naha"

    # with linear interpolation inbetween
    if interpolation:
        new_size = (image.shape[1] // stride, image.shape[0] // stride, channels)
        new_image = np.zeros((new_size[0], new_size[1], channels), dtype=image.dtype)
        for i in range(new_size[1]):
            for j in range(new_size[0]):
                new_image[i, j] = image[i * stride, j * stride, :]
        upscale_img = cv2.resize(new_image, (image.shape[1], image.shape[0]), interpolation=interpolation)
        # no-data-values inbetween
    else:
        new_size = (image.shape[1], image.shape[0], channels)
        upscale_img = np.zeros((image.shape[0], image.shape[1], channels), dtype=image.dtype)
        for i in range(new_size[1]):
            for j in range(new_size[0]):
                if i % stride == 0 or j % stride == 0:
                    upscale_img[i, j] = image[i, j, :]
    return upscale_img


def get_cropped_path(image_path, output_dir, stride):
    return output_dir / f"{image_path.stem}_cropped_s{stride}.jpg"


def crop_io(image_path, output_dir, stride: int, **crop_args):
    _image = io.imread(image_path)
    _cropped_image = crop_image(_image, stride=stride, **crop_args)
    crop_image_path = get_cropped_path(image_path, output_dir, stride)
    if not crop_image_path.exists():
        return io.imsave(crop_image_path, _cropped_image, quality=100, check_contrast=False)


def preprocess(input_dir, output_dir, stride, patch_size):
    """todo make this main() and add arg parser:
        - input_fp
        - output_fp
        - stride
        - patch_size
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    patches_dir = output_dir / PATCHES_TEMPLATE.format(patch_size)
    cropped_dir = output_dir / CROPPED_TEMPLATE.format(patch_size, stride)
    baseline_dir = output_dir / BASELINE_TEMPLATE.format(patch_size, stride)

    [path.mkdir(parents=True, exist_ok=True) for path in [patches_dir, output_dir, cropped_dir, baseline_dir]]

    patchify(input_dir, patches_dir, patch_size=patch_size)

    patches_files = patches_dir.glob("*.jpg")
    for image_fp in tqdm(patches_files, desc=f"cropping images with {stride=}"):
        crop_io(image_fp, output_dir=cropped_dir, stride=stride, interpolation=False)
        crop_io(image_fp, output_dir=baseline_dir, stride=stride, interpolation=cv2.INTER_LINEAR)


def main():
    preprocess(
        input_dir="./test",
        output_dir="./data/",
        stride=STRIDE,
        patch_size=PATCH_SIZE
    )


if __name__ == '__main__':
    main()
