import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def compare_images(image_a, image_b):
    ssim = structural_similarity(
        image_a, image_b,
        channel_axis=2,
        data_range=image_b.max() - image_b.min(),
        win_size=None
    )
    mse = mean_squared_error(image_a, image_b)
    psnr = peak_signal_noise_ratio(image_a, image_b)
    return ssim, mse, psnr


def compare_batches(input_batch: torch.Tensor, target_batch: torch.Tensor):
    """ returns all three similarity measures"""
    return np.mean(
        np.array([
            compare_images(
                image_a=input_batch[i].cpu().numpy().squeeze().transpose(1, 2, 0),
                image_b=target_batch[i].cpu().numpy().squeeze().transpose(1, 2, 0),
            )
            for i in range(input_batch.shape[0])
        ]), axis=0
    )


def plot_comparison(input_batch, target_batch):
    """ wip """
    fig, ax = plt.subplots(2, input_batch.shape[0])
    for i in range(input_batch.shape[0]):
        inp_im = input_batch[i].cpu().numpy().squeeze().transpose(1, 2, 0)
        targ_im = target_batch[i].cpu().numpy().squeeze().transpose(1, 2, 0)
        ssim, mse, psnr = compare_images(inp_im, targ_im)

        ax[0, i].imshow(inp_im)
        ax[0, i].set_title(f"ssim: {ssim=} {mse=} {psnr=}")

        ax[1, i].imshow(targ_im)
        plt.show()
