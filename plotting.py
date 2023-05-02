from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import rc, pyplot as plt
from skimage import io
from tqdm import tqdm

from data import Data
from evaluate import compare_images
from srgan import UNet

# rc("font",**{"family":"sans-serif","sans-serif":["Helvetica"]})
rc("font", **{"size": 14, "family": "serif", "serif": ["Times"]})
rc("text", usetex=True)


def release_tensor(tensor):
    return tensor.cpu().detach().numpy().squeeze().transpose(1, 2, 0)


def to_int(image):
    return np.ceil(image * 255).astype(int)


set_size = 20  # int(1920 / 4)
plot = True
device = "cpu"

data_set = Data(data_dir="./data", n_images=set_size, load_baseline=True)

srgan_model_path = "/Users/roetzermatthias/Documents/studies/semester1/02506aia/super_resolution/runs/unet_11220105_weekender_hiccups/model.pt"
unet_model_path = "/Users/roetzermatthias/Documents/studies/semester1/02506aia/super_resolution/runs/unet_21152704_mse_output_slink/model.pt"

generator = UNet(n_filters=8).to(device)
unet = UNet(n_filters=32).to(device)
peek_index = 6
generator.load_state_dict(torch.load(srgan_model_path, map_location=torch.device(device)))
unet.load_state_dict(torch.load(unet_model_path, map_location=torch.device(device)))
means = pd.DataFrame(0, columns=["ssim", "mse", "psnr"], index=["baseline", "Unet", "SRGAN"], )
for peek_index in tqdm(range(len(data_set))):

    input_image, target_image, baseline_image = data_set[peek_index]
    srgan_logits = generator(input_image.unsqueeze(0).to(device))
    unet_logits = unet(input_image.unsqueeze(0).to(device))

    srgan_reconstruction = to_int(torch.nn.functional.tanh(
        torch.nn.functional.relu(srgan_logits)
    ).cpu().detach().numpy().squeeze().transpose(1, 2, 0))
    unet_reconstruction = to_int(torch.nn.functional.tanh(
        torch.nn.functional.relu(unet_logits)
    ).cpu().detach().numpy().squeeze().transpose(1, 2, 0))

    baseline_image = to_int(release_tensor(baseline_image[:, 20:-20, 20:-20]))
    input_display = to_int(release_tensor(input_image))
    target_display = to_int(release_tensor(target_image))

    assert target_display.shape == baseline_image.shape
    assert srgan_reconstruction.shape == baseline_image.shape

    gan_ssim, gan_mse, gan_psnr = compare_images(target_display, srgan_reconstruction)
    means.loc["SRGAN"] += gan_ssim, gan_mse, gan_psnr
    unet_ssim, unet_mse, unet_psnr = compare_images(target_display, unet_reconstruction)
    means.loc["Unet"] += unet_ssim, unet_mse, unet_psnr
    base_ssim, base_mse, base_psnr = compare_images(target_display, baseline_image)
    means.loc["baseline"] += base_ssim, base_mse, base_psnr
    # 1 input 2 reconstruction 3 baseline 4 target
    if plot:
        images = [input_display, unet_reconstruction, srgan_reconstruction, baseline_image, target_display]
        titles = [
            "Low-resolution input\nStride:4",
            f"UNet Reconstruction\nSSIM: {unet_ssim:.2f} | PSNR: {unet_psnr:.2f}",
            f"SRGAN Reconstruction\nSSIM: {gan_ssim:.2f} | PSNR: {gan_psnr:.2f}",
            f"Lin.Interpolation Baseline\nSSIM: {base_ssim:.2f} | PSNR: {base_psnr:.2f}",
            "High-resolution Ground Truth\n",
        ]
        fig, ax = plt.subplots(2, len(images), figsize=(15, 8))
        for i, (image, title) in enumerate(zip(images, titles)):
            ax[0, i].imshow(image[100:200, 100:200, :])
            ax[0, i].set_title(title)
            ax[0, i].axes.get_xaxis().set_ticks([])
            ax[0, i].axes.get_yaxis().set_ticks([])

            ax[1, i].imshow(image[:, :, :])
            ax[1, i].axes.get_xaxis().set_ticks([])
            ax[1, i].axes.get_yaxis().set_ticks([])
        plt.tight_layout()
        Path("plots").mkdir(exist_ok=True)
        # plt.show()
        plt.savefig(Path("plots") / f"evaluation_{peek_index}.jpg", dpi=300)
means /= set_size
np.round(means, 2)
print(means)

fig, ax = plt.subplots(2, 1, figsize=(5, 10))
ax[0].imshow(input_display[100:200, 100:200, :])
ax[0].set_title("Low-resolution input\nStride:4")
ax[1].imshow(input_display[:, :, :])

ax[0].axes.get_xaxis().set_ticks([])
ax[0].axes.get_yaxis().set_ticks([])

ax[1].axes.get_xaxis().set_ticks([])
ax[1].axes.get_yaxis().set_ticks([])
plt.tight_layout()
plt.show()

target_channel = target_display[:, :, 0].flatten()
srgan_channel = srgan_reconstruction[:, :, 0].flatten()
srgan_channel.shape == target_channel.shape

srgan_reconstruction = to_int(release_tensor(
    torch.nn.functional.relu(srgan_logits)
))

plt.hist(srgan_reconstruction[:, :, 0].flatten(), bins=255)
plt.hist(baseline_image[:, :, 0].flatten(), bins=255)
plt.hist(target_display[:, :, 0].flatten(), bins=255)
plt.show()

difference = target_display - srgan_reconstruction
difference = difference / np.var(difference, axis=(0, 1)) - np.mean(difference, axis=(0, 1))
plt.imshow(difference[:, :, 0], cmap="jet")
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(5, 10))
ax[0].imshow(difference[100:200, 100:200, 0], cmap='RdBu_r')
ax[0].set_title("Low-resolution input\nStride:4")
ax[1].imshow(difference[:, :, 0], cmap='RdBu_r')

ax[0].axes.get_xaxis().set_ticks([])
ax[0].axes.get_yaxis().set_ticks([])

ax[1].axes.get_xaxis().set_ticks([])
ax[1].axes.get_yaxis().set_ticks([])
plt.tight_layout()
plt.show()

plt.hist(difference[:, :, 0].flatten(), bins=255)
plt.show()

test = io.imread("/Users/roetzermatthias/Documents/DSC05080.jpeg")

targ_mean = np.mean(target_display, axis=(0, 1))
targ_var = np.var(target_display, axis=(0, 1))

mean = np.mean(test, axis=(0, 1))
var = np.var(test, axis=(0, 1))
