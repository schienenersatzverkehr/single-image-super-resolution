from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
import torch.nn.functional as F
from config import PATCHES_TEMPLATE, CROPPED_TEMPLATE, PATCH_SIZE, STRIDE, SAVE_DIR_TEMPLATE, BASELINE_TEMPLATE

from loguru import logger


class Data(torch.utils.data.Dataset):
    def __init__(
            self, data_dir: str, n_images: int, load_baseline: bool = False
    ):
        self.ground_truth = []
        self.input_images = []
        self.baseline_images = []

        ground_truth_dir = Path(data_dir) / PATCHES_TEMPLATE.format(PATCH_SIZE)
        cropped_dir = Path(data_dir) / CROPPED_TEMPLATE.format(PATCH_SIZE, STRIDE)
        baseline_dir = Path(data_dir) / BASELINE_TEMPLATE.format(PATCH_SIZE, STRIDE)

        if not cropped_dir.exists() and not ground_truth_dir.exists():
            raise FileNotFoundError(
                "Make sure you've preprocessed your imagery with this configuration:"
                f"\n{STRIDE = } \n{PATCH_SIZE = }"
            )
        ground_truth_files = sorted(list(ground_truth_dir.glob("*.jpg")))  # dangerous wildcards
        cropped_files = sorted(list(cropped_dir.glob("*.jpg")))
        baseline_files = sorted(list(baseline_dir.glob("*.jpg")))

        if not ground_truth_files or not cropped_files:
            raise FileNotFoundError("Empty dirs")

        n_images = len(ground_truth_files) if n_images == -1 else n_images

        for index in tqdm(range(n_images), desc="loading images"):
            patch = np.array(io.imread(ground_truth_files[index]))
            patch = patch.transpose(2, 0, 1) / 255
            self.ground_truth.append(torch.tensor(patch, dtype=torch.float32))

            cropped_image = np.array(io.imread(cropped_files[index]))
            # cropped_image = cropped_image[margin_size:-margin_size, margin_size:-margin_size]
            cropped_image = cropped_image.transpose(2, 0, 1) / 255
            cropped_image = torch.tensor(cropped_image, dtype=torch.float32)
            # todo make this dependent on the patch_size:
            cropped_image = F.pad(cropped_image, (20, 20, 20, 20), mode='reflect')
            self.input_images.append(torch.tensor(cropped_image, dtype=torch.float32))

            if load_baseline:
                baseline_image = np.array(io.imread(baseline_files[index]))
                baseline_image = baseline_image.transpose(2, 0, 1) / 255
                baseline_image = torch.tensor(baseline_image, dtype=torch.float32)
                baseline_image = F.pad(baseline_image, (20, 20, 20, 20), mode='reflect')
                self.baseline_images.append(torch.tensor(baseline_image, dtype=torch.float32))

        self.input_images = torch.stack(self.input_images)
        self.ground_truth = torch.stack(self.ground_truth)
        self.baseline_images = torch.stack(self.baseline_images) if load_baseline else []

    def __getitem__(self, index):
        if self.baseline_images.dim() > 1:
            return self.input_images[index], self.ground_truth[index], self.baseline_images[index]
        else:
            return self.input_images[index], self.ground_truth[index]

    def __len__(self):
        return len(self.input_images)


# utils functions...


def init_save_dir():
    try:
        # generate a random word
        from wonderwords import RandomWord
        r = RandomWord()
        identifiers = [r.word(), r.word()]
    except Exception:
        identifiers = ["model"]

    now = datetime.now().strftime("%H%M%d%m")
    save_dir = Path(SAVE_DIR_TEMPLATE.format(now, "_".join(identifiers)))
    save_dir.mkdir(exist_ok=False, parents=True)
    return save_dir
