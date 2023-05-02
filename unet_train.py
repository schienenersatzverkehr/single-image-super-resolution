from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

from config import SAVE_DIR_TEMPLATE
from data import Data, compare_images
import argparse

from data import Data


def PSNRLoss(batch_1, batch_2):
    """peak signal-to-noise ratio loss"""
    # mse = torch.nn.MSELoss()
    # mse_loss = mse(batch_1, batch_2)
    # psnr = 10 * torch.log10(1 / mse_loss)
    # return psnr
    mse = np.mean((batch_1 - batch_2) ** 2, axis=(1, 2, 3))
    max_pixel = 1.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return np.mean(psnr)


LOSSES = dict(
    mse=torch.nn.MSELoss(),
    bce=torch.nn.BCELoss(),
    psnr=PSNRLoss,
    # maybe add DiceLoss
)


class UNet(torch.nn.Module):
    def __init__(self, input_channels=3, out_channels=3, n_filters=64):
        super().__init__()
        # Learnable
        self.conv1A = torch.nn.Conv2d(input_channels, n_filters, 3)
        self.conv1B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.conv2A = torch.nn.Conv2d(n_filters, 2 * n_filters, 3)
        self.conv2B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv3A = torch.nn.Conv2d(2 * n_filters, 4 * n_filters, 3)
        self.conv3B = torch.nn.Conv2d(4 * n_filters, 4 * n_filters, 3)
        self.conv4A = torch.nn.Conv2d(4 * n_filters, 2 * n_filters, 3)
        self.conv4B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv5A = torch.nn.Conv2d(2 * n_filters, n_filters, 3)
        self.conv5B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.convtrans34 = torch.nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, 2, stride=2)
        self.convtrans45 = torch.nn.ConvTranspose2d(2 * n_filters, n_filters, 2, stride=2)

        self.convfinal = torch.nn.Conv2d(n_filters, out_channels, 1)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2)))))
        out = torch.cat([self.convtrans34(out), l2[:, :, 4:-4, 4:-4]], dim=1)  # copy & crop

        # out = torch.cat([self.convtrans34(out), l2], dim=1)
        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))
        out = torch.cat([self.convtrans45(out), l1[:, :, 16:-16, 16:-16]], dim=1)
        # out = torch.cat([self.convtrans45(out), l1], dim=1)
        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))

        # Finishing
        out = self.convfinal(out)

        return out


def PSNRLoss(batch_1, batch_2):
    """peak signal-to-noise ratio loss"""
    # mse = torch.nn.MSELoss()
    # mse_loss = mse(batch_1, batch_2)
    # psnr = 10 * torch.log10(1 / mse_loss)
    # return psnr
    mse = np.mean((batch_1 - batch_2) ** 2, axis=(1,2,3))
    max_pixel = 1.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return np.mean(psnr)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--n_images",  action="amount of images")
    # args = parser.parse_args()
    # full_set = args.n_images

    full_set = 400  # running oom for 1000
    data_set = Data(data_dir="./data", n_images=full_set)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info('Using cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info('Using mps')
    else:
        device = torch.device('cpu')
        logger.info('Using cpu')

    # device = torch.device('cpu')
    validation_split = .2  # percent we want to
    shuffle_dataset = True
    random_seed = 42
    batch_size = 10

    indices = list(range(full_set))
    split = int(np.floor(validation_split * full_set))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(
        Subset(data_set, train_indices),
        batch_size=batch_size,
        drop_last=True,
        # sampler=SubsetRandomSampler(train_indices) # why does this not work?
    )
    validation_loader = torch.utils.data.DataLoader(
        Subset(data_set, val_indices),
        batch_size=batch_size * 2,
        # sampler=SubsetRandomSampler(val_indices)
    )

    lr = 0.0001
    nr_epochs = 50
    choose_loss = "mse"
    n_filters = 32

    model = UNet(n_filters=n_filters).to(device)
    loss_function = LOSSES[choose_loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10)

    batch_losses = []
    epoch_losses = []
    val_losses = []

    try:
        # generate a random word
        from wonderwords import RandomWord
        r = RandomWord()
        identifiers = [r.word(), r.word()]
    except:
        identifiers = ["model"]

    now = datetime.now().strftime("%H%M%d%m")
    save_dir = Path(SAVE_DIR_TEMPLATE.format(now, choose_loss, "_".join(identifiers)))
    save_dir.mkdir(exist_ok=False, parents=True)

    logger.info(
        f"\nRunning training for {nr_epochs} epochs using {choose_loss.upper()} Loss"
        f"\nSaving results to:"
        f"\n{save_dir}"
    )

    image = data_set.input_images[3]  # for sample output during training

    for epoch in range(nr_epochs):
        epoch_loss = 0.0
        for n_batch, batch in enumerate(train_loader):
            image_batch, target_batch = batch  # unpack the data
            image_batch = image_batch.to(device)
            target_batch = target_batch.to(device)

            logits_batch = model(image_batch)
            optimizer.zero_grad()
            loss = loss_function(logits_batch, target_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_losses.append(loss.item())

        epoch_losses.append(epoch_loss / len(train_loader))
        logger.info(f'Epoch {epoch}/{nr_epochs}, loss {epoch_losses[-1]}')

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            val_loss = 0
            for validation_batch in validation_loader:
                val_image_batch, val_target_batch = validation_batch  # unpack the data
                val_image_batch = val_image_batch.to(device)
                val_target_batch = val_target_batch.to(device)
                logits_batch = model(val_image_batch)
                loss = loss_function(logits_batch, val_target_batch)
                val_loss += loss.item()
            val_losses.append(val_loss / len(validation_loader))
        if epoch % 5 == 4:
            logits = model.relu(logits)
            prob = torch.nn.functional.tanh(logits)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(prob[0, 1].cpu().detach())
            ax[0].imshow(prob.cpu().numpy().squeeze().transpose(1, 2, 0))
            ax[0].set_title(f'Prediction, epoch:{len(epoch_losses) - 1}')

            ax[1].plot(np.linspace(0, len(epoch_losses), len(batch_losses)),
                       batch_losses, lw=0.5)  # blue
            ax[1].plot(np.arange(len(epoch_losses)) + 0.5, epoch_losses, lw=2)  # orange
            ax[1].plot(np.linspace(0, len(epoch_losses) - 0.5, len(val_losses)),
                       val_losses, lw=1)  # green
            ax[1].set_title('Batch loss, epoch loss (training) and test loss')
            ax[1].set_ylim(0, 1.1 * max(epoch_losses + val_losses))
            plt.savefig(save_dir / f"loss_epoch{epoch:03}.jpg", dpi=300)

    torch.save(model.state_dict(), save_dir / "model.pt")

    # export epoch and validation loss as a csv
    pd.DataFrame(
        [epoch_losses, val_losses],
        columns=range(nr_epochs), index=["train", "test"]
    ).T.to_csv(save_dir / "losses.csv")

    # plotting an inference example
    peek_index = 9

    input_image = image_batch[peek_index]
    target_image = target_batch[peek_index]

    logits = model(input_image.unsqueeze(0).to(device))
    reconstruction = torch.nn.functional.tanh(logits).cpu().detach().numpy().squeeze().transpose(1, 2, 0)

    input_display = input_image.cpu().numpy().squeeze().transpose(1, 2, 0)
    target_display = target_image.cpu().numpy().squeeze().transpose(1, 2, 0)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(reconstruction[100:200, 100:200, :])
    ax[0, 1].imshow(input_display[100:200, 100:200, :])
    ax[0, 2].imshow(target_display[100:200, 100:200, :])

    ax[1, 0].imshow(reconstruction[:, :, :])
    ax[1, 1].imshow(input_display[:, :, :])
    ax[1, 2].imshow(target_display[:, :, :])
    plt.savefig(save_dir / "reconstruction_sample.jpg", dpi=300)

    fig = compare_images(reconstruction, target_image)
    plt.show()


if __name__ == '__main__':
    main()
