from loguru import logger

from datetime import datetime
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data import Data, init_save_dir
from evaluate import compare_batches, compare_images
from srgan import UNet, VGG16Discriminator


def main():
    full_set = 200  # running oom for 1000
    data_set = Data(data_dir="./data", n_images=full_set, load_baseline=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info('Using cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info('Using mps')
    else:
        device = torch.device('cpu')
        logger.info('Using cpu')

    device = torch.device('cpu')
    validation_split = .2  # percent we want to use for validation
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
    )
    validation_loader = torch.utils.data.DataLoader(
        Subset(data_set, val_indices),
        batch_size=batch_size * 2,
    )

    lr = 0.0001
    nr_epochs = 100
    n_filters = 16

    batch_losses = []
    epoch_losses = []
    val_losses = []

    save_dir = init_save_dir()
    logger.info(
        f"\nRunning training with "
        f"\n{nr_epochs=} \n{batch_size=}\n{lr=}\n{n_filters=}"
        f"\nSaving results to:"
        f"\n{save_dir}"
    )

    #  ---- init model settings and models ----

    # to tune the complexity of the generator
    # and thus the complexity of the images
    latent_space = 100

    discriminator = VGG16Discriminator(pretrained=False).to(device)
    generator = UNet(n_filters=n_filters).to(device)

    # loss_function = LOSSES[choose_loss]

    # content_loss = torch.nn.L1Loss()  # or torch.nn.MSELoss()
    content_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    # generator_optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10)

    image = data_set.input_images[3]  # for sample output during training
    losses_df = pd.DataFrame(
        index=range(nr_epochs),
        columns=["d_real", "d_fake", "discr", "gen", "gen_val", "ssim", "mse", "psnr"]
    )
    # Training Begin
    for epoch in range(nr_epochs):
        # for n_batch, (input_batch, target_batch) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}/{nr_epochs}"):
        for n_batch, (_, target_batch, input_batch) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}/{nr_epochs}"):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            discriminator_optimizer.zero_grad()

            real_outputs = discriminator(target_batch)
            real_loss = adversarial_loss(real_outputs, real_labels)
            losses_df.loc[epoch, "d_real"] = real_loss.item()

            fake_images = generator(input_batch)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = adversarial_loss(fake_outputs, fake_labels)
            losses_df.loc[epoch, "d_fake"] = fake_loss.item()

            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()
            losses_df.loc[epoch, "discr"] = discriminator_loss.item()

            # Train the generator
            generator_optimizer.zero_grad()
            generations = generator(input_batch)
            content_outputs = content_loss(generations, target_batch)
            adversarial_outputs = adversarial_loss(
                discriminator(generations), real_labels
            )
            generator_loss = content_outputs + adversarial_outputs
            generator_loss.backward()
            generator_optimizer.step()
            losses_df.loc[epoch, "gen"] = generator_loss.item()

            # logging
            '\t'.join([f"{k}={v:.3f}" for k, v in losses_df.loc[epoch].to_dict().items()])
            # epoch_loss += loss.item()
            batch_loss = losses_df.loc[epoch].mean()
            batch_losses.append(batch_loss)
        epoch_loss = losses_df.loc[epoch].mean()
        epoch_losses.append(epoch_loss / len(train_loader))
        # logger.info(f'Epoch {epoch}/{nr_epochs}, loss {epoch_losses[-1]}')

        with torch.no_grad():
            logits = generator(image.unsqueeze(0).to(device))
            val_loss = 0
            for _, val_target_batch, val_image_batch in validation_loader:
                val_image_batch, val_target_batch = val_image_batch.to(device), val_target_batch.to(device)

                logits_batch = generator(val_image_batch)
                loss = content_loss(logits_batch, val_target_batch)
                val_loss += loss.item()
            val_losses.append(val_loss / len(validation_loader))
            losses_df.loc[epoch, "gen_val"] = val_loss
            losses_df.loc[epoch, ["ssim", "mse", "psnr"]] = compare_batches(
                input_batch=torch.nn.functional.tanh(
                    torch.nn.functional.relu(logits_batch)
                ),
                target_batch=val_target_batch
            )
        print(
            f"\t Losses: \t" + " | ".join(
                [f"{k}={v:06.3f}" for k, v in losses_df.loc[epoch].to_dict().items()])
        )
        if epoch % 5 == 4:
            logits = generator.relu(logits)
            prob = torch.nn.functional.tanh(logits)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(prob[0, 1].cpu().detach())
            ax[0].imshow(prob.cpu().numpy().squeeze().transpose(1, 2, 0))
            ax[0].set_title(f'Prediction, epoch:{epoch}')

            sns.lineplot(
                losses_df[["gen", "gen_val", "ssim", "mse"]],
                ax=ax[1],
                markers=True
            )
            # plt.show(block=False)
            plt.savefig(save_dir / f"loss_epoch{epoch:03}.jpg", dpi=300)
    # End of training

    torch.save(generator.state_dict(), save_dir / "model.pt")

    # export epoch and validation loss as a csv
    # pd.DataFrame(lo
    #     [epoch_losses, val_losses],
    #     columns=range(nr_epochs), index=["train", "test"]
    # ).T.to_csv(save_dir / "losses.csv")

    losses_df.to_csv(save_dir / "losses.csv")
    # plotting an inference example
    peek_index = 9

    input_image = input_batch[peek_index]
    target_image = target_batch[peek_index]

    logits = generator(input_image.unsqueeze(0).to(device))
    reconstruction = torch.nn.functional.tanh(logits).cpu().detach().numpy().squeeze().transpose(1, 2, 0)

    input_display = input_image.cpu().numpy().squeeze().transpose(1, 2, 0)
    target_display = target_image.cpu().numpy().squeeze().transpose(1, 2, 0)

    eval_srgan = compare_images(target_image, reconstruction)
    eval_baseline = compare_images(target_image, reconstruction)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(input_display[100:200, 100:200, :])
    ax[0, 0].set_title("LowRes")
    ax[0, 1].imshow(reconstruction[100:200, 100:200, :])
    ax[0, 1].set_title("SRGAN reconstruction")
    ax[0, 2].imshow(target_display[100:200, 100:200, :])
    ax[1, 1].set_title("HighRes")

    ax[1, 0].imshow(input_display[:, :, :])
    ax[1, 0].imshow(reconstruction[:, :, :])
    ax[1, 2].imshow(target_display[:, :, :])

    plt.savefig(save_dir / "srgan_reconstruction_sample.jpg", dpi=300)

    # inferenece_dir = save_dir / "inference"
    # inferenece_dir.mkdir()

    # fig = compare_images(reconstruction, target_image)
    # plt.show()


if __name__ == '__main__':
    start = datetime.now()
    main()
    logger.info(f"Total Runtime: {datetime.now() - start}")
