from typing import Literal
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from tqdm import tqdm
import numpy as np
import torchvision
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from lightning.fabric import Fabric
from torch.optim.adam import Adam
import config


class Discriminator(nn.Module):
    def __init__(
        self, channels_img: int, features_d: int, num_classes: int, img_size: int
    ):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img + 1, features_d, kernel_size=4, stride=2, padding=1
            ),  # img: 32 x 32
            nn.LeakyReLU(0.2),
            # The definition is: _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),  # img: 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # img: 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # img: 4 x 4
            # Conv2d below makes into 1x1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        # To handel class labels' information
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(
        self,
        channels_noise: int,
        channels_img: int,
        features_g: int,
        num_classes: int,
        img_size: int,
        embed_size: int,
    ):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(
                channels_noise + embed_size, features_g * 16, 4, 1, 0
            ),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.img_size = img_size
        # To handel class labels' information
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        embedding = self.embed(labels)
        embedding = embedding.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


class GAN(nn.Module):
    """
    WGAN-GP model implementation. Generator and Discriminator are CNNs.
    """

    def __init__(
        self,
        latent_dim: int = config.Z_DIM,
        lr: float = config.LEARNING_RATE,
        b1: float = config.B1,
        b2: float = config.B2,
        critic_iterations: int = config.CRITIC_ITERATIONS,
        lambda_gp: float = config.LAMBDA_GP,
        channel_img: int = config.CHANNELS_IMG,
        features_d: int = config.FEATURES_DISC,
        features_g: int = config.FEATURES_GEN,
        img_size: int = config.IMG_SIZE,
        embed_size: int = config.GEN_EMBEDDING,
        num_classes: int = config.NUM_CLASSES,
        precision: Literal["64", "32", "bf16-mixed", "16-mixed"] = config.PRECISION,
        num_devices: int = config.NUM_DEVICES,
    ):
        super(GAN, self).__init__()
        self.step = 0
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.critic_iterations = critic_iterations
        self.labmda_gp = lambda_gp
        self.precision = precision
        self.num_devices = num_devices
        self.critic = Discriminator(channel_img, features_d, num_classes, img_size)
        self.generator = Generator(
            latent_dim, channel_img, features_g, num_classes, img_size, embed_size
        )
        self._initialize_weights(self.critic)
        self._initialize_weights(self.generator)
        self.fabric = Fabric(
            accelerator="auto",
            precision=self.precision,
            devices=self.num_devices,
            strategy="auto",
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        return self.generator(z, y)

    def training_loop(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_epochs: int = config.NUM_EPOCHS,
        debug_mode: bool = False,
        debug_grad: bool = False,
    ):
        """
        Training loop for the WGAN-GP model. Train and test dataloader are used sequentially to train the model on the whole dataset.
        """
        if debug_grad:
            torch.autograd.set_detect_anomaly(True)
        print("Training started!")

        # Fabric framework initialization
        opt_critic, opt_g = self._configure_optimizers()
        self.critic, opt_critic = self.fabric.setup(self.critic, opt_critic)
        self.generator, opt_g = self.fabric.setup(self.generator, opt_g)
        data_loader_fabric = self.fabric.setup_dataloaders(data_loader)
        print("End fabric setup")

        self.critic.train()
        self.generator.train()
        if debug_mode:
            self.fabric._loggers = [TensorBoardLogger(config.LOG_DIR, name="food_gan")]
            self.critic_loss_list = []
            self.gen_loss_list = []
            self.validation_z = torch.randn(
                data_loader_fabric.batch_size,  # type: ignore
                self.latent_dim,
                1,
                1,
                device=self.fabric.device,
            )

        for epoch in range(num_epochs):
            for batch_idx, (real, labels) in enumerate(tqdm(data_loader_fabric)):
                # Train Critic: max E[critic(real)] - E[critic(fake)], equivalent to minimizing the negative of that
                noise = torch.randn(
                    real.shape[0], self.latent_dim, 1, 1, device=self.fabric.device
                )
                fake = self.generator(noise, labels)
                critic_real = self.critic(real, labels).reshape(-1)
                critic_fake = self.critic(fake, labels).reshape(-1)
                gp = self._gradient_penalty(labels, real, fake, self.fabric.device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + self.labmda_gp * gp
                )
                loss_critic_item = round(loss_critic.item(), 4)  # for debug purposes
                self.critic.zero_grad()
                self.fabric.backward(loss_critic, retain_graph=True)
                opt_critic.step()

                # Train the generator every `critic_iterations` steps
                if (batch_idx % self.critic_iterations) == 0 and batch_idx > 0:
                    # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                    gen_fake = self.critic(fake, labels).reshape(-1)
                    loss_gen = -torch.mean(gen_fake)
                    loss_gen_item = round(loss_gen.item(), 4)  # for debug purposes
                    self.generator.zero_grad()
                    self.fabric.backward(loss_gen)
                    opt_g.step()

                    # Logging
                    if debug_mode:
                        self.critic_loss_list.append(loss_critic_item)
                        self.gen_loss_list.append(loss_gen_item)

                    if (
                        debug_mode
                        and (
                            batch_idx
                            % (config.DEBUG_EVERY_ITER * self.critic_iterations)
                        )
                        == 0
                        and batch_idx > 0
                    ):
                        self.fabric.log(
                            "Loss of discriminator",
                            round(np.mean(self.critic_loss_list), 5),
                            self.step,
                        )
                        self.fabric.log(
                            "Loss of generator",
                            round(np.mean(self.gen_loss_list), 5),
                            self.step,
                        )
                        self._on_debug_batch_end(
                            real,
                            labels,
                            epoch + 1,
                            batch_idx,
                            data_loader_fabric,
                            loss_critic_item,
                            loss_gen_item,
                            num_epochs,
                            real.shape[0],
                        )
                        self.critic_loss_list = []
                        self.gen_loss_list = []
                        self.step += 1

            if debug_mode:
                self.fabric.logger.experiment.flush()
            if (epoch + 1) % 5 == 0 and (epoch + 1) != num_epochs:
                print("Checkpointing at epoch ", epoch + 1)
                GAN.save_model_ckp(self, f"food101_wgan_gp_epochs_{epoch+1}.ckpt")

        print("Training ended!")
        if debug_mode:
            self.fabric.logger.finalize("success")

        GAN.save_model_ckp(self, f"food101_wgan_gp_epochs_{num_epochs}.ckpt")

    def _gradient_penalty(self, labels, real, fake, device):
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * alpha + fake * (1 - alpha)
        # Calculate critic scores
        mixed_scores = self.critic(interpolated_images, labels)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores, device=device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def _configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        opt_d = Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        opt_g = Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return opt_d, opt_g

    def _on_debug_batch_end(
        self,
        x,
        labels,
        epoch,
        batch_idx,
        loader,
        loss_critic,
        loss_gen,
        num_epochs,
        batch_size,
    ):
        print(
            f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )
        with torch.no_grad():
            fake = self(self.validation_z[:batch_size], labels).detach()
            real_grid = torchvision.utils.make_grid(x[:32], normalize=True)
            fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
            self.fabric.logger.experiment.add_image("Real images", real_grid, self.step)
            self.fabric.logger.experiment.add_image("Fake images", fake_grid, self.step)

    def generate(self, label, num_pred=8):
        """
        Generates `num_pred` images of the specified `label`.
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(
                num_pred, self.latent_dim, 1, 1, device=self.fabric.device
            )
            labels = None
            if isinstance(label, int):
                labels = torch.tensor([label], device=self.fabric.device).repeat(
                    num_pred
                )
            else:
                labels = label.to(self.fabric.device)
            img = self(noise, labels).detach()
            grid = torchvision.utils.make_grid(img, normalize=True).cpu()
            grid = grid.permute(1, 2, 0).numpy()
            return grid

    @staticmethod
    def _initialize_weights(model):
        # Initializes weights according to the DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)  # in-place update

    @staticmethod
    def save_model_ckp(model, filename="food101_wgan_gp.ckpt"):
        state = {
            "latent_dim": model.latent_dim,
            "critic": model.critic,
            "generator": model.generator,
        }
        print("=> Saving checkpoint")
        model.fabric.save(config.CHECKPOINT_DIR + filename, state)
        print("=> Saving done!")

    @staticmethod
    def load_model_from_ckp(checkpoint_file):
        print("=> Loading checkpoint")
        model = GAN()
        full_dict = model.fabric.load(checkpoint_file)
        model.critic.load_state_dict(full_dict["critic"])
        model.generator.load_state_dict(full_dict["generator"])
        model.latent_dim = full_dict["latent_dim"]
        print("=> Loading done!")
        return model
