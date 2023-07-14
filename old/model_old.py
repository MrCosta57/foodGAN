from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torchvision
import lightning.pytorch as pl
from torch.autograd import Variable
import config


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.discr = nn.Sequential(
            # input: N x channels_img x 64 x 64
            # channels_img+1 because also lables are taken into account as an additional dimension
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed=nn.Embedding(num_classes, img_size*img_size)
        self.img_size=img_size


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def forward(self, x, lables):
        embedding=self.embed(lables).view(lables.shape[0], 1, self.img_size, self.img_size)
        x=torch.cat([x, embedding], dim=1)
        return self.discr(x)


# Generator model
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.gener = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.img_size=img_size
        self.embed=nn.Embedding(num_classes, embed_size)


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def forward(self, x, labels):
        #latent vector z: N x noise_dim x 1 x 1
        embedding=self.embed(labels)
        embedding=embedding.unsqueeze(2).unsqueeze(3)
        x=torch.cat([x, embedding], dim=1)
        return self.gener(x)
    



class GAN(pl.LightningModule):
    def __init__(self,
                 latent_dim: int = config.Z_DIM, lr: float = config.LEARNING_RATE,
                 b1: float = config.B1,
                 b2: float = config.B2,
                 critic_iterations : int = config.CRITIC_ITERATIONS,
                 lambda_gp : float = config.LAMBDA_GP,
                 channel_img :int = config.CHANNELS_IMG,
                 features_d: int = config.FEATURES_DISC,
                 features_g: int = config.FEATURES_GEN,
                 img_size: int = config.IMG_SIZE,
                 embed_size: int = config.GEN_EMBEDDING,
                 num_classes: int =config.NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.discriminator = Discriminator(channel_img, features_d, num_classes, img_size)
        self.generator = Generator(latent_dim, channel_img, features_g, num_classes, img_size, embed_size)
        self._initialize_weights(self.discriminator)
        self._initialize_weights(self.generator)

        fixed_noise = torch.randn(32, latent_dim, 1, 1)
        self.step=0
        #self.validation_z = torch.randn(8, self.hparams.latent_dim)
        #self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z, y):
        return self.generator(z, y)


    def training_step(self, batch, batch_idx):
        real, labels = batch

        print("REAL SHAPE: ", real.shape)
        print("LABELS SHAPE: ", labels.shape)
        optimizer_d, optimizer_g = self.optimizers()

        # TRAIN Discriminator: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        # (Measure discriminator's ability to classify real from generated samples)
        self.toggle_optimizer(optimizer_d)

        for _ in range(self.hparams.critic_iterations):
            noise = torch.randn(real.shape[0], self.hparams.latent_dim, 1, 1).to(self.device)

            fake = self.generator(noise, labels)
            discr_real = self.discriminator(real, labels).reshape(-1)
            discr_fake = self.discriminator(fake, labels).reshape(-1)

            gp = self._gradient_penalty(labels, real, fake).to(self.device)
            loss_discr = ( -(torch.mean(discr_real) - torch.mean(discr_fake)) + self.hparams.lambda_gp * gp)

            self.log("loss_discr", loss_discr, prog_bar=True)
        
            self.manual_backward(loss_discr, retain_graph=True) #
            optimizer_d.step()
            optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)


        #TRAIN Generator
        fake.require_grad=True
        self.toggle_optimizer(optimizer_g)
        gen_fake = self.discriminator(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)

        self.log("loss_gen", loss_gen, prog_bar=True)
        self.manual_backward(loss_gen)
        optimizer_g.step()
        optimizer_g.zero_grad()

        self.untoggle_optimizer(optimizer_g)

        """ if batch_idx % 100 == 0 and batch_idx > 0:

            with torch.no_grad():
                fake = self.generator(self.fixed_noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                self.logger.experiment.log_image(key="real", images=[img_grid_real])
                self.logger.experiment.log_image(key="fake", images=[img_grid_fake]) """
        
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx: int):
        if batch_idx % 100 == 0 and batch_idx > 0:
            x, _= batch
            z = self.fixed_noise.type_as(self.generator.model[0].weight)
            
            # log sampled images
            fake = self(z)
            real_grid = torchvision.utils.make_grid(x[:32], normalize=True)
            fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
            self.logger.experiment.add_image("Real Images", real_grid, global_step=self.step)
            self.logger.experiment.add_image("Fake Images", fake_grid, global_step=self.step)

            self.step+=1


    def _gradient_penalty(self, labels, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
        interpolated_images = real * alpha + fake * (1 - alpha)
        interpolated_images = Variable(interpolated_images, requires_grad=True).to(self.device)

        # Calculate discriminator scores
        mixed_scores = self.discriminator(interpolated_images, labels)
        
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty
    

    """ def validation_step(self, batch, batch_idx):
        #self.batch_size
        pass
    
    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass
    
    def on_train_epoch_end(self):
        pass
    """

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))

        return [opt_d, opt_g], []

    @staticmethod
    def _initialize_weights(model):
        # Initializes weights according to the DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02) #in-place update