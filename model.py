import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from lightning.fabric import Fabric
import config

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            #Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1), # img: 32 x 32
            nn.LeakyReLU(0.2),

            #The definition is: _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1), # img: 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1), # img: 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1), # img: 4 x 4

            # Conv2d below makes into 1x1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

        #To handel class labels' information
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
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding=self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x=torch.cat([x, embedding], dim=1)
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            #Input: N x channels_noise x 1 x 1
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            #Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.img_size=img_size
        #To handel class labels' information
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
        embedding=self.embed(labels)
        embedding=embedding.unsqueeze(2).unsqueeze(3)
        x=torch.cat([x, embedding], dim=1)
        return self.net(x)



class GAN(nn.Module):
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
                 num_classes: int = config.NUM_CLASSES,
                 precision: str = config.PRECISION,
                 num_devices: int = config.NUM_DEVICES):
        super(GAN, self).__init__()

        #PARAMETERS
        self.step=0
        self.latent_dim=latent_dim
        self.lr=lr
        self.b1=b1
        self.b2=b2
        self.critic_iterations=critic_iterations
        self.labmda_gp=lambda_gp
        self.precision=precision
        self.num_devices=num_devices

        self.critic = Discriminator(channel_img, features_d, num_classes, img_size)#.to(device)
        self.generator = Generator(latent_dim, channel_img, features_g, num_classes, img_size, embed_size)#.to(device)
        self._initialize_weights(self.critic)
        self._initialize_weights(self.generator)
        

    def forward(self, z, y):
        return self.generator(z, y)
        
        
    def training_loop(self, data_loader, num_epochs=config.NUM_EPOCHS, debug_mode=False, debug_grad=False):
        if debug_grad: torch.autograd.set_detect_anomaly(True)
        print("Training started!")
        opt_critic, opt_g=self._configure_optimizers()

        #Fabric framework optimization
        fabric=Fabric(accelerator="auto", precision=self.precision, devices=self.num_devices, strategy="auto")
        self.critic, opt_critic=fabric.setup(self.critic, opt_critic)
        self.generator, opt_g=fabric.setup(self.generator, opt_g)
        data_loader=fabric.setup_dataloaders(data_loader)
        print("End fabric setup")
        
        self.critic.train()
        self.generator.train()

        if debug_mode:
            self.logger = SummaryWriter(f"logs/food_gan/")
            self.critic_loss_list=[]
            self.gen_loss_list=[]
            self.validation_z = torch.randn(data_loader.batch_size, self.latent_dim, 1, 1, device=fabric.device)
        
        for epoch in range(num_epochs):
            for batch_idx, (real, labels) in enumerate(tqdm(data_loader)):
                #real=real.to(self.device)
                #labels=labels.to(self.device)

                for _ in range(self.critic_iterations):
                    noise = torch.randn(real.shape[0], self.latent_dim, 1, 1, device=fabric.device)#.to(self.device)
                    fake = self.generator(noise, labels)

                    critic_real = self.critic(real, labels).reshape(-1)
                    critic_fake = self.critic(fake, labels).reshape(-1)

                    gp = self._gradient_penalty(labels, real, fake, fabric.device)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.labmda_gp * gp
                    if debug_mode: self.critic_loss_list.append(loss_critic.item())
                    if debug_mode and batch_idx % config.DEBUG_EVERY_ITER == 0 and batch_idx > 0:
                        self.logger.add_scalar("Loss of Critic", np.mean(self.critic_loss_list))
                        self.critic_loss_list=[]
                    self.critic.zero_grad()
                    fabric.backward(loss_critic, retain_graph=True)
                    opt_critic.step()

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                gen_fake = self.critic(fake, labels).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                if debug_mode: self.gen_loss_list.append(loss_gen.item())
                if debug_mode and batch_idx % config.DEBUG_EVERY_ITER == 0 and batch_idx > 0:
                    self.logger.add_scalar("Loss of Gen", np.mean(self.gen_loss_list))
                    self.gen_loss_list=[]
                self.generator.zero_grad()
                fabric.backward(loss_gen)
                opt_g.step()

                if debug_mode and batch_idx % config.DEBUG_EVERY_ITER == 0 and batch_idx > 0:
                    self._on_train_batch_end(real, labels, epoch, batch_idx, data_loader, loss_critic.item(), loss_gen.item(), num_epochs)
        print("Training ended!")


    def _on_train_batch_end(self, x, labels, epoch, batch_idx, loader, loss_critic, loss_gen, num_epochs):
        print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
        with torch.no_grad():   
            # log sampled images
            fake = self(self.validation_z, labels)
            real_grid = torchvision.utils.make_grid(x[:32], normalize=True)
            fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
            self.logger.add_image("Real Images", real_grid, global_step=self.step)
            self.logger.add_image("Fake Images", fake_grid, global_step=self.step)
        self.step+=1

        

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
        opt_d = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))

        return opt_d, opt_g

    @staticmethod
    def _initialize_weights(model):
        # Initializes weights according to the DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02) #in-place update



    def save_checkpoint(state, filename="food101_wgan_gp.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)


    def load_checkpoint(checkpoint, gen, disc):
        print("=> Loading checkpoint")
        gen.load_state_dict(checkpoint['gen'])
        disc.load_state_dict(checkpoint['disc'])
