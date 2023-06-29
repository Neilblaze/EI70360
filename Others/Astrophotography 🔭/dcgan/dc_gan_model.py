import torch.nn as nn
import torch


class DCGAN:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, device: torch.device, z_dim: int,
                 generator_lr=1e-3, discriminator_lr=2e-4):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        self.z_dim = z_dim

        self.criterion = nn.BCEWithLogitsLoss()

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=generator_lr)
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=discriminator_lr)

        self.cnt_parameters = {'G': sum(p.numel() for p in self.G.parameters() if p.requires_grad),
                               'D': sum(p.numel() for p in self.D.parameters() if p.requires_grad)
                               }

        self.real_label, self.fake_label = self.get_labels(0)

    def get_labels(self, batch_size: int):
        # Generate labels for real and fake images
        real_label = torch.full((batch_size, 1), 1., device=self.device)
        fake_label = torch.full((batch_size, 1), 0., device=self.device)
        return real_label, fake_label

    def train_gen_step(self, batch):
        # Generator training step
        self.G_optim.zero_grad()

        real_label, _ = self.get_labels(batch.shape[0])

        z = torch.randn(batch.shape[0], self.z_dim, 1, 1, device=self.device)

        fake_im = self.G(z)
        fake_im_prob = self.D(fake_im)

        G_loss = self.criterion(fake_im_prob, real_label)
        G_loss.backward()

        self.G_optim.step()

        return G_loss.item()

    def train_dis_step(self, batch):
        # Discriminator training step
        self.D_optim.zero_grad()

        real_im_prob = self.D(batch.to(self.device))
        real_D_loss = self.criterion(real_im_prob, self.real_label)

        with torch.no_grad():
            z = torch.randn(batch.shape[0], self.z_dim, 1, 1, device=self.device)
            fake_im = self.G(z)

        fake_im_prob = self.D(fake_im.detach())
        fake_D_loss = self.criterion(fake_im_prob, self.fake_label)

        D_loss = 0.5 * real_D_loss + 0.5 * fake_D_loss
        D_loss.backward()

        self.D_optim.step()

        return D_loss.cpu().item(), (real_D_loss.cpu().item(), fake_D_loss.cpu().item())
