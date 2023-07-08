import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim, ngf, n_ch):
        super(Generator, self).__init__()

        # Generator network architecture
        self.main = nn.Sequential(
            # Transposed convolutional layers
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),  # Input: z_dim x 1 x 1, Output: (ngf * 8) x 4 x 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # Input: (ngf * 8) x 4 x 4, Output: (ngf * 4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # Input: (ngf * 4) x 8 x 8, Output: (ngf * 2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 2, 5, 1, 2),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # Input: (ngf * 2) x 16 x 16, Output: ngf x 32 x 32
            nn.Conv2d(ngf, ngf, 7, 1, 3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf, n_ch, 4, 2, 1, bias=False),  # Input: ngf x 32 x 32, Output: n_ch x 64 x 64
            nn.Conv2d(n_ch, n_ch, 7, 1, 3),
            nn.Tanh()  # Output: n_ch x 64 x 64
        )
        
        self.cnt_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, n_ch, ndf):
        super(Discriminator, self).__init__()

        # Discriminator network architecture
        self.main = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(n_ch, ndf, 4, 2, 1, bias=False),  # Input: n_ch x 64 x 64, Output: ndf x 32 x 32
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # Input: ndf x 32 x 32, Output: (ndf * 2) x 16 x 16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # Input: (ndf * 2) x 16 x 16, Output: (ndf * 4) x 8 x 8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # Input: (ndf * 4) x 8 x 8, Output: (ndf * 8) x 4 x 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # Input: (ndf * 8) x 4 x 4, Output: 1 x 1 x 1
            nn.Sigmoid()
        )

        self.cnt_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, im):
        return self.main(im)
