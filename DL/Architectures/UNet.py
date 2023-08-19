import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# Definition of the DoubleConv2d block
class DoubleConv2d(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DoubleConv2d, self).__init__()
        #  (two consecutive convolutional layers with ReLU activations)
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv2d(x)


# Definition of the UNet architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # (encoder, bottleneck, and decoder parts)

        self.encoder = nn.ModuleList()
        in_channel = 1
        out_channel = 64
        for i in range(4):
            self.encoder.append(DoubleConv2d(in_channel, out_channel))
            in_channel = out_channel
            out_channel = out_channel * 2

        self.bottleneck = DoubleConv2d(512, 1024)

        self.decoder = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        in_channel = 1024
        out_channel = 512
        for i in range(4):
            self.decoder.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2))
            self.up_conv.append(DoubleConv2d(in_channel, out_channel))
            in_channel = out_channel
            out_channel = out_channel // 2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # (forward pass through encoder, bottleneck, and decoder)
        skip_connection = []
        for down in self.encoder:
            x = down(x)
            skip_connection.append(x)
            x = self.pool(x)
        skip_connection = skip_connection[::-1]

        x = self.bottleneck(x)

        for i in range(4):
            x = self.decoder[i](x)
            y = skip_connection[i]
            y = TF.resize(y, x.shape[2])
            x = torch.cat((x, y), dim=1)
            x = self.up_conv[i](x)
        x = self.final_conv(x)
        return x

# Instantiate the UNet model
model = UNet()
# Create a random input tensor for testing
input_t = torch.rand(1, 1, 572, 572)
# Print the shape of the output from the model
print(model(input_t).shape)
# print(model)