import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()

        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4,
                         stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                  stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.use_dropout = use_dropout
        self.dropout     = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    """
    U-Net Generator
    Input:  L channel   (1, 256, 256)
    Output: AB channels (2, 256, 256)
    """
    def __init__(self):
        super().__init__()

        self.enc1       = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2       = UNetBlock(64,  128)
        self.enc3       = UNetBlock(128, 256)
        self.enc4       = UNetBlock(256, 512)
        self.enc5       = UNetBlock(512, 512)
        self.enc6       = UNetBlock(512, 512)
        self.enc7       = UNetBlock(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec1  = UNetBlock(512,  512, down=False, use_dropout=True)
        self.dec2  = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.dec3  = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.dec4  = UNetBlock(1024, 512, down=False)
        self.dec5  = UNetBlock(1024, 256, down=False)
        self.dec6  = UNetBlock(512,  128, down=False)
        self.dec7  = UNetBlock(256,   64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b  = self.bottleneck(e7)
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        return self.final(torch.cat([d7, e1], dim=1))


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    Input:  L + AB concatenated (3, 256, 256)
    Output: Patch predictions   (1, 30, 30)
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, L, AB):
        x = torch.cat([L, AB], dim=1)
        return self.model(x)


def init_weights(model):
    """
    Initialize weights using normal distribution
    as per the original Pix2Pix paper
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, 0)
    return model