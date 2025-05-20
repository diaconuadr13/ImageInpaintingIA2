import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        PatchGAN discriminator.
        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            ndf (int): Number of filters in the first convolutional layer.
            n_layers (int): Number of convolutional layers (excluding the first and last).
            norm_layer (nn.Module): Normalization layer type.
        """
        super(PatchGANDiscriminator, self).__init__()

        if type(norm_layer) == nn.BatchNorm2d:
            use_bias = False # BatchNorm has its own bias
        else:
            use_bias = True

        kw = 4 # Kernel_width
        padw = 1 # Padding_width

        # Initial convolutional layer
        sequence = [
            nn.Conv2d(input_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # Intermediate convolutional layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final convolutional layer (before the output layer)
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer: 1-channel prediction map (each value is real/fake for a patch)
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input_image):
        """Standard forward."""
        return self.model(input_image)