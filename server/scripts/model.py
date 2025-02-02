import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_block_1 = self._create_conv_block(
            config.INPUT_CHANNELS,
            config.HIDDEN_UNITS
        )
        self.conv_block_2 = self._create_conv_block(
            config.HIDDEN_UNITS,
            config.HIDDEN_UNITS
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.HIDDEN_UNITS * 7 * 7, config.NUM_CLASSES)
        )

    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x
