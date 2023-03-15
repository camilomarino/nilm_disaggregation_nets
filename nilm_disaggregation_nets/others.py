from torch import nn


class FullyConvolutionalDAE(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.name = "FullyConvolutionalDAE"
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=4, out_channels=out_channels, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        return self.layers(x)
