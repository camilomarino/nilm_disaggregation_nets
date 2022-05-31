"""
Neural NILM: Deep Neural Networks Applied to Energy Disaggregation
https://arxiv.org/abs/1507.06594
"""


from torch import nn


class NeuralNilmDAE(nn.Module):
    def __init__(
        self, sequence_length: int, in_channels: int = 1, out_channels: int = 1,
    ):
        super().__init__()
        self.name = "NeuralNilmDAE"
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        # border_mode=valid is the same as padding=0
        self.layer_2 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=4)
        self.flatten = nn.Flatten()
        self.layer_3 = nn.Sequential(
            nn.Linear((sequence_length - 3) * 8, (sequence_length - 3) * 8), nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Linear((sequence_length - 3) * 8, 128), nn.ReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.Linear(128, (sequence_length - 3) * 8), nn.ReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(8, sequence_length - 3))
        self.layer_6 = nn.ConvTranspose1d(
            in_channels=8, out_channels=out_channels, kernel_size=4
        )

    def forward(self, x):
        # x: batch_size x in_channels x sequence_length
        x = self.layer_2(x)  # batch_size x 8 x (sequence_length-3)
        x = self.flatten(x)  # batch_size x 8*(sequence_length-3)
        x = self.layer_3(x)  # batch_size x 8*(sequence_length-3)
        x = self.layer_4(x)  # batch_size x 128
        x = self.layer_5(x)  # batch_size x 8*(sequence_length-3)
        x = self.unflatten(x)  # batch_size x 8 x (sequence_length-3)
        x = self.layer_6(x)  # batch_size x out_channels x sequence_length
        return x


class NeuralNilmBiLSTM(nn.Module):
    def __init__(
        self, sequence_length: int, in_channels: int = 1, out_channels: int = 1,
    ):
        super().__init__()
        self.name = "NeuralNilmBiLSTM"
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv1d(
            in_channels=in_channels, out_channels=16, kernel_size=4, padding="same"
        )

        self.lstm_1 = nn.LSTM(
            input_size=16, hidden_size=64, batch_first=True, bidirectional=True
        )
        self.lstm_2 = nn.LSTM(
            input_size=128, hidden_size=128, batch_first=True, bidirectional=True
        )

        self.fc_1 = nn.Sequential(nn.Linear(256, 128), nn.Tanh())
        self.fc_2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: batch_size x in_channels x input_length
        x = self.conv_1(x)  # batch_size x 16 x input_length

        x = x.transpose(1, 2)  # batch_size x input_length x 16
        x, _ = self.lstm_1(x)  # batch_size x input_length x 64*2
        x, _ = self.lstm_2(x)  # batch_size x input_length x 128*2

        x = self.fc_1(x)  # batch_size x input_length x 128
        x = self.fc_2(x)  # batch_size x input_length x out_channels

        x = x.transpose(1, 2)  # batch_size x out_channels x input_length
        return x
