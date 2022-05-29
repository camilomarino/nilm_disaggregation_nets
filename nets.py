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


class SeqToBase(nn.Module):
    def __init__(self, input_length: int, in_channels: int = 1):
        super().__init__()
        self.input_length = input_length
        self.in_channels = in_channels

        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=30, kernel_size=10),
            nn.ReLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8), nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6), nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5), nn.ReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.layer_6 = nn.Sequential(
            nn.Linear(in_features=50 * (input_length - 29), out_features=1024),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: batch_size x in_channels x input_length
        x = self.layer_1(x)  # batch_size x 30 x (input_length-9)
        x = self.layer_2(x)  # batch_size x 30 x (input_length-16)
        x = self.layer_3(x)  # batch_size x 40 x (input_length-21)
        x = self.layer_4(x)  # batch_size x 50 x (input_length-25)
        x = self.layer_5(x)  # batch_size x 50 x (input_length-29)
        x = self.flatten(x)  # batch_size x 50*(input_length-29)
        x = self.layer_6(x)  # batch_size x 1024
        return x


class SeqToSeq(nn.Module):
    def __init__(self, input_length: int, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.name = "SeqToSeq"
        self.input_length = input_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.seq_to_base = SeqToBase(input_length=input_length, in_channels=in_channels)
        self.last_layer = nn.Linear(
            in_features=1024, out_features=out_channels * input_length
        )

    def forward(self, x):
        # x: batch_size x in_channels x input_length
        x = self.seq_to_base(x)  # batch_size x 1024
        x = self.last_layer(x)  # batch_size x (out_channels*input_length)
        x = x.view(
            (x.shape[0], self.out_channels, self.input_length)
        )  # batch_size x out_channels x input_length
        return x


class SeqToPoint(nn.Module):
    def __init__(self, input_length: int, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.name = "SeqToPoint"
        self.input_length = input_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.seq_to_base = SeqToBase(input_length=input_length, in_channels=in_channels)
        self.last_layer = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        # x: batch_size x in_channels x input_length
        x = self.seq_to_base(x)  # batch_size x 1024
        x = self.last_layer(x)  # batch_size x out_channels
        x = x.view((x.shape[0], self.out_channels, 1))  # batch_size x out_channels x 1
        return x
