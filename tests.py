from itertools import product

import pytest
import torch

import nets

# ### NEURAL NILM ###
in_channels = [i for i in range(1, 6)]
out_channels = [i for i in range(1, 6)]
sequence_length = [i * 10 for i in range(2, 10)]
batch_size = [i * 10 for i in range(1, 6)]


@pytest.mark.parametrize(
    "in_channels,out_channels,sequence_length,batch_size",
    list(product(in_channels, out_channels, sequence_length, batch_size)),
)
def test_NeuralNilmDAE(in_channels, out_channels, sequence_length, batch_size):

    net = nets.NeuralNilmDAE(
        sequence_length=sequence_length,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    x = torch.empty((batch_size, in_channels, sequence_length))
    y = net(x)

    assert list(y.shape) == [batch_size, out_channels, sequence_length]


@pytest.mark.parametrize(
    "in_channels,out_channels,sequence_length,batch_size",
    list(product(in_channels, out_channels, sequence_length, batch_size)),
)
def test_NeuralNilmBiLSTM(in_channels, out_channels, sequence_length, batch_size):

    net = nets.NeuralNilmBiLSTM(
        sequence_length=sequence_length,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    x = torch.empty((batch_size, in_channels, sequence_length))
    y = net(x)

    assert list(y.shape) == [batch_size, out_channels, sequence_length]


### SEQ2SEQ SEQ2POINT ###

in_channels = [i for i in range(1, 6)]
out_channels = [i for i in range(1, 6)]
input_length = [i * 10 for i in range(3, 10)]  # the minimun is 29
batch_size = [i * 10 for i in range(1, 6)]


@pytest.mark.parametrize(
    "in_channels,input_length,batch_size",
    list(product(in_channels, input_length, batch_size)),
)
def test_SeqToBase(in_channels, input_length, batch_size):

    net = nets.SeqToBase(input_length=input_length, in_channels=in_channels)
    x = torch.empty((batch_size, in_channels, input_length))
    y = net(x)

    assert list(y.shape) == [batch_size, 1024]


@pytest.mark.parametrize(
    "in_channels,out_channels,input_length,batch_size",
    list(product(in_channels, out_channels, input_length, batch_size)),
)
def test_SeqToSeq(in_channels, out_channels, input_length, batch_size):

    net = nets.SeqToSeq(
        input_length=input_length, in_channels=in_channels, out_channels=out_channels,
    )
    x = torch.empty((batch_size, in_channels, input_length))
    y = net(x)

    assert list(y.shape) == [batch_size, out_channels, input_length]


@pytest.mark.parametrize(
    "in_channels,out_channels,input_length,batch_size",
    list(product(in_channels, out_channels, input_length, batch_size)),
)
def test_SeqToPoin(in_channels, out_channels, input_length, batch_size):

    net = nets.SeqToPoint(
        input_length=input_length, in_channels=in_channels, out_channels=out_channels,
    )
    x = torch.empty((batch_size, in_channels, input_length))
    y = net(x)

    assert list(y.shape) == [batch_size, out_channels, 1]
