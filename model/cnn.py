import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 2), stride=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 2), stride=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 2), stride=(2, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 2), stride=(2, 1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 2), stride=(2, 1))

        # Decoder
        self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))
        # output_padding Is 1, otherwise it’s 79
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 2), stride=(2, 1), output_padding=(1, 0))
        self.convT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))

    def forward(self, x):
        # Encoder
        # (B, in_c, T, F)

        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x1))
        x3 = F.elu(self.conv3(x2))
        x4 = F.elu(self.conv4(x3))
        x5 = F.elu(self.conv5(x4))
        output = x5

        # Decoder
        res1 = F.elu(self.convT1(output))
        res2 = F.elu(self.convT2(res1))
        res3 = F.elu(self.convT3(res2))
        res4 = F.elu(self.convT4(res3))
        # (B, o_c, T. F)
        res5 = F.relu(self.convT5(res4))
        return res5.squeeze()
