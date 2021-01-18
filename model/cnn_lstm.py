import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_LSTM(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 2), stride=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 2), stride=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 2), stride=(2, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 2), stride=(2, 1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 2), stride=(2, 1))

        # LSTM
        #self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)
        self.lstm = nn.LSTM(
            input_size=2304, hidden_size=2304, num_layers=2, batch_first=True
        )
        # Decoder
        self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))
        # output_padding Is 1, otherwise it’s 79
        self.convT4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 2), stride=(2, 1), output_padding=(1, 0))
        self.convT5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 2), stride=(2, 1), output_padding=(0, 0))

    def forward(self, x):
        self.lstm.flatten_parameters()
        # Encoder
        # (B, in_c, T, F)
        print('input.shape: ', x.shape)
        x1 = F.elu(self.conv1(x))
        print('x1.shape: ', x1.shape)
        x2 = F.elu(self.conv2(x1))
        print('x2.shape: ', x2.shape)
        x3 = F.elu(self.conv3(x2))
        print('x3.shape: ', x3.shape)
        x4 = F.elu(self.conv4(x3))
        print('x4.shape: ', x4.shape)
        x5 = F.elu(self.conv5(x4))
        output_enc = x5
        print('x5: ', output_enc.shape)
        batch_size, n_channels, n_f_bins, n_frame_size = output_enc.shape
        output_enc = output_enc.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        print('output_enc.shape: ', output_enc.shape)
        lstm_out, (hn, cn) = self.lstm(output_enc)
        print('lstm_out.shape: ', lstm_out.shape)
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)
        
        # Decoder
        res1 = F.elu(self.convT1(lstm_out))
        res2 = F.elu(self.convT2(res1))
        res3 = F.elu(self.convT3(res2))
        res4 = F.elu(self.convT4(res3))
        # (B, o_c, T. F)
        res5 = F.relu(self.convT5(res4))
        return res5.squeeze()
