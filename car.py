import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_size=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2)

        self.mu_layer = nn.Linear(in_features=2*2*256, out_features=z_size)
        self.log_var_layer = nn.Linear(in_features=2*2*256, out_features=z_size)

        self.z_size = z_size

    def forward(self, data):
        data = nn.ReLU()(self.conv1(data))
        data = nn.ReLU()(self.conv2(data))
        data = nn.ReLU()(self.conv3(data))
        data = nn.ReLU()(self.conv4(data))
        data = data.reshape(-1, 2*2*256)

        mu = nn.LeakyReLU()(self.mu_layer(data))
        log_var = nn.LeakyReLU()(self.log_var_layer(data))
        sigma = torch.exp(log_var/2)
        eps = torch.randn_like(log_var)

        kl_loss = 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1. - log_var)
        z = eps*sigma + mu  # Get a normally distributed vector with mean mu, standard deviation sigma
        return z, kl_loss


class Decoder(nn.Module):
    def __init__(self, z_size=32):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(in_features=z_size, out_features=2*2*256)
        self.convt1 = nn.ConvTranspose2d(in_channels=2*2*256, out_channels=128, kernel_size=(5, 5), stride=2)
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2)
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(6, 6), stride=2)
        self.convt4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(6, 6), stride=2)

    def forward(self, z):
        h = self.dense(z)
        h = h.reshape(-1, 2*2*256, 1, 1)
        h = nn.ReLU()(self.convt1(h))
        h = nn.ReLU()(self.convt2(h))
        h = nn.ReLU()(self.convt3(h))
        image_reconstruction = nn.Sigmoid()(self.convt4(h))
        return image_reconstruction


class V(nn.Module):
    def __init__(self):
        super(V, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_frame):
        z, kl_loss = self.encoder(input_frame)
        output = self.decoder(z)
        return z, output, kl_loss
