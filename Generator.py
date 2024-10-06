import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, text_embed_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100 + text_embed_size, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, noise, text_embed):
        x = torch.cat((noise, text_embed), dim=1)
        x = self.fc1(x).view(-1, 256, 8, 8)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.tanh(self.deconv3(x))

        return x
