import torch
import torch.nn as nn
from functorch import jacrev, vmap

from loss_and_metrics.loss import nll_surrogate

import numpy as np
import math

class resnet(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_blocks, output_dim, hidden_layers):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.n_blocks = n_blocks
    self.output_dim = output_dim
    self.hidden_layers = hidden_layers
    self.blocks = nn.ModuleList()

    self.first_layer = nn.Linear(self.input_dim, self.hidden_dim)

    for _ in range(self.n_blocks):
      layers = []
      for i in range(self.hidden_layers):
        if not i==0:
            layers.append(nn.SiLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
      self.blocks.append(
          nn.Sequential(*layers))
            # nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()))

    self.last_layer = nn.Linear(self.hidden_dim, self.output_dim)

  def forward(self, input):
    input = self.first_layer(input)
    for b in range(self.n_blocks):
      residual = self.blocks[b](input)
      input = input + residual
    return self.last_layer(input)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=1, device="cpu"):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for i in range(self.n_hidden_layers):
            x = self.relu(self.hidden_layers[i](x))
        x = self.output_layer(x)
        return x
    

# defining encoder and decoder
class brazy_encoder(nn.Module):
    def __init__(self, c_small=32, f1_dim=512, f2_dim=1024, input_dim=28*28, output_dim=28*28, batchnorm=True, third_conv=False, p_dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c1 = c_small
        self.c2 = 2*c_small
        if third_conv:
            self.c3 = 4*c_small
        else:
            self.c3 = self.c2
        self.f1_dim = f1_dim
        self.f2_dim = f2_dim
        self.batchnorm = batchnorm
        self.third_conv = third_conv
        self.p_dropout=p_dropout

        self.dropout = nn.Dropout(self.p_dropout)

        self.conv1 = nn.Conv2d(1, self.c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.c1, self.c2, kernel_size=3, padding=1)
        if third_conv:
            self.conv3 = nn.Conv2d(self.c2, self.c3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * self.c3, self.f1_dim)
        self.fc2 = nn.Linear(self.f1_dim, self.f2_dim)
        self.latent = nn.Linear(self.f2_dim, self.output_dim)

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.c1)
            self.batchnorm2 = nn.BatchNorm2d(self.c2)
        else:
            self.batchnorm1 = nn.Identity()
            self.batchnorm2 = nn.Identity()


    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = nn.functional.relu(self.batchnorm1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = nn.functional.relu(self.batchnorm2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        if self.third_conv:
            x = nn.functional.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        return self.latent(x)

class brazy_decoder(nn.Module):
    def __init__(self, c_small=32, f1_dim=512, f2_dim=1024, input_dim=28*28, output_dim=28*28, batchnorm=True, third_conv=True, p_dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c1 = c_small
        self.c2 = 2*c_small
        if third_conv:
            self.c3 = 4*c_small
        else:
            self.c3 = self.c2
        self.f1_dim = f1_dim
        self.f2_dim = f2_dim
        self.batchnorm = batchnorm
        self.third_conv = third_conv
        self.p_dropout = p_dropout

        self.dropout = nn.Dropout(self.p_dropout)

        self.fc2 = nn.Linear(self.input_dim, self.f2_dim)  # Opposite of Encoder's latent layer
        self.fc1 = nn.Linear(self.f2_dim, self.f1_dim)
        self.fc_to_conv = nn.Linear(self.f1_dim, 7 * 7 * self.c3)  # Convert back to the flattened conv output size

        if third_conv:
            self.deconv3 = nn.ConvTranspose2d(self.c3, self.c2, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.c2, self.c1, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(self.c1, 1, kernel_size=3, padding=1)

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.c1)
            self.batchnorm2 = nn.BatchNorm2d(self.c2)
        else:
            self.batchnorm1 = nn.Identity()
            self.batchnorm2 = nn.Identity()

    def forward(self, z):
        # Fully connected layers (reverse of encoder's)
        z = nn.functional.relu(self.fc2(z))
        z = self.dropout(z)
        z = nn.functional.relu(self.fc1(z))
        z = self.dropout(z)
        # Reshape back to a 3D tensor (for convolution layers)
        z = self.fc_to_conv(z)
        z = z.view(z.shape[0], self.c3, 7, 7)  # Reshape to match conv output

        # Upsample through transposed convolutions
        z = nn.functional.interpolate(z, scale_factor=2)  # Upsample to 14x14
        if self.third_conv:
            z = nn.functional.relu(self.deconv3(z))
        z = nn.functional.relu(self.deconv2(self.batchnorm2(z)))
        z = self.dropout(z)

        z = nn.functional.interpolate(z, scale_factor=2)  # Upsample to 28x28
        z = self.deconv1(self.batchnorm1(z))
        z = z.reshape(-1, 784)
        return z

class cond_conv_encoder(nn.Module):
    def __init__(self, c_small=32, f1_dim=512, f2_dim=1024, input_dim=28*28,
                 output_dim=28*28, batchnorm=True, third_conv=False, cond_dim=10,
                 p_dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c1 = c_small
        self.c2 = 2*c_small
        if third_conv:
            self.c3 = 4*c_small
        else:
            self.c3 = self.c2
        self.f1_dim = f1_dim
        self.f2_dim = f2_dim
        self.batchnorm = batchnorm
        self.third_conv = third_conv
        self.cond_dim = cond_dim
        self.p_dropout = p_dropout

        self.dropout = nn.Dropout(self.p_dropout)

        self.conv1 = nn.Conv2d(1, self.c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.c1, self.c2, kernel_size=3, padding=1)
        if third_conv:
            self.conv3 = nn.Conv2d(self.c2, self.c3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * self.c3 + self.cond_dim, self.f1_dim)
        self.fc2 = nn.Linear(self.f1_dim + self.cond_dim, self.f2_dim)
        #self.fc3 = nn.Linear(self.f2_dim + self.cond_dim, self.f2_dim)
        self.latent = nn.Linear(self.f2_dim + self.cond_dim, self.output_dim)

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.c1)
            self.batchnorm2 = nn.BatchNorm2d(self.c2)
        else:
            self.batchnorm1 = nn.Identity()
            self.batchnorm2 = nn.Identity()


    def forward(self, x, cond):
        x = x.reshape(-1, 1, 28, 28)
        x = self.dropout(x)
        x = nn.functional.relu(self.batchnorm1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = nn.functional.relu(self.batchnorm2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        if self.third_conv:
            x = nn.functional.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(self.add_cond_to_x(x, cond)))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(self.add_cond_to_x(x, cond)))
        x = self.dropout(x)
        #x = nn.functional.relu(self.fc3(self.add_cond_to_x(x, cond)))
        return self.latent(self.add_cond_to_x(x,cond))

    def add_cond_to_x(self, x, cond):
        assert cond.shape[-1] == self.cond_dim
        assert cond.shape[0] == x.shape[0]
        return torch.cat([x, cond], dim=-1)

class cond_conv_decoder(nn.Module):
    def __init__(self, c_small=32, f1_dim=512, f2_dim=1024, input_dim=28*28,
                 output_dim=28*28, batchnorm=True, third_conv=True, cond_dim=10,
                 p_dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c1 = c_small
        self.c2 = 2*c_small
        if third_conv:
            self.c3 = 4*c_small
        else:
            self.c3 = self.c2
        self.f1_dim = f1_dim
        self.f2_dim = f2_dim
        self.batchnorm = batchnorm
        self.third_conv = third_conv
        self.cond_dim = cond_dim
        self.p_dropout = p_dropout

        self.dropout = nn.Dropout(self.p_dropout)

        self.fc3 = nn.Linear(self.input_dim+self.cond_dim, self.f2_dim)
        #self.fc2 = nn.Linear(self.f2_dim+self.cond_dim, self.f2_dim)  # Opposite of Encoder's latent layer
        self.fc1 = nn.Linear(self.f2_dim+self.cond_dim, self.f1_dim)
        self.fc_to_conv = nn.Linear(self.f1_dim+self.cond_dim, 7 * 7 * self.c3)  # Convert back to the flattened conv output size

        if third_conv:
            self.deconv3 = nn.ConvTranspose2d(self.c3, self.c2, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.c2, self.c1, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(self.c1, 1, kernel_size=3, padding=1)

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.c1)
            self.batchnorm2 = nn.BatchNorm2d(self.c2)
        else:
            self.batchnorm1 = nn.Identity()
            self.batchnorm2 = nn.Identity()

    def forward(self, z, cond):
        # Fully connected layers (reverse of encoder's)
        z = self.dropout(z)
        z = nn.functional.relu(self.fc3(self.add_cond_to_z(z, cond)))
        #z = nn.functional.relu(self.fc2(self.add_cond_to_z(z, cond)))
        z = self.dropout(z)
        z = nn.functional.relu(self.fc1(self.add_cond_to_z(z, cond)))

        # Reshape back to a 3D tensor (for convolution layers)
        z = self.dropout(z)
        z = self.fc_to_conv(self.add_cond_to_z(z,cond))
        z = z.view(z.shape[0], self.c3, 7, 7)  # Reshape to match conv output

        # Upsample through transposed convolutions
        z = nn.functional.interpolate(z, scale_factor=2)  # Upsample to 14x14
        z = self.dropout(z)
        if self.third_conv:
            z = nn.functional.relu(self.deconv3(z))
        z = self.dropout(z)
        z = nn.functional.relu(self.deconv2(self.batchnorm2(z)))

        z = nn.functional.interpolate(z, scale_factor=2)  # Upsample to 28x28
        z = self.dropout(z)
        z = self.deconv1(self.batchnorm1(z))
        z = z.reshape(-1, 784)
        return z

    def add_cond_to_z(self, z, cond):
        assert cond.shape[-1] == self.cond_dim
        assert cond.shape[0] == z.shape[0]
        return torch.cat([z, cond], dim=-1)



#### FREE FORM FLOW ###
class FreeFormFlow(torch.nn.Module):
    def __init__(self, encoder, decoder, data_dims=2, device="cpu"):
        super().__init__()
        #self.encoder = MLP(input_dim, hidden_dim, output_dim, n_hidden_layers=n_hidden_layers)
        #self.decoder = MLP(output_dim, hidden_dim, input_dim, n_hidden_layers=n_hidden_layers)
        #self.encoder = resnet(input_dim=input_dim, hidden_dim=hidden_dim, n_blocks=n_blocks, output_dim=output_dim)
        #self.decoder = resnet(input_dim=input_dim, hidden_dim=hidden_dim, n_blocks=n_blocks, output_dim=output_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = encoder.input_dim
        self.output_dim = encoder.output_dim
        self.data_dims = data_dims
        self.device = device

    def forward(self, x_or_z, rev=False, cond=None):
        if rev: return self.decoder(x_or_z)
        else: return self.encoder(x_or_z)

    def sample(self, num_samples, cond=None):
        z = torch.normal(mean=torch.zeros((num_samples, self.input_dim)), std=torch.ones((num_samples, self.input_dim))).to(self.device)
        return self.decoder(z)[..., :self.data_dims]

    def logprob(self, x, exact=True, jac_of_enc=True, cond=None, verbose=False):
        if exact:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            z = self.encoder(x)
            z = z[..., :self.data_dims]
            latent_logprob = - 0.5*torch.sum(z**2, -1) - 0.5 * z.shape[1] * np.log(2*math.pi)
            if jac_of_enc:
                jac = vmap(jacrev(self.encoder))(x)
                jac = jac[:, :self.data_dims, :]
            else:
                jac = vmap(jacrev(self.decoder))(z)
                jac = jac[:, :self.data_dims, :]
            ljd = torch.linalg.slogdet(jac)
            if verbose: return latent_logprob + ljd[1], latent_logprob, ljd[1]
            return latent_logprob + ljd[1]

        else:
            #return l.nll_surrogate(x, self.encoder, self.decoder)[3]
            return nll_surrogate(x, self.encoder, self.decoder)[3]

    def ljd(self, x, cond=None):
      jac = vmap(jacrev(self.encoder))(x)
      jac = jac[:, :self.data_dims, :]
      ljd = torch.linalg.slogdet(jac)
      return ljd

#### CONDITIONAL FREE FORM FLOW ####
## undefined: l.nll_surrogate
## UPDATE (fixed:added above): nll_surrogate is defined in the cell under the section: loss.py
class condFreeFormFlow(torch.nn.Module):
    def __init__(self, encoder, decoder, cond_dim=10, data_dims=2, device="cpu"):
        super().__init__()
        #self.encoder = MLP(input_dim, hidden_dim, output_dim, n_hidden_layers=n_hidden_layers)
        #self.decoder = MLP(output_dim, hidden_dim, input_dim, n_hidden_layers=n_hidden_layers)
        #self.encoder = resnet(input_dim=input_dim, hidden_dim=hidden_dim, n_blocks=n_blocks, output_dim=output_dim)
        #self.decoder = resnet(input_dim=input_dim, hidden_dim=hidden_dim, n_blocks=n_blocks, output_dim=output_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = encoder.input_dim
        self.output_dim = encoder.output_dim
        self.data_dims = data_dims
        self.cond_dim = cond_dim
        self.device = device

    def forward(self, x_or_z, cond, rev=False):
        if rev: return self.decoder(x_or_z, cond)
        else: return self.encoder(x_or_z, cond)

    def sample(self, batchsize, cond):
        assert cond.shape[0] == batchsize
        assert cond.shape[1] == self.cond_dim
        num_samples = cond.shape[0]
        z = torch.normal(mean=torch.zeros((num_samples, self.input_dim)), std=torch.ones((num_samples, self.input_dim))).to(self.device)
        return self.decoder(z, cond)[..., :self.data_dims]

    def logprob(self, x, cond, exact=True, jac_of_enc=True):
        if exact:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            z = self.encoder(x)
            z = z[..., :self.data_dims]
            latent_logprob = - 0.5*torch.sum(z**2, -1) - 0.5 * z.shape[1] * np.log(2*math.pi)
            if jac_of_enc:
                jac = vmap(jacrev(self.encoder))(x)
                jac = jac[:, :self.data_dims, :]
            else:
                jac = vmap(jacrev(self.decoder))(z)
            ljd = torch.linalg.slogdet(jac)
            return latent_logprob + ljd[1]
        else:
            #return l.nll_surrogate(x, self.encoder, self.decoder)[3]
            return nll_surrogate(x, self.encoder, self.decoder)[3]


# this is some feature extractor
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Adjust to 3*3 after pooling
        self.fc2 = nn.Linear(256, 10)  # 10 classes for MNIST
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        # self.batchnorm1 = nn.Identity()
        # self.batchnorm2 = nn.Identity()
        # self.batchnorm3 = nn.Identity()
        self.fc1_bn = nn.BatchNorm1d(256)
        self.is_trained = False

    def forward(self, x, extract_layer=2):
        x = x.view(-1, 1, 28, 28)  # Reshape for Conv2d
        x = self.conv1(x)
        if extract_layer is not None and extract_layer == 0:
            x = x.view(x.size(0), -1)
            return x
        x = nn.functional.relu(self.batchnorm1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        if extract_layer == 1:
            x = x.view(x.size(0), -1)
            return x
        x = nn.functional.relu(self.batchnorm2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        if extract_layer == 2:
            x = x.view(x.size(0), -1)
            return x
        x = nn.functional.relu(self.batchnorm3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1_bn(self.fc1(x))
        if extract_layer == 3:
            return x
        x = self.dropout(nn.functional.relu(x))
        x = self.fc2(x)
        return x
