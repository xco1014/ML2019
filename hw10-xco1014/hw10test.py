from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

task = "ae"
patharg =sys.argv
if os.path.split(patharg[2])[-1] == "best.pth":
    model_type = "fcn"
else:
    model_type = "vae"

test = np.load(patharg[1], allow_pickle=True)


class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 32 * 32 * 3
            ), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
                  nn.Conv2d(128, 32, 3, stride=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, stride=1),           # [batch, 48, 4, 4]
            nn.ReLU(),

        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 32, 3, stride=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 128, 3, stride=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 3, stride=1 ),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(32*32*3, 400)
        self.fc21 = nn.Linear(400, 3)
        self.fc22 = nn.Linear(400, 3)
        self.fc3 = nn.Linear(3, 400)
        self.fc4 = nn.Linear(400, 32*32*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return mse + KLD




if task == 'ae':
    if model_type == 'fcn' or model_type == 'vae':
        y = test.reshape(len(test), -1)
    else:
        y = test
    
    batch_size = 128
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    #test_dataset = ImgDataset(data,source_transform)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    model = torch.load(patharg[2], map_location='cuda')
    #pr_y = list()
    model.eval()
    reconstructed = list()
    for i, data in enumerate(test_dataloader): 
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
            #img = data.cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type == 'cnn':
            output = output.transpose(3, 1)
        elif model_type == 'vae':
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())
        #pr_y.append(img.cpu().detach().numpy())
    #pr_y = np.concatenate(pr_y, axis=0)
    reconstructed = np.concatenate(reconstructed, axis=0)
    
    
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))

    y_pred = anomality
    with open(patharg[3], 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))