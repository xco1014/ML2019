import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import sys
patharg =sys.argv

train = np.load(patharg[1], allow_pickle=True)
task = "ae"
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


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
task ='ae'

if task == 'ae':
    num_epochs = 3
    batch_size = 128
    learning_rate = 1e-4

    #{'fcn', 'cnn', 'vae'} 
    model_type = "fcn" 

    x = train
    if model_type == 'fcn' or model_type == 'vae':
        x = x.reshape(len(x), -1)
        
    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
  

    model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE()}
    model = model_classes[model_type].cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate)
    
    best_loss = np.inf
    model.train()
    for epoch in range(num_epochs):
        for data in train_dataloader:
            if model_type == 'cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            output = model(img)
            if model_type == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================save====================
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model,patharg[2])
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

