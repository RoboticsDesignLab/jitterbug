import os
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as fn

from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import datasets, transforms, utils

import scipy.io as sio
import numpy as np

class Encoder(nn.Module):
    """VAE Encoder - maps inputs to mean and covariance vectors"""

    def __init__(self, data_size=784, latent_size=2):
        """c-tor
​
        Args:
            data_size (int): Size of the input data vectors
            latent_size (int): Size of the latent dimension
        """

        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(data_size, int(data_size // 4))
        self.fc2 = nn.Linear(int(data_size // 4), int(data_size // 16))

        self.out_mean = nn.Linear(int(data_size // 16), latent_size)
        self.out_std = nn.Linear(int(data_size // 16), latent_size)

    def forward(self, x):
        """Encode data to latent mean and covariance vectors
​
        Args:
            x (torch.Tensor): Input data tensor
​
        Returns:
            (torch.Tensor): Output latent mean vector
            (torch.Tensor): Output latent log standard deviation vector
        """
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))

        return self.out_mean(x), self.out_std(x)


class Decoder(nn.Module):
    """VAE Decoder - maps latent samples to data estimates"""

    def __init__(self, latent_size=2, data_size=784):
        """c-tor
​
        Args:
            latent_size (int): Size of the input latent dimension
            data_size (int): Size of the output data vectors
        """

        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_size, int(data_size // 16))
        self.fc2 = nn.Linear(int(data_size // 16), int(data_size // 4))

        self.out = nn.Linear(int(data_size // 4), data_size)

    def forward(self, x):
        """Encode data to latent mean and covariance vectors
​
        Args:
            x (torch.Tensor): Input latent tensor
​
        Returns:
            (torch.Tensor): Output reconstruction data vector
        """
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = self.out(x)

        return torch.sigmoid(x)


class VAE(nn.Module):
    """Variational AutoEncoder implementation"""

    def __init__(self, data_size=784, latent_size=2):
        """c-tor
​
        Args:
            data_size (int): Size of the input data vectors
            latent_size (int): Size of the latent dimension
        """

        super(VAE, self).__init__()

        self.enc = Encoder(data_size=data_size, latent_size=latent_size)
        self.dec = Decoder(latent_size=latent_size, data_size=data_size)

    def forward(self, x):
        """Forward pass
​
        Args:
            x (torch.Tensor): Input data
​
        Returns:
            (torch.Tensor): Reconstructed image
            (torch.Tensor): Latent mean vector
            (torch.Tensor): Latent log variance vector
        """

        # Encode to latent mean and log variance
        mean, logvar = self.enc(x)
        std = torch.exp(0.5 * logvar)

        # Re-parameterize and sample
        eps = torch.randn_like(std)
        z = mean + std * eps

        # Decode
        x_hat = self.dec(z)

        return x_hat, mean, logvar

    def decode(self, z):
        """Decode a latent sample
​
        Args:
            z (torch.Tensor): Latent sample
​
        Returns:
            (torch.Tensor): Decoded data vector
        """
        return self.dec(z)

    def encode(self,x):
        """Encode an input

        Args:
            x (torch.Tensor): Input

        Returns:
            (torch.Tensor): Encoded data vector
        """
        mean, std = self.enc(x)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z.data.numpy()


    def loss_fn(self, x, x_hat, mean, logvar):
        """Get VAE loss
​
        Args:
            x (torch.Tensor): Ground truth input
            x_hat (torch.Tensor): Reconstructed output
            mean (torch.Tensor): Latent mean vector
            logvar (torch.Tensor): Latent log variance vector
​
        Returns:
            (torch.Tensor): Loss
        """

        # Compute reconstruction loss (BCE)
        loss_bce = fn.binary_cross_entropy(x_hat, x, reduction='sum')

        # Compute regularizing loss (KL divergence)
        loss_kld = -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp())

        return loss_bce + loss_kld

    def save_autoencoder(self, save_path):
        """Save the autoencoder model.

        Args:
            save_path (str): path where to save the autoencoder"""
        torch.save({'model_state_dict': self.state_dict()},
                   save_path)
        print("Model saved in path: %s" % save_path)

    def load_autoencoder(self,save_path):
        """Use the trained encoder saved in the file "saved_path" to reduce the input dimension

          Args:
             save_path (str): path where the file containing the autoencoder model is saved"""

        print("Load autoencoder...")
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint)
        self.eval()
        print("Autoencoder loaded.")



def segment_data(data, h):
    """Segment input and target data for DDAE
    Args:
        data (numpy array): Nx16 Numpy array of input data
        h (int): Dynamics horizon, 0 corresponds to a regular denoising AE
    Returns:
        (numpy array): Input data
        (numpy array): Target data
    """

    assert h >= 0,\
        "Dynamics horizon must be h >= 0, but was {}".format(h)

    # Apply dynamics horizon offset to segment training and testing data
    if h == 0:
        input = data.copy()
        target = data.copy()
    else:
        input = data[0:-h, :].copy()
        target = data[h:].copy()

    return input, target

def main():
    """Main function"""

    use_gpu = False
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_gpu
        else "cpu"
    )
    batch_size = 256
    seed = 1337
    latent_size = 15
    log_interval = 20
    h = 2

    # Set seed
    torch.manual_seed(seed)

    # Load dataset
    # trf = transforms.Compose([
    #     transforms.ToTensor(),
    #     #transforms.Normalize((0.1307,), (0.3081,)),
    #     transforms.Lambda(lambda t: t.flatten())
    # ])
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(".", train=True, transform=trf, download=True),
    #     shuffle=True,
    #     batch_size=batch_size,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(".", train=False, transform=trf, download=True),
    #     shuffle=True,
    #     batch_size=batch_size,
    # )

    print("Loading data...")
    mat = sio.loadmat('observations_mid_random_normalized.mat')
    dataset = mat['observations']

    # inputs, targets = segment_data(data=dataset,
    #                                h=h
    #                                )
    # N = len(dataset)
    # splitting_percentage = 0.7
    #
    # # Split the data into training and testing sets
    # splitting_int = int(round(splitting_percentage * N, 0))
    # training_data = inputs[:splitting_int]
    # training_labels = targets[:splitting_int]
    # testing_data = inputs[splitting_int:]
    # testing_labels = targets[splitting_int:]
    #
    # training_data_tensor = torch.utils.data.TensorDataset(torch.stack([torch.Tensor(i) for i in training_data]),
    #                                                       torch.stack([torch.Tensor(i) for i in training_labels])
    #                                                       )
    # testing_data_tensor = torch.utils.data.TensorDataset(torch.stack([torch.Tensor(i) for i in testing_data]),
    #                                                      torch.stack([torch.Tensor(i) for i in testing_labels])
    #                                                      )
    #
    # train_loader = torch.utils.data.DataLoader(training_data_tensor,
    #                                            shuffle=True,
    #                                            batch_size=batch_size,
    #                                            )
    # test_loader = torch.utils.data.DataLoader(testing_data_tensor,
    #                                           shuffle=True,
    #                                           batch_size=batch_size,
    #                                           )
    print("Data loaded.")
    # print(train_loader)


    mdl = VAE(data_size= len(dataset[0]),
              latent_size=latent_size).to(device)
    optimizer = opt.Adam(mdl.parameters(), lr=1e-3)

    mdl.load_autoencoder("./VAE.pt")
    datapoint = dataset[0]
    print(mdl.encode(torch.Tensor(datapoint)))

    #print("Load autoencoder...")
    #checkpoint = torch.load("./VAE.pt")
    #mdl.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint)
    #mdl.eval()
    #print("Autoencoder loaded.")

    # try:
    #     os.mkdir("FashionMNIST-results")
    # except FileExistsError:
    #     pass
    #
    # for epoch in range(5):
    #
    #     # Train model
    #     mdl.train()
    #     train_loss = 0
    #     #for el in enumerate(train_loader):
    #     #    print(el)
    #
    #     for batch_idx, (data, _) in enumerate(train_loader):
    #         #data = data[0]
    #         data.to(device)
    #         optimizer.zero_grad()
    #
    #         # data[0].to(device)
    #         # optimizer.zero_grad()
    #         # print(data[0])
    #
    #         x_hat, mean, logvar = mdl(data)
    #         loss = mdl.loss_fn(data, x_hat, mean, logvar)
    #         loss.backward()
    #         train_loss += loss.item()
    #         optimizer.step()
    #
    #         if batch_idx % log_interval == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch,
    #                 batch_idx * len(data),
    #                 len(train_loader.dataset),
    #                 100.0 * batch_idx / len(train_loader),
    #                 loss.item() / len(data)
    #             ))
    #
    #     print('====> Epoch: {} Average loss: {:.4f}'.format(
    #          epoch,
    #          train_loss / len(train_loader.dataset)
    #      ))
    #
    #     # Test model
    #     mdl.eval()
    #     test_loss = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, _) in enumerate(test_loader):
    #             #data = data[0]
    #             data = data.to(device)
    #             x_hat, mean, logvar = mdl(data)
    #             loss = mdl.loss_fn(data, x_hat, mean, logvar)
    #             test_loss += loss
    #
    # mdl.save_autoencoder(save_path="./VAEh2.pt")
        #         # Save a comparison of the reconstruction
        #         if batch_idx == 0:
        #             num_rows = 8
        #             comparison = torch.cat([
        #                 data.view(-1, 1, 28, 28)[:num_rows],
        #                 x_hat.view(-1, 1, 28, 28)[:num_rows]
        #             ])
        #             utils.save_image(
        #                 comparison.cpu(),
        #                 f'FashionMNIST-results/reconstruction_{epoch}.png',
        #                 nrow=num_rows
        #             )
        #
        # # Visualise latent space
        # with torch.no_grad():
        #     sample = torch.randn(64, latent_size).to(device)
        #     sample = mdl.decode(sample).cpu()
        #     utils.save_image(
        #         sample.view(64, 1, 28, 28),
        #         f'FashionMNIST-results/sample_{epoch}.png'
        #     )



if __name__ == '__main__':
    main()
