import torch
from torch import nn


class Vae(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dim=200):
        super().__init__()

        # encoder
        self.input_2_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_2_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_2_sigma = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.latent_2_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_2_output = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        mu, sigma = self._encode(x)
        eps = torch.randn_like(sigma)
        z_reparametrized = mu + eps * sigma
        x_reconstructed = self._decode(z_reparametrized)
        return x_reconstructed, mu, sigma

    def _encode(self, x):
        h = self.relu(self.input_2_hidden(x))
        mu, sigma = self.hidden_2_mu(h), self.hidden_2_sigma(h)
        return mu, sigma

    def _decode(self, z):
        h = self.relu(self.latent_2_hidden(z))
        return torch.sigmoid(self.hidden_2_output(h))


if __name__ == "__main__":
    model = Vae(784)
    x_reconstructed, mu, sigma = model(torch.randn(4, 28 * 28))
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
