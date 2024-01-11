import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from model import Vae

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}")
INPUT_DIM = 784
LATENT_DIM = 20
HIDDEN_DIM = 200
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Vae(INPUT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.to(DEVICE).view(x.shape[0], 28 * 28)
        x_reconstructed, mu, sigma = model(x)

        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)

        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
