"""This is a very simple linear autoencoder model running on the bars data."""

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from lib.bars_data import sample_bars


torch.manual_seed(0)

image_size = 8
dim_z = image_size
num_samples = 1024
num_epochs = 100000

X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size * 2,
    torch.zeros(image_size) / image_size
  ).view(-1)
  for _ in range(num_samples)
])

encoder = torch.nn.Linear(image_size * image_size, dim_z)
decoder = torch.nn.Linear(dim_z, image_size * image_size)

lr = 1e0 / 32
optimizer = torch.optim.SGD([
  {'params': encoder.parameters(), 'lr': lr, 'momentum': 0.9},
  {'params': decoder.parameters(), 'lr': lr, 'momentum': 0.9}
])

def debug(count):
  fig, ax = plt.subplots(2, count, figsize=(12, 4))

  # True images
  for i in range(count):
    ax[0, i].imshow(X[i].view(image_size, image_size).numpy(), vmin=0, vmax=1)
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i in range(count):
    Xvar = Variable(X[i])
    fX = decoder(encoder(Xvar)).view(image_size, image_size)
    ax[1, i].imshow(fX.data.numpy(), vmin=0, vmax=1)
    loss = torch.sum(torch.pow(fX - Xvar.view(image_size, image_size), 2))
    ax[1, i].set_title('{:6.4f}'.format(loss.data[0]))
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('reconstructed')

  return fig

plot_interval = 1000

for i in range(num_epochs):
  Xvar = Variable(X)
  residual = decoder(encoder(Xvar)) - Xvar
  loss = torch.sum(torch.pow(residual, 2)) / num_samples
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print('epoch', i, 'loss', loss.data[0])

  if i % plot_interval == 0:
    debug(12)
    plt.suptitle('Iteration {}'.format(i))
    plt.show()
