import torch
from torch.autograd import Variable

from bars_data import sample_bars

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 4
num_samples = 1024
num_epochs = 1024

X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size,
    torch.ones(image_size) / image_size
  ).view(-1)
  for _ in range(num_samples)
])

encoder = torch.nn.Linear(image_size * image_size, 2 * image_size)
decoder = torch.nn.Linear(2 * image_size, image_size * image_size)

optimizer = torch.optim.SGD([
  {'params': encoder.parameters(), 'lr': 1e-4, 'momentum': 0.9},
  {'params': decoder.parameters(), 'lr': 1e-4, 'momentum': 0.9}
])

def debug(count):
  plt.figure(figsize=(12, 4))

  # True images
  for i in range(count):
    plt.subplot(2, count, i + 1)
    plt.imshow(X[i].view(image_size, image_size).numpy())
    plt.axis('off')

  # Reconstructed images
  for i in range(count):
    Xvar = Variable(X[i])
    fX = decoder(encoder(Xvar)).view(image_size, image_size)
    plt.subplot(2, count, count + i + 1)
    plt.imshow(fX.data.numpy())

    loss = torch.sum(torch.pow(fX - Xvar.view(image_size, image_size), 2))
    plt.title('{:6.4f}'.format(loss.data[0]))
    # plt.imshow(fX[i].numpy())
    plt.axis('off')

  plt.subplot(2, count, 1)
  plt.ylabel('true image')

  plt.subplot(2, count, count + 1)
  plt.ylabel('reconstructed')

  plt.show()

plot_interval = 1000

for i in range(num_epochs):
  Xvar = Variable(X)
  residual = decoder(encoder(Xvar)) - Xvar
  loss = torch.sum(torch.pow(residual, 2))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print('epoch', i, 'loss', loss.data[0])

  if i % plot_interval == 0:
    debug(8)
