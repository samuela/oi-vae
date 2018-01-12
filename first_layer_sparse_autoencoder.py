import itertools

import torch
from torch.autograd import Variable

from bars_data import sample_bars

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 8
dim_z = image_size
num_samples = 1024
num_epochs = 100000

# lam = 1e0 / 128
lam = 0

X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size * 2,
    torch.zeros(image_size) / image_size
  ).view(-1)
  for _ in range(num_samples)
])

encoder = torch.nn.Linear(image_size * image_size, dim_z)
# decoder = torch.nn.Linear(dim_z, image_size * image_size)

class FirstLayerSparseDecoder(object):
  def __init__(self, group_generators, group_generators_input_dims, input_dim):
    assert len(group_generators) == len(group_generators_input_dims)
    self.group_generators = group_generators
    self.group_generators_input_dims = group_generators_input_dims
    self.input_dim = input_dim

    # The (hopefully sparse) mappings from the latent z to the inputs to each of
    # the group generators.
    self.latent_to_group_maps = [
      torch.nn.Linear(self.input_dim, k)
      for k in self.group_generators_input_dims
    ]

  def __call__(self, z):
    return torch.cat(
      [gen(m(z))
       for gen, m in zip(self.group_generators, self.latent_to_group_maps)],
      dim=-1
    )

  def parameters(self):
    return itertools.chain(
      *[gen.parameters() for gen in self.group_generators],
      *[m.parameters() for m in self.latent_to_group_maps]
    )

  def group_lasso_penalty(self):
    return sum([
      torch.sum(torch.sqrt(torch.sum(torch.pow(m.weight, 2), dim=0)))
      for m in self.latent_to_group_maps
    ])

decoder = FirstLayerSparseDecoder(
  [torch.nn.Linear(1, image_size)
   for _ in range(image_size)],
  [1 for _ in range(image_size)],
  dim_z
)

lr = 1e0 / 32
optimizer = torch.optim.SGD([
  {'params': encoder.parameters(), 'lr': lr, 'momentum': 0.9},
  {'params': decoder.parameters(), 'lr': lr, 'momentum': 0.9}
])

def debug(ixs):
  fig, ax = plt.subplots(2, len(ixs), figsize=(12, 4))

  # True images
  for i, ix in enumerate(ixs):
    ax[0, i].imshow(X[ix].view(image_size, image_size).numpy(), vmin=0, vmax=1)
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])
    # ax[0, i].set_title(i)

  # Reconstructed images
  for i, ix in enumerate(ixs):
    Xvar = Variable(X[ix])
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
  reconstructed = decoder(encoder(Xvar))
  if torch.sum(torch.abs(reconstructed - reconstructed[0])).data[0] / num_samples <= 1e-3:
    print('solution has collapsed!')

  residual = reconstructed - Xvar
  reconstruction_error = torch.sum(torch.pow(residual, 2)) / num_samples
  sparsity_penalty = lam * decoder.group_lasso_penalty() / (image_size ** 2)
  loss = reconstruction_error + sparsity_penalty
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print('epoch', i, 'loss', loss.data[0])

  if i % plot_interval == 0:
    for m in decoder.latent_to_group_maps:
      print(m.weight)

    debug([0, 1, 4, 7, 8, 9, 12, 25])
    plt.suptitle('Iteration {}, lambda = {}'.format(i, lam))
    plt.show()
