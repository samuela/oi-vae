import itertools
import math

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from lib.bars_data import sample_bars, sample_bars_one


torch.manual_seed(0)

image_size = 8
dim_z = image_size
num_samples = 1024
num_epochs = 100000

lam = 0.5
# lam = 0

X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size * 2,
    torch.ones(image_size) / image_size * 0
  ).view(-1)
  for _ in range(num_samples)
])

# X = torch.stack([
#   sample_bars_one(image_size).view(-1)
#   for _ in range(num_samples)
# ])

encoder = torch.nn.Linear(image_size * image_size, dim_z)
# decoder = torch.nn.Linear(dim_z, image_size * image_size)

class LastLayerSparseDecoder(object):
  def __init__(
      self,
      z_generators,
      z_generators_output_dims,
      group_dims,
      input_dim
  ):
    assert len(z_generators) == len(z_generators_output_dims)
    self.z_generators = z_generators
    self.z_generators_output_dims = z_generators_output_dims
    self.group_dims = group_dims
    self.input_dim = input_dim

    # The (hopefully sparse) mappings from z_generator outputs to each of the
    # observed group variables.
    self.z_gen_to_group_maps = [
      [torch.nn.Linear(a, b, bias=False) for b in self.group_dims]
      for a in self.z_generators_output_dims
    ]

    self.group_biases = [
      Variable(
        # torch.Tensor(d).uniform_(-1. / math.sqrt(d), 1. / math.sqrt(d)),
        torch.zeros(d),
        requires_grad=True
      )
      for d in self.z_generators_output_dims
    ]

  def __call__(self, z):
    z_gen_outs = [z_gen(z[:, [i]]) for i, z_gen in enumerate(self.z_generators)]

    return torch.cat(
      [(sum(self.z_gen_to_group_maps[i][g](out)
            for i, out in enumerate(z_gen_outs)) +
        self.group_biases[g])
       for g in range(len(self.group_dims))],
      dim=-1
    )

  def parameters(self):
    return itertools.chain(
      *[z_gen.parameters() for z_gen in self.z_generators],
      *[m.parameters() for l in self.z_gen_to_group_maps for m in l],
      # *self.group_biases ######XXX
    )

  def group_lasso_penalty(self):
    return sum([
      torch.sqrt(torch.sum(torch.pow(m.weight, 2)))
      for l in self.z_gen_to_group_maps
      for m in l
    ])

  def proximal_step(self, t):
    for g in self.z_gen_to_group_maps:
      for m in g:
        norm = math.sqrt(torch.sum(torch.pow(m.weight.data, 2)))
        if norm <= t:
          m.weight.data.zero_()
        else:
          m.weight.data.mul_(1.0 - t / norm)

decoder = LastLayerSparseDecoder(
  # The z_generators
  [torch.nn.Linear(1, 1) for _ in range(image_size)],

  # z_generators output dimensions
  [1 for _ in range(image_size)],

  # Observed group sizes
  [image_size for _ in range(image_size)],

  # input dimension
  dim_z
)

def hardcode_model():
  baseline_val = 0.5

  encoder.bias.data[:] = 0
  for i in range(image_size):
    A = torch.zeros(image_size, image_size)
    A[i, :] = 1
    encoder.weight.data[i, :] = A.view(-1) / image_size / baseline_val

    # These just feed straight through
    for z_gen in decoder.z_generators:
      z_gen.weight.data[:] = 1
      z_gen.bias.data[:] = 0

    # The only group that counts is the one for this particular z
    assert dim_z == image_size
    for i, mappings in enumerate(decoder.z_gen_to_group_maps):
      for m in mappings:
        m.weight.data[:] = 0

      mappings[i].weight.data[:] = baseline_val

# hardcode_model()

lr = 1e0 / 64
optimizer = torch.optim.SGD([
  {'params': encoder.parameters(), 'lr': lr, 'momentum': 0.0},
  {'params': decoder.parameters(), 'lr': lr, 'momentum': 0.0}
])

def debug(ixs):
  fig, ax = plt.subplots(2, len(ixs), figsize=(12, 4))

  # True images
  for i, ix in enumerate(ixs):
    Xvar = Variable(torch.unsqueeze(X[ix], 0))
    ax[0, i].imshow(X[ix].view(image_size, image_size).numpy(), vmin=0, vmax=1)
    # ax[0, i].set_title('{:6.4f}'.format(encoder(Xvar)[2].data[0]))
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])
    # ax[0, i].set_title(i)

  # Reconstructed images
  for i, ix in enumerate(ixs):
    Xvar = Variable(torch.unsqueeze(X[ix], 0))
    fX = decoder(encoder(Xvar)).view(image_size, image_size)
    ax[1, i].imshow(fX.data.numpy(), vmin=0, vmax=1)
    loss = torch.sum(torch.pow(fX - Xvar.view(image_size, image_size), 2))
    ax[1, i].set_title('{:6.4f}'.format(loss.data[0]))
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('reconstructed')

  return fig

def debug2():
  plt.figure(figsize=(12, 4))
  for i in range(dim_z):
    plt.subplot(1, dim_z, i + 1)
    z = torch.zeros(dim_z)
    z[i] = 1
    zvar = Variable(torch.unsqueeze(z, 0))
    plt.imshow(decoder(zvar).view(image_size, image_size).data.numpy(), vmin=0, vmax=1)
    plt.title('Component {}'.format(i))

  plt.colorbar()

plot_interval = 1000

for i in range(num_epochs):
  Xvar = Variable(X)
  reconstructed = decoder(encoder(Xvar))
  if torch.sum(torch.abs(reconstructed - reconstructed[0])).data[0] / num_samples <= 1e-3:
    print('solution has collapsed!')

  residual = reconstructed - Xvar
  reconstruction_error = torch.sum(torch.pow(residual, 2)) / num_samples
  sparsity_penalty = lam * decoder.group_lasso_penalty()
  loss = reconstruction_error
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  decoder.proximal_step(lr * lam)
  print('epoch', i)
  print('  reconstruction loss:', reconstruction_error.data[0])
  print('  regularization:     ', sparsity_penalty.data[0])
  print('  combined loss:      ', (reconstruction_error + sparsity_penalty).data[0])

  if i % plot_interval == 0:
    # print('### decoder.latent_to_group_maps parameters')
    # for m in decoder.latent_to_group_maps:
    #   print(m.weight)

    # debug([0, 1, 4, 7, 8, 9, 12, 25])
    debug([0, 1, 2, 3, 4, 5, 6, 7])
    plt.suptitle('Iteration {}, lambda = {}'.format(i, lam))

    debug2()

    plt.show()
