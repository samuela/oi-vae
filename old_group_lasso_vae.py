import itertools

import torch
from torch.autograd import Variable

from lib.bars_data import sample_bars
from lib.common import LearnedTiedVarianceNet, FirstLayerSparseDecoder
# from lib.distributions import DiagonalMVN

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 8
dim_z = image_size // 2
num_samples = 1024
num_epochs = 1000000
mc_samples = 1
batch_size = 64

lam = 0.0

# Basic bars
# X = torch.stack([
#   sample_bars(
#     torch.ones(image_size) / image_size * 2,
#     torch.ones(image_size) / image_size * 0
#   ).view(-1)
#   for _ in range(num_samples)
# ])

# Shared bars
halfX = torch.stack([
  sample_bars(
    torch.ones(image_size // 2) / image_size * 2,
    torch.ones(image_size) / image_size * 0
  ).view(-1)
  for _ in range(num_samples)
])
X = torch.cat([halfX, halfX], dim=1)

# inference_net = ConstantVarianceNet(
#   torch.nn.Linear(image_size * image_size, 2 * image_size),
#   Variable(-10 * torch.ones(2 * image_size))
# )

# inference_net = LearnedVarianceNet(
#   torch.nn.Linear(image_size * image_size, 2 * image_size),
#   torch.nn.Linear(image_size * image_size, 2 * image_size)
# )
# inference_net.log_stddev_net.weight.data[:] = 0
# inference_net.log_stddev_net.bias.data[:] = -6

inference_net = LearnedTiedVarianceNet(
  torch.nn.Linear(image_size * image_size, dim_z),
  torch.nn.Linear(image_size * image_size, 1)
)
inference_net.log_stddev_net.weight.data[:] = 0
inference_net.log_stddev_net.bias.data[:] = -6

# generative_net = ConstantVarianceNet(
#   torch.nn.Linear(2 * image_size, image_size * image_size),
#   Variable(-3 * torch.ones(image_size * image_size))
# )

decoder = FirstLayerSparseDecoder(
  [torch.nn.Linear(1, image_size, bias=False) for _ in range(image_size)],
  [1 for _ in range(image_size)],
  dim_z
)
generative_net = LearnedTiedVarianceNet(
  decoder,
  torch.nn.Linear(dim_z, 1)
)
generative_net.log_stddev_net.weight.data[:] = 0
generative_net.log_stddev_net.bias.data[:] = -2

z_prior = DiagonalMVN(
  Variable(torch.zeros(dim_z)),
  Variable(torch.zeros(dim_z))
)

# lr = 1e1 / 4096
# momentum = 0.7
# optimizer = torch.optim.SGD([
#   {'params': inference_net.parameters(), 'lr': lr, 'momentum': momentum},
#   {'params': generative_net.parameters(), 'lr': lr, 'momentum': momentum}
# ])

# from ASGD import ASGD
# lr = 1e1 / 4096
# optimizer = ASGD([
#   {'params': inference_net.parameters(), 'lr': lr, 'cnh': 10.0, 'skapRatio': 2},
#   {'params': generative_net.parameters(), 'lr': lr, 'cnh': 10.0, 'skapRatio': 2}
# ])

lr = 1e-2
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
  {'params': generative_net.parameters(), 'lr': lr}
])

def debug(count):
  fig, ax = plt.subplots(3, count, figsize=(12, 4))

  # True images
  for i in range(count):
    ax[0, i].imshow(X[i].view(image_size, image_size).numpy())
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])

  # latent representation
  for i in range(count):
    Xvar = Variable(X[i])
    ax[1, i].bar(range(dim_z), inference_net(Xvar).mean.data.numpy())
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i in range(count):
    Xvar = Variable(X[i])
    fX = generative_net(inference_net(Xvar).mean).mean.view(image_size, image_size)
    ax[2, i].imshow(fX.data.numpy())
    ax[2, i].axes.xaxis.set_ticks([])
    ax[2, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('z')
  ax[2, 0].set_ylabel('reconstructed')

  return fig

def debug_incoming_weights():
  fig, ax = plt.subplots(1, image_size, figsize=(12, 4))

  # See https://matplotlib.org/examples/color/colormaps_reference.html
  cmap = 'bwr'
  for i, m in enumerate(decoder.latent_to_group_maps):
    ax[i].imshow(torch.stack([m.weight.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('group {}'.format(i))
    ax[i].set_xlabel('z_i')
    ax[i].axes.xaxis.set_ticks(range(dim_z))
    ax[i].axes.yaxis.set_ticks([])

  ax[0].set_ylabel('learned weights')

  return fig

def debug_outgoing_weights():
  fig, ax = plt.subplots(1, dim_z, figsize=(12, 4))

  # rows correspond to groups and cols correspond to z_i's
  col_norms = torch.stack([
    torch.sqrt(torch.sum(torch.pow(m.weight.data, 2), dim=0))
    for m in decoder.latent_to_group_maps
  ])

  # See https://matplotlib.org/examples/color/colormaps_reference.html
  cmap = 'bwr'
  for i in range(dim_z):
    ax[i].imshow(torch.stack([col_norms[:, i] for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('z_{}'.format(i))
    ax[i].set_xlabel('groups')
    ax[i].axes.xaxis.set_ticks(range(image_size))
    ax[i].axes.yaxis.set_ticks([])

  return fig

plot_interval = 500

for i in range(num_epochs):
  # loss = 0
  total_kl = 0
  total_loglik = 0
  for j in torch.randperm(num_samples)[:batch_size]:
    Xvar = Variable(X[j])
    vi_posterior = inference_net(Xvar)
    loglik_term = sum(
      generative_net(vi_posterior.sample()).logprob(Xvar)
      # generative_net(vi_posterior.mean).logprob(Xvar)
      for _ in range(mc_samples)
    )
    kl = KL_DiagonalMVNs(vi_posterior, z_prior)

    total_kl += kl
    total_loglik += loglik_term / mc_samples

    # elbo = -kl + loglik_term / mc_samples
    # loss -= elbo

  # loss /= batch_size
  loss = -1 * (-total_kl + total_loglik) / batch_size

  if i % plot_interval == 0 and i > 0:
    debug(8)
    plt.suptitle('Group Lasso VAE, Adam, Iteration {}, lr = {}, lam = {}, batch_size = {}, num_samples = {}'.format(i, lr, lam, batch_size, num_samples))

    debug_incoming_weights()
    plt.suptitle('incoming z weights')

    debug_outgoing_weights()
    plt.suptitle('outgoing z weight norms')

    plt.show()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  decoder.proximal_step(lr * lam)

  print('iter', i)
  print('  neg. kl:        ', -total_kl.data[0] / batch_size)
  print('  log lik.:       ', total_loglik.data[0] / batch_size)
  print('  cumulative ELBO:', -loss.data[0])
  print('  regularization: ', -lam * decoder.group_lasso_penalty().data[0])
  print('  total objective:', -loss.data[0] - lam * decoder.group_lasso_penalty().data[0])
  # print(-kl.data[0], loglik_term.data[0] / mc_samples)
