import itertools

import torch
from torch.autograd import Variable

from bars_data import sample_bars
from common import KL_DiagonalMVNs, LearnedTiedVarianceNet
from distributions import DiagonalMVN

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 8
dim_z = image_size
num_samples = 1024
num_epochs = 1000000
mc_samples = 1
batch_size = 64

X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size * 2,
    torch.ones(image_size) / image_size * 0
  ).view(-1)
  for _ in range(num_samples)
])

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

generative_net = LearnedTiedVarianceNet(
  torch.nn.Linear(dim_z, image_size * image_size),
  torch.nn.Linear(dim_z, 1)
)
generative_net.log_stddev_net.weight.data[:] = 0
generative_net.log_stddev_net.bias.data[:] = -2

z_prior = DiagonalMVN(
  Variable(torch.zeros(dim_z)),
  Variable(torch.zeros(dim_z))
)

# lr = 1e0 / 4096
# momentum = 0.9
# optimizer = torch.optim.SGD([
#   {'params': inference_net.parameters(), 'lr': lr, 'momentum': momentum},
#   {'params': generative_net.parameters(), 'lr': lr, 'momentum': momentum}
# ])

lr = 1e-3
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
  {'params': generative_net.parameters(), 'lr': lr}
])

def debug(count):
  fig, ax = plt.subplots(2, count, figsize=(12, 4))

  # True images
  for i in range(count):
    ax[0, i].imshow(X[i].view(image_size, image_size).numpy())
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i in range(count):
    Xvar = Variable(X[i])
    fX = generative_net(inference_net(Xvar).mean).mean.view(image_size, image_size)
    ax[1, i].imshow(fX.data.numpy())
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('reconstructed')

  return fig

plot_interval = 250

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
    # plt.suptitle('VAE, Iteration {}, lr = {}, momentum = {}, batch_size = {}, num_samples = {}'.format(i, lr, momentum, batch_size, num_samples))
    plt.suptitle('VAE, Adam, Iteration {}, lr = {}, batch_size = {}, num_samples = {}'.format(i, lr, batch_size, num_samples))
    plt.show()

  # scheduler.step()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print('iter', i)
  print('  neg. kl:        ', -total_kl.data[0] / batch_size)
  print('  log lik.:       ', total_loglik.data[0] / batch_size)
  print('  cumulative ELBO:', -loss.data[0])
  # print(-kl.data[0], loglik_term.data[0] / mc_samples)
