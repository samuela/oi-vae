"""Bayesian group lasso in a VAE setup on the bars data.

* Point estimates on the sparse weight parameters.
* Proximal gradient steps are taken to handle the group lasso penalty.
"""

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from lib.bars_data import (sample_bars_image, sample_many_bars_images,
                           sample_one_bar_image)
from lib.distributions import Normal
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.oivae import NormalPriorTheta, OIVAE
from lib.utils import Lambda

torch.manual_seed(0)

image_size = 8
dim_z = image_size
dim_x = image_size * image_size
num_groups = image_size
group_input_dim = 1

prior_theta_scale = 1
lam = 1
lam_adjustment = 1

num_train_samples = 2048
num_epochs = 20000
mc_samples = 1
batch_size = 64

# Basic bars
# X = sample_many_bars_images(num_train_samples, image_size, 2, 0)

# One bar per image
X = torch.stack([
  sample_one_bar_image(image_size).view(-1)
  for _ in range(num_train_samples)
])
X += 0.05 * torch.randn(X.size())

# Shared bars
# halfX = torch.stack([
#   sample_bars_image(
#     torch.ones(image_size // 2) / image_size * 2,
#     torch.ones(image_size) / image_size * 0
#   ).view(-1)
#   for _ in range(num_train_samples)
# ])
# X = torch.cat([halfX, halfX], dim=1)

# This value adjusts the impact of our learned variances in the sigma_net of
# `inference_net` below. Zero means that the model has no actual connection to
# the output and therefore the standard deviation defaults to the minimum. One
# means that we're learning the real model. This value is flipped to 1 after
# some number of iterations.
stddev_multiple = 0.1

# inference_net_log_stddev = Variable(
#   torch.log(1e-2 * torch.ones(dim_z)),
#   requires_grad=True
# )
inference_net = NormalNet(
  mu_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_x, dim_z)
  ),

  # Fixed standard deviation
  # sigma_net=Lambda(lambda x: 1e-3 * Variable(torch.ones(x.size(0), dim_z)))

  # Learned constant standard deviation
  # sigma_net=Lambda(
  #   lambda x: torch.exp(inference_net_log_stddev.expand(x.size(0), -1)) + 1e-3
  # )

  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_x, dim_z),
    Lambda(torch.exp),
    Lambda(lambda x: x * stddev_multiple + 1e-3)
  )
)

def make_group_generator():
  # Note that this Variable is NOT going to show up in `net.parameters()` and
  # therefore it is implicitly free from the ridge penalty/p(theta) prior.
  log_sigma = Variable(
    torch.log(1e-2 * torch.ones(image_size)),
    requires_grad=True
  )
  return NormalNet(
    mu_net=torch.nn.Linear(group_input_dim, image_size),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )

generative_net = BayesianGroupLassoGenerator(
  group_generators=[make_group_generator() for _ in range(image_size)],
  group_input_dim=group_input_dim,
  dim_z=dim_z
)

def debug(count):
  """Create a plot showing the first `count` training samples along with their
  mean z value, x mean, x standard deviation, and a sample from the full model
  (sample z and then sample x)."""
  fig, ax = plt.subplots(5, count, figsize=(12, 4))

  # True images
  for i in range(count):
    ax[0, i].imshow(X[i].view(image_size, image_size).numpy())
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])

  # latent representation
  for i in range(count):
    Xvar = Variable(X[[i]])
    ax[1, i].bar(range(dim_z), inference_net(Xvar).mu.data.squeeze().numpy())
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i in range(count):
    Xvar = Variable(X[[i]])
    # fX = generative_net(inference_net(Xvar).mu).mu.view(image_size, image_size)
    fX = generative_net(inference_net(Xvar).mu).mu.view(image_size, image_size)
    ax[2, i].imshow(fX.data.squeeze().numpy())
    ax[2, i].axes.xaxis.set_ticks([])
    ax[2, i].axes.yaxis.set_ticks([])

  for i in range(count):
    Xvar = Variable(X[[i]])
    # fX = generative_net(inference_net(Xvar).mu).mu.view(image_size, image_size)
    fX = generative_net(inference_net(Xvar).mu).sigma.view(image_size, image_size)
    ax[3, i].imshow(fX.data.squeeze().numpy())
    ax[3, i].axes.xaxis.set_ticks([])
    ax[3, i].axes.yaxis.set_ticks([])

  for i in range(count):
    Xvar = Variable(X[[i]])
    # fX = generative_net(inference_net(Xvar).mu).mu.view(image_size, image_size)
    fX = generative_net(inference_net(Xvar).sample()).sample().view(image_size, image_size)
    ax[4, i].imshow(fX.data.squeeze().numpy())
    ax[4, i].axes.xaxis.set_ticks([])
    ax[4, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('z')
  ax[2, 0].set_ylabel('x mu')
  ax[3, 0].set_ylabel('x sigma')
  ax[4, 0].set_ylabel('x sample')

  return fig

def debug_incoming_weights():
  fig, ax = plt.subplots(1, image_size, figsize=(12, 4))

  # See https://matplotlib.org/examples/color/colormaps_reference.html
  cmap = 'bwr'
  for i in range(generative_net.Ws.size(0)):
    m = generative_net.Ws[i]
    ax[i].imshow(torch.stack([m.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('group {}'.format(i))
    ax[i].set_xlabel('z_i')
    ax[i].axes.xaxis.set_ticks(range(dim_z))
    ax[i].axes.yaxis.set_ticks([])

  ax[0].set_ylabel('learned weights')
  # fig.colorbar(ax[-1])

  return fig

def debug_outgoing_weights():
  fig, ax = plt.subplots(1, dim_z, figsize=(12, 4))

  # rows correspond to groups and cols correspond to z_i's
  col_norms = torch.stack([
    torch.sqrt(torch.sum(torch.pow(generative_net.Ws[i].data.t(), 2), dim=0))
    for i in range(generative_net.Ws.size(0))
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

def debug_z_by_group_matrix():
  fig, ax = plt.subplots()
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.Ws.data, 2), dim=2)
  )
  ax.imshow(W_col_norms, aspect='equal')
  ax.set_xlabel('dimensions of z')
  ax.set_ylabel('group generative nets')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  # plt.title('Connectivity between dimensions of z and group generator networks')

prior_z = Normal(
  Variable(torch.zeros(batch_size, dim_z)),
  Variable(torch.ones(batch_size, dim_z))
)

# lr = 1e-3
# optimizer = torch.optim.Adam([
#   {'params': inference_net.parameters(), 'lr': lr},
#   {'params': generative_net.parameters(), 'lr': lr}
# ])

# lr = 1e-4
# Ws_lr = 5e-3
# momentum = 0.9
# optimizer = torch.optim.SGD([
#   {'params': inference_net.parameters(), 'lr': lr, 'momentum': momentum},
#   # {'params': generative_net.parameters(), 'lr': lr, 'momentum': momentum}
#   {'params': generative_net.group_generators_parameters(), 'lr': lr, 'momentum': momentum},
#   {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
# ])

lr = 1e-2
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
  # {'params': [inference_net_log_stddev], 'lr': lr},
  {'params': generative_net.group_generators_parameters(), 'lr': lr},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr}
])

Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
])

vae = OIVAE(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_z=prior_z,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws]
)

plot_interval = 100000
elbo_per_iter = []
for i in range(num_epochs):
  if i > 1000:
    stddev_multiple = 1

  # [batch_size, dim_x]
  Xvar = Variable(X[torch.randperm(num_train_samples)[:batch_size]])

  info = vae.step(
    X=Xvar,
    prox_step_size=Ws_lr * lam * lam_adjustment,
    mc_samples=mc_samples
  )
  elbo_per_iter.append(info['elbo'].data[0])

  if i % plot_interval == 0 and i > 0:
    debug(8)
    plt.suptitle('OI-VAE, Iteration {}, lr = {}, lam = {}, batch_size = {}, num_train_samples = {}'.format(i, lr, lam, batch_size, num_train_samples))

    debug_incoming_weights()
    plt.suptitle('incoming z weights')

    debug_outgoing_weights()
    plt.suptitle('outgoing z weight norms')

    debug_z_by_group_matrix()

    plt.figure()
    plt.plot(elbo_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO per iteration. lam = {}'.format(lam))
    plt.show()

  # Print the learned (but fixed) standard deviations of each of the generators
  # print(torch.exp(torch.stack([gen.sigma_net.extra_args[0] for gen in generative_net.group_generators])) + 1e-3)

  print('iter', i)
  print('  ELBO:', info['elbo'].data[0])
  print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
  print('    loglik_term      ', info['loglik_term'].data[0])
  print('    log p(theta)     ', info['logprob_theta'].data[0])
  print('    log p(W)         ', info['logprob_W'].data[0])

# Plot the final connectivity matrix and save
debug_z_by_group_matrix()
plt.savefig('bars_data_connectivity_matrix.pdf', format='pdf')

def save_img_and_reconstruction(ix):
  plt.figure()
  plt.imshow(X[ix].view(image_size, image_size).numpy())
  plt.savefig('{}_true.pdf'.format(ix), format='pdf')

  plt.figure()
  Xvar = Variable(X[[i]])
  fX = generative_net(inference_net(Xvar).sample()).sample().view(image_size, image_size)
  plt.imshow(fX.data.squeeze().numpy())
  plt.savefig('{}_reconstruction_full_sample.pdf'.format(ix), format='pdf')

for i in range(16):
  save_img_and_reconstruction(i)
