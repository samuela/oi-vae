"""Bayesian group lasso in a VAE setup.

* Point estimates on the sparse weight parameters and on the thetas.
* Gamma variables are reparameterized by an inverse softmax.

This model is WRONG in that it does not actually correspond to group lasso since
it optimizes over the gamma variables (zeta/theta).
"""

import torch
from torch.autograd import Variable

from bars_data import sample_bars
from common import KL_Normals, BayesianGroupLassoGenerator, NormalNet
from distributions import Gamma, Normal
from utils import invert_bijection, softplus

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 8
dim_z = image_size
dim_x = image_size * image_size
num_groups = image_size
group_input_dim = 1

lam = 1

num_train_samples = 1024
num_epochs = 1000000
mc_samples = 1
batch_size = 128

# Basic bars
X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size * 2,
    torch.ones(image_size) / image_size * 0
  ).view(-1)
  for _ in range(num_train_samples)
])

# Shared bars
# halfX = torch.stack([
#   sample_bars(
#     torch.ones(image_size // 2) / image_size * 2,
#     torch.ones(image_size) / image_size * 0
#   ).view(-1)
#   for _ in range(num_train_samples)
# ])
# X = torch.cat([halfX, halfX], dim=1)

class Lambda(torch.nn.Module):
  def __init__(self, func):
    super(Lambda, self).__init__()
    self.func = func

  def forward(self, x):
    return self.func(x)

inference_net = NormalNet(
  mu_net=torch.nn.Linear(dim_x, dim_z),
  sigma_net=torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_z),
    torch.nn.Softplus(),
    Lambda(lambda x: x * 0 + 1e-3)
  )
)

generative_net = BayesianGroupLassoGenerator(
  group_generators=[
    NormalNet(
      mu_net=torch.nn.Linear(group_input_dim, image_size),
      sigma_net=torch.nn.Sequential(
        torch.nn.Linear(group_input_dim, image_size),
        torch.nn.Softplus(),
        Lambda(lambda x: x * 0 + 1e-3)
      )
    )
    for _ in range(image_size)
  ],
  group_input_dim=group_input_dim,
  dim_z=dim_z
)

# lr = 1e1 / 4096
# momentum = 0.7
# optimizer = torch.optim.SGD([
#   {'params': inference_net.parameters(), 'lr': lr, 'momentum': momentum},
#   {'params': generative_net.parameters(), 'lr': lr, 'momentum': momentum}
# ])

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
    ax[1, i].bar(range(dim_z), inference_net(Xvar).mu.data.numpy())
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i in range(count):
    Xvar = Variable(X[i])
    # fX = generative_net(inference_net(Xvar).mu).mu.view(image_size, image_size)
    fX = generative_net(inference_net(Xvar).sample()).sample().view(image_size, image_size)
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
  for i in range(generative_net.Ws.size(0)):
    m = generative_net.Ws[i]
    ax[i].imshow(torch.stack([m.data for _ in range(image_size)]).squeeze(), vmin=-1, vmax=1, cmap=cmap)
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
    ax[i].imshow(torch.stack([col_norms[:, i] for _ in range(image_size)]).squeeze(), vmin=-1, vmax=1, cmap=cmap)
    ax[i].set_title('z_{}'.format(i))
    ax[i].set_xlabel('groups')
    ax[i].axes.xaxis.set_ticks(range(image_size))
    ax[i].axes.yaxis.set_ticks([])

  return fig

prior_z = Normal(
  Variable(torch.zeros(batch_size, dim_z)),
  Variable(torch.ones(batch_size, dim_z))
)

# [num_groups, dim_z]
prior_thetas = Gamma(
  (group_input_dim + 1) / 2 * Variable(torch.ones(num_groups, dim_z)),
  (lam ** 2) / 2 * Variable(torch.ones(num_groups, dim_z))
)

# [num_groups, dim_z, group_input_dim]
prior_Ws = lambda thetas: Normal(
  Variable(torch.zeros(num_groups, dim_z, group_input_dim)),
  torch.sqrt(thetas + 1e-12).unsqueeze(2).expand(-1, -1, group_input_dim)
)

# T(theta) = zeta  <=>  theta = T^-1(zeta)
T = invert_bijection(softplus)
zetas = Variable(torch.randn(num_groups, dim_z), requires_grad=True)

lr = 1e-2
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
  {'params': generative_net.parameters(), 'lr': lr},
  {'params': [zetas], 'lr': lr}
])

plot_interval = 10000
for i in range(num_epochs):
  # [batch_size, dim_x]
  Xvar = Variable(X[torch.randperm(num_train_samples)[:batch_size]])

  # [batch_size, dim_z]
  q_z = inference_net(Xvar)

  # KL divergence is additive across independent joint distributions, so this
  # works appropriately.
  z_kl = KL_Normals(q_z, prior_z)

  z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)
  Xrep = Variable(Xvar.data.repeat(mc_samples, 1))
  loglik_term = generative_net(z_sample).logprob(Xrep) / mc_samples

  # loglik_term = 1.0 / mc_samples * sum(
  #   generative_net(q_z.sample()).logprob(Xvar)
  #   for _ in range(mc_samples)
  # )

  # loglik_term = 0
  # for j in range(mc_samples):
  #   z_sample = q_z.sample()
  #   x_dist = generative_net(z_sample)
  #   loglik_term += x_dist.logprob(Xvar)
  # loglik_term /= mc_samples

  thetas = T.inverse(zetas)
  W_term = prior_Ws(thetas).logprob(generative_net.Ws)
  theta_term = prior_thetas.logprob(thetas)
  jacobian_term = T.inverse_log_abs_det_jacobian(zetas)

  loss = -1.0 / batch_size * (
    -z_kl
    + loglik_term
    + W_term
    + theta_term
    + jacobian_term
  )

  if i % plot_interval == 0 and i > 0:
    debug(8)
    plt.suptitle('Group Lasso VAE, Adam, Iteration {}, lr = {}, lam = {}, batch_size = {}, num_train_samples = {}'.format(i, lr, lam, batch_size, num_train_samples))

    debug_incoming_weights()
    plt.suptitle('incoming z weights')

    debug_outgoing_weights()
    plt.suptitle('outgoing z weight norms')

    plt.show()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print('iter', i)
  print('  ELBO:', -loss.data[0])
  print('    z_kl', z_kl.data[0] / batch_size)
  print('    loglik_term', loglik_term.data[0] / batch_size)
  print('    log p(W, theta)', (W_term + theta_term + jacobian_term).data[0] / batch_size)

  print(zetas)
  print(thetas)
