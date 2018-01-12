import torch
from torch.autograd import Variable

from bars_data import sample_bars
from distributions import DiagonalMVN

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 4
num_samples = 1024
num_epochs = 1000000
mc_samples = 1
batch_size = 64

X = torch.stack([
  sample_bars(
    torch.ones(image_size) / image_size,
    torch.ones(image_size) / image_size
  ).view(-1)
  for _ in range(num_samples)
])

# See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
def KL_DiagonalMVNs(d0, d1):
  assert d0.mean.size() == d1.mean.size()
  k = d0.mean.numel()
  v0 = torch.exp(2 * d0.log_stddev)
  v1inv = torch.exp(-2 * d1.log_stddev)
  return 0.5 * (
    torch.sum(v1inv * v0)
    + torch.sum((d1.mean - d0.mean) * v1inv * (d1.mean - d0.mean))
    - k
    + 2 * torch.sum(d1.log_stddev)
    - 2 * torch.sum(d0.log_stddev)
  )

class FixedVarianceNet(object):
  def __init__(self, mean_net, log_stddev):
    self.mean_net = mean_net
    self.log_stddev = log_stddev

  def __call__(self, x):
    mean = self.mean_net(x)
    # return DiagonalMVN(mean, self.log_stddev * torch.ones(mean.size()))
    return DiagonalMVN(mean, self.log_stddev)

  def parameters(self):
    return self.mean_net.parameters()

inference_net = FixedVarianceNet(
  torch.nn.Linear(image_size * image_size, 2 * image_size),
  Variable(-6 * torch.ones(2 * image_size))
)

generative_net = FixedVarianceNet(
  torch.nn.Linear(2 * image_size, image_size * image_size),
  Variable(0 * torch.ones(image_size * image_size))
)

z_prior = DiagonalMVN(
  Variable(torch.zeros(2 * image_size)),
  Variable(torch.zeros(2 * image_size))
)

lr = 1e0 / 4
optimizer = torch.optim.SGD([
  {'params': inference_net.parameters(), 'lr': lr, 'momentum': 0.9},
  {'params': generative_net.parameters(), 'lr': lr, 'momentum': 0.9}
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

plot_interval = 1000

for i in range(num_epochs):
  loss = 0
  for j in torch.randperm(num_samples)[:batch_size]:
    Xvar = Variable(X[j])
    vi_posterior = inference_net(Xvar)
    likelihood_term = sum(
      generative_net(vi_posterior.sample()).logprob(Xvar)
      # generative_net(vi_posterior.mean).logprob(Xvar)
      for _ in range(mc_samples)
    )
    kl = KL_DiagonalMVNs(vi_posterior, z_prior)
    elbo = -kl + likelihood_term / mc_samples
    loss -= elbo

  loss /= batch_size

  if i % plot_interval == 0:
    debug(8)
    plt.suptitle('Iteration {}'.format(i))
    plt.show()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print('epoch', i, 'cumulative ELBO:', -loss.data[0])
  # print(-kl.data[0], likelihood_term.data[0] / mc_samples)
