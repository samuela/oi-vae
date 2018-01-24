import itertools

import torch
from torch.autograd import Variable

from distributions import DiagonalMVN


class ConstantVarianceNet(object):
  def __init__(self, mean_net, log_stddev):
    self.mean_net = mean_net
    self.log_stddev = log_stddev

  def __call__(self, x):
    mean = self.mean_net(x)
    # return DiagonalMVN(mean, self.log_stddev * torch.ones(mean.size()))
    return DiagonalMVN(mean, self.log_stddev)

  def parameters(self):
    return self.mean_net.parameters()

class LearnedVarianceNet(object):
  def __init__(self, mean_net, log_stddev_net):
    self.mean_net = mean_net
    self.log_stddev_net = log_stddev_net

  def __call__(self, x):
    mean = self.mean_net(x)
    log_stddev = self.log_stddev_net(x)
    return DiagonalMVN(mean, log_stddev)

  def parameters(self):
    return itertools.chain(
      self.mean_net.parameters(),
      self.log_stddev_net.parameters()
    )

class LearnedTiedVarianceNet(object):
  def __init__(self, mean_net, log_stddev_net):
    self.mean_net = mean_net
    self.log_stddev_net = log_stddev_net

  def __call__(self, x):
    mean = self.mean_net(x)
    log_stddev = self.log_stddev_net(x)
    return DiagonalMVN(mean, log_stddev * Variable(torch.ones(mean.size())))

  def parameters(self):
    return itertools.chain(
      self.mean_net.parameters(),
      self.log_stddev_net.parameters()
    )

class FirstLayerSparseDecoder(object):
  def __init__(self, group_generators, group_generators_input_dims, input_dim):
    assert len(group_generators) == len(group_generators_input_dims)
    self.group_generators = group_generators
    self.group_generators_input_dims = group_generators_input_dims
    self.input_dim = input_dim

    # The (hopefully sparse) mappings from the latent z to the inputs to each of
    # the group generators.
    self.latent_to_group_maps = [
      torch.nn.Linear(self.input_dim, k, bias=False)
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

  def proximal_step(self, t):
    for m in self.latent_to_group_maps:
      col_norms = torch.sqrt(torch.sum(torch.pow(m.weight.data, 2), dim=0))

      # We clamp the col_norms to prevent divide by 0 NaNs.
      m.weight.data.div_(torch.clamp(col_norms, min=1e-16))
      m.weight.data.mul_(torch.clamp(col_norms - t, min=0))

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
