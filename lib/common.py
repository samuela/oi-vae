import itertools

import torch
from torch.autograd import Variable

from .distributions import Normal, JointDistribution


# class ConstantVarianceNet(object):
#   def __init__(self, mu_net, log_stddev):
#     self.mu_net = mu_net
#     self.log_stddev = log_stddev

#   def __call__(self, x):
#     mean = self.mu_net(x)
#     # return DiagonalMVN(mean, self.log_stddev * torch.ones(mean.size()))
#     return DiagonalMVN(mean, self.log_stddev)

#   def parameters(self):
#     return self.mu_net.parameters()

class NormalNet(object):
  def __init__(self, mu_net, sigma_net):
    self.mu_net = mu_net
    self.sigma_net = sigma_net

  def __call__(self, x):
    return Normal(self.mu_net(x), self.sigma_net(x))

  def parameters(self):
    return itertools.chain(
      self.mu_net.parameters(),
      self.sigma_net.parameters()
    )

# class LearnedTiedVarianceNet(object):
#   def __init__(self, mu_net, log_stddev_net):
#     self.mu_net = mu_net
#     self.log_stddev_net = log_stddev_net

#   def __call__(self, x):
#     mean = self.mu_net(x)
#     log_stddev = self.log_stddev_net(x)
#     return DiagonalMVN(mean, log_stddev * Variable(torch.ones(mean.size())))

#   def parameters(self):
#     return itertools.chain(
#       self.mu_net.parameters(),
#       self.log_stddev_net.parameters()
#     )

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

class BayesianGroupLassoGenerator(object):
  def __init__(self, group_generators, group_input_dim, dim_z):
    self.group_generators = group_generators
    self.group_input_dim = group_input_dim
    self.dim_z = dim_z
    self.num_groups = len(group_generators)
    self.Ws = Variable(
      torch.randn(self.num_groups, self.dim_z, self.group_input_dim),
      requires_grad=True
    )
    # self.Ws = [
    #   Variable(
    #     torch.randn(self.dim_z, self.group_input_dim),
    #     requires_grad=True
    #   )
    #   for _ in self.group_generators
    # ]

  def __call__(self, z):
    return JointDistribution(
      [gen(z @ self.Ws[i]) for i, gen in enumerate(self.group_generators)],
      dim=-1
    )

  def parameters(self):
    return itertools.chain(
      *[gen.parameters() for gen in self.group_generators],
      [self.Ws]
    )

  def proximal_step(self, t):
    row_norms = torch.sqrt(
      torch.sum(torch.pow(self.Ws.data, 2), dim=1, keepdim=True)
    )
    self.Ws.data.div_(torch.clamp(row_norms, min=1e-16))
    self.Ws.data.mul_(torch.clamp(row_norms - t, min=0))

# See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
# def KL_DiagonalMVNs(d0, d1):
#   assert d0.mean.size() == d1.mean.size()
#   k = d0.mean.numel()
#   v0 = torch.exp(2 * d0.log_stddev)
#   v1inv = torch.exp(-2 * d1.log_stddev)
#   return 0.5 * (
#     torch.sum(v1inv * v0)
#     + torch.sum((d1.mean - d0.mean) * v1inv * (d1.mean - d0.mean))
#     - k
#     + 2 * torch.sum(d1.log_stddev)
#     - 2 * torch.sum(d0.log_stddev)
#   )

# See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
def KL_Normals(d0, d1):
  assert d0.mu.size() == d1.mu.size()
  sigma0_sqr = torch.pow(d0.sigma, 2)
  sigma1_sqr = torch.pow(d1.sigma, 2)
  return torch.sum(
    -0.5
    + (sigma0_sqr + torch.pow(d0.mu - d1.mu, 2)) / (2 * sigma1_sqr)
    + torch.log(d1.sigma)
    - torch.log(d0.sigma)
  )
