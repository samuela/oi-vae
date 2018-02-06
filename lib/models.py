import itertools

import torch
from torch.autograd import Variable
from torch.nn import Module

from .distributions import Normal, DistributionCat


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
  
  def cuda(self):
    self.mu_net.cuda()
    self.sigma_net.cuda()


class FirstLayerSparseDecoder(object):
  """A net architecture that enforces group lasso sparsity on the first
  layer. Designed for use with standard autoencoders."""

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
  """A net architecture with group lasso sparsity on the first layer. Each group
  generator is assumed to output a distribution as opposed to a Tensor in the
  `FirstLayerSparseDecoder` model."""

  def __init__(self, group_generators, group_input_dim, dim_z):
    self.group_generators = group_generators
    self.group_input_dim = group_input_dim
    self.dim_z = dim_z
    self.num_groups = len(group_generators)

    # Starting this off with reasonably large values is helpful so that proximal
    # gradient descent doesn't prematurely kill them.
    self.Ws = Variable(
      torch.randn(self.num_groups, self.dim_z, self.group_input_dim), 
      requires_grad=True
    )

  def __call__(self, z):
    return DistributionCat(
      [gen(z @ self.Ws[i]) for i, gen in enumerate(self.group_generators)],
      dim=-1
    )

  def group_generators_parameters(self):
    return itertools.chain(*[gen.parameters() for gen in self.group_generators])

  def parameters(self):
    return itertools.chain(
      *[gen.parameters() for gen in self.group_generators],
      [self.Ws]
    )

  def proximal_step(self, t):
    row_norms = torch.sqrt(
      torch.sum(torch.pow(self.Ws.data, 2), dim=2, keepdim=True)
    )
    self.Ws.data.div_(torch.clamp(row_norms, min=1e-16))
    self.Ws.data.mul_(torch.clamp(row_norms - t, min=0))

  def group_lasso_penalty(self):
    return torch.sum(torch.sqrt(torch.sum(torch.pow(self.Ws, 2), dim=2)))

  def cuda(self):
    self.Ws = Variable(self.Ws.data.cuda(), requires_grad=True)
    for gen in self.group_generators:
      gen.cuda()
