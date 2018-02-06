import math

import torch
from torch.autograd import Variable

from .utils import split


LOG2PI = torch.log(torch.FloatTensor([2 * math.pi]))[0]

# class NormalLogStdDev(object):
#   def __init__(self, mean, log_stddev):
#     assert mean.size() == log_stddev.size()
#     self.mean = mean
#     self.log_stddev = log_stddev

#   def sample(self):
#     return self.mean + torch.exp(self.log_stddev) * Variable(torch.randn(self.mean.size()))

#   def logprob(self, x):
#     return -0.5 * (
#       self.mean.numel() * LOG2PI
#       + 2 * torch.sum(self.log_stddev)
#       + torch.sum((x - self.mean) * torch.exp(-2 * self.log_stddev) * (x - self.mean))
#     )

class Normal(object):
  def __init__(self, mu, sigma):
    assert mu.size() == sigma.size()
    self.mu = mu
    self.sigma = sigma

  def size(self, *args, **kwargs):
    return self.mu.size(*args, **kwargs)

  def sample(self):
    eps = torch.randn(self.mu.size()).type_as(self.mu.data)
    return self.mu + self.sigma * Variable(eps)

  def logprob(self, x):
    return torch.sum(
      -0.5 * LOG2PI
      - torch.log(self.sigma)
      -0.5 * torch.pow((x - self.mu) / self.sigma, 2)
    )

  def expand(self, *args, **kwargs):
    return Normal(
      self.mu.expand(*args, **kwargs),
      self.sigma.expand(*args, **kwargs)
    )

  def expand_as(self, *args, **kwargs):
    return Normal(
      self.mu.expand_as(*args, **kwargs),
      self.sigma.expand_as(*args, **kwargs)
    )

class Gamma(object):
  def __init__(self, shape, rate):
    assert shape.size() == rate.size()
    self.shape = shape
    self.rate = rate

  def logprob(self, x):
    return torch.sum(
      self.shape * torch.log(self.rate)
      - torch.lgamma(self.shape)
      + (self.shape - 1) * torch.log(x)
      - self.rate * x
    )

class _DistributionCat(object):
  def __init__(self, distributions, dim=-1):
    self.distributions = distributions
    self.dim = dim

  def sample(self):
    return torch.cat([d.sample() for d in self.distributions], dim=self.dim)

  def logprob(self, x):
    xs = split(x, [d.size(self.dim) for d in self.distributions], dim=self.dim)
    return sum(d.logprob(xx) for d, xx in zip(self.distributions, xs))

  def entropy(self):
    return sum(d.entropy() for d in self.distributions)

def DistributionCat(distributions, dim=-1):
  if all(isinstance(d, Normal) for d in distributions):
    return Normal(
      mu=torch.cat([d.mu for d in distributions], dim=dim),
      sigma=torch.cat([d.sigma for d in distributions], dim=dim)
    )
  else:
    return _DistributionCat(distributions, dim=-1)
