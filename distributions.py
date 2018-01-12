import torch
from torch.autograd import Variable

import math


LOG2PI = torch.log(torch.FloatTensor([2 * math.pi]))[0]

class DiagonalMVN(object):
  def __init__(self, mean, log_stddev):
    assert mean.size() == log_stddev.size()
    self.mean = mean
    self.log_stddev = log_stddev

  def sample(self):
    return self.mean + torch.exp(self.log_stddev) * Variable(torch.randn(self.mean.size()))
    # return torch.normal(self.mean, torch.exp(self.log_stddev))

  def logprob(self, x):
    return -0.5 * (
      self.mean.numel() * LOG2PI
      + 2 * torch.sum(self.log_stddev)
      + torch.sum((x - self.mean) * torch.exp(-2 * self.log_stddev) * (x - self.mean))
    )
