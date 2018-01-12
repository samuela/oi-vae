import torch
from torch.autograd import Variable

import math


torch.manual_seed(0)

LOG2PI = torch.log(torch.FloatTensor([2 * math.pi]))[0]

class DiagonalMVN(object):
  def __init__(self, mean, log_stddev):
    assert mean.size() == log_stddev.size()
    self.mean = mean
    self.log_stddev = log_stddev

  def sample_bad(self):
    return torch.normal(self.mean, torch.exp(self.log_stddev))

  def sample_good(self):
    return self.mean + torch.exp(self.log_stddev) * Variable(torch.randn(self.mean.size()))

  def logprob(self, x):
    return -0.5 * (
      self.mean.numel() * LOG2PI
      + 2 * torch.sum(self.log_stddev)
      + torch.sum((x - self.mean) * torch.exp(-2 * self.log_stddev) * (x - self.mean))
    )

class FixedVarianceNet(object):
  def __init__(self, mean_net, log_stddev):
    self.mean_net = mean_net
    self.log_stddev = log_stddev

  def __call__(self, x):
    return DiagonalMVN(self.mean_net(x), self.log_stddev)

inference_net = FixedVarianceNet(torch.nn.Linear(2, 1), Variable(torch.zeros(1)))
generative_net = FixedVarianceNet(torch.nn.Linear(1, 2), Variable(torch.zeros(2)))

print('### torch.normal broken!')
Xvar = Variable(torch.randn(2))
vi_posterior = inference_net(Xvar)
loss = -generative_net(vi_posterior.sample_bad()).logprob(Xvar)
loss.backward()

print('inference_net.mean_net.bias.grad ==', inference_net.mean_net.bias.grad)
print()

print('### Custom sample() works...')
Xvar = Variable(torch.randn(2))
vi_posterior = inference_net(Xvar)
loss = -generative_net(vi_posterior.sample_good()).logprob(Xvar)
loss.backward()

print('inference_net.mean_net.bias.grad ==', inference_net.mean_net.bias.grad)
