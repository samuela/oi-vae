import torch
from torch.autograd import Variable


mu = Variable(torch.Tensor([1]), requires_grad=True)
sigma = Variable(torch.Tensor([1]), requires_grad=False)
loss = torch.pow(torch.normal(mu, sigma), 2)
loss.backward()
