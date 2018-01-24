import torch


def sample_bars(row_probs, col_probs):
  m = row_probs.numel()
  n = col_probs.numel()
  return (torch.bernoulli(row_probs).repeat(n, 1).t() +
          torch.bernoulli(col_probs).repeat(m, 1)) / 2

def sample_bars_one(image_size):
  i = torch.randperm(image_size)[0]
  A = torch.zeros(image_size, image_size)
  A[i, :] = 0.5 + 0.1 * torch.randn(image_size)
  return A
