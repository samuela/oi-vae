import torch


def sample_bars(row_probs, col_probs):
  m = row_probs.numel()
  n = col_probs.numel()
  return (torch.bernoulli(row_probs).repeat(n, 1).t() +
          torch.bernoulli(col_probs).repeat(m, 1)) / 2
