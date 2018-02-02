import torch


def sample_bars_image(row_probs, col_probs):
  m = row_probs.numel()
  n = col_probs.numel()
  return (torch.bernoulli(row_probs).repeat(n, 1).t() +
          torch.bernoulli(col_probs).repeat(m, 1)) / 2

def sample_one_bar_image(image_size):
  i = torch.randperm(image_size)[0]
  A = torch.zeros(image_size, image_size)
  A[i, :] = 0.5
  return A

def sample_many_bars_images(num_images, image_size, row_mul, col_mul):
  return torch.stack([
    sample_bars_image(
      torch.ones(image_size) / image_size * row_mul,
      torch.ones(image_size) / image_size * col_mul
    ).view(-1)
    for _ in range(num_images)
  ])
