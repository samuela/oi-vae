import torch
from torch.autograd import Variable

from bars_data import sample_bars, sample_bars_one
from common import FirstLayerSparseDecoder

import matplotlib.pyplot as plt


torch.manual_seed(0)

image_size = 16
dim_z = image_size // 2
nums_train_samples = [2, 4, 8, 16, 32, 64, 128, 256, 512]
num_test_samples = 2048
num_epochs = 2500

# lams = [0.0, 0.001, 0.01, 0.1, 1]
lams = [0.0, 0.1]

def make_linear_encoder():
  return torch.nn.Linear(image_size * image_size, dim_z, bias=False)

def make_linear_decoder():
  return FirstLayerSparseDecoder(
    [torch.nn.Linear(1, image_size, bias=False) for _ in range(image_size)],
    [1 for _ in range(image_size)],
    dim_z
  )

def make_nonlinear_decoder():
  return FirstLayerSparseDecoder(
    [
      torch.nn.Sequential(
        torch.nn.Linear(1, image_size, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(image_size, image_size, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(image_size, image_size, bias=False)
      )
      for _ in range(image_size)
    ],
    [1 for _ in range(image_size)],
    dim_z
  )

def sample_bars_data(n):
  return torch.stack([
    sample_bars(
      torch.ones(image_size) / image_size * 2,
      torch.ones(image_size) / image_size * 0
    ).view(-1)
    for _ in range(n)
  ])

def sample_single_bars_data(n):
  return torch.stack([
    sample_bars_one(image_size).view(-1)
    for _ in range(n)
  ])

def sample_tied_bars_data(n, num_repeats):
  halfX = torch.stack([
    sample_bars(
      torch.ones(image_size // num_repeats) / image_size * 2,
      torch.ones(image_size) / image_size * 0
    ).view(-1)
    for _ in range(n)
  ])
  return torch.cat([halfX] * num_repeats, dim=1)

make_encoder = make_linear_encoder
make_decoder = make_nonlinear_decoder
sample_data = lambda n: sample_tied_bars_data(n, 2)

def make_optimizer(encoder, decoder):
  lr = 1e-2
  return lr, torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': lr},
    {'params': decoder.parameters(), 'lr': lr}
  ])

def reconstruction_loss(encoder, decoder, data):
  Xvar = Variable(data)
  reconstructed = decoder(encoder(Xvar))
  if torch.sum(torch.abs(reconstructed - reconstructed[0])).data[0] / Xvar.size(0) <= 1e-3:
    print('solution has collapsed!')

  residual = reconstructed - Xvar
  return torch.sum(torch.pow(residual, 2)) / Xvar.size(0)

def run():
  results = {}
  Xtest = sample_data(num_test_samples)
  for i, num_train_samples in enumerate(nums_train_samples):
    Xtrain = sample_data(num_train_samples)
    for j, lam in enumerate(lams):
      print('i = {}/{}, j = {}/{}'.format(i, len(nums_train_samples), j, len(lams)))
      encoder = make_encoder()
      decoder = make_decoder()
      lr, optimizer = make_optimizer(encoder, decoder)

      test_loss_per_iter = []
      train_loss_per_iter = []
      sparsity_penalty_per_iter = []
      for _ in range(num_epochs):
        test_loss = reconstruction_loss(encoder, decoder, Xtest)
        train_loss = reconstruction_loss(encoder, decoder, Xtrain)
        sparsity_penalty = lam * decoder.group_lasso_penalty()

        test_loss_per_iter.append(test_loss.data[0])
        train_loss_per_iter.append(train_loss.data[0])
        sparsity_penalty_per_iter.append(sparsity_penalty.data[0])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        decoder.proximal_step(lr * lam)

        # print('epoch', i)
        # print('  reconstruction loss:     ', train_loss.data[0])
        # print('  regularization:          ', sparsity_penalty.data[0])
        # print('  combined loss:           ', (train_loss + sparsity_penalty).data[0])
        # print('  test reconstruction loss:', test_loss.data[0])

      # plt.figure()
      # plt.plot(test_loss_per_iter)
      # plt.plot(train_loss_per_iter)
      # plt.plot(sparsity_penalty_per_iter)
      # plt.xlabel('epoch')
      # plt.legend(['test reconstruction loss', 'train reconstruction loss', 'sparsity penalty'])
      # plt.title('num_train_samples = {}, lam = {}'.format(num_train_samples, lam))
      # plt.show()

      results[(i, j)] = (
        test_loss_per_iter,
        train_loss_per_iter,
        sparsity_penalty_per_iter
      )

  return results

res = run()

import pickle
pickle.dump(res, open('poop.p', 'wb'))

plt.figure()
for j in range(len(lams)):
  test_losses = [res[(i, j)][0][-1] for i in range(len(nums_train_samples))]
  # plt.plot(nums_train_samples, test_losses)
  plt.semilogx(nums_train_samples, test_losses, basex=2, marker='o')
plt.xlabel('num. training samples')
plt.ylabel('test reconstruction loss')
plt.legend(['lambda = {}'.format(lam) for lam in lams])
plt.show()
