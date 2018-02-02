"""A standard autoencoder model with the group lasso sparsity penalty and
proximal gradient descent. Runs on bars data."""

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from lib.bars_data import sample_bars, sample_bars_one
from lib.common import FirstLayerSparseDecoder


torch.manual_seed(0)

image_size = 16
dim_z = image_size // 2
num_train_samples = 1028
num_test_samples = 2048
num_epochs = 100000

lam = 0.1
# lam = 0

encoder = torch.nn.Linear(image_size * image_size, dim_z, bias=False)
# decoder = torch.nn.Linear(dim_z, image_size * image_size)

def make_linear_decoder():
  return FirstLayerSparseDecoder(
    [torch.nn.Linear(1, image_size, bias=False)
     for _ in range(image_size)],
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

# decoder = make_nonlinear_decoder()
decoder = make_linear_decoder()

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

# Sample data from a random generator net
# def sample_decoder_data(z_connectivity):
#   sampler = make_decoder()
#   for m in sampler.latent_to_group_maps:
#     clamped = torch.clamp(m.weight.data, min=-1, max=1)
#     _, ix = torch.topk(torch.sum(torch.pow(clamped, 2), dim=0), z_connectivity)

#     mask = torch.zeros(dim_z)
#     # mask[torch.randperm(dim_z)[:z_connectivity]] = 1
#     mask[ix] = 1
#     m.weight.data = clamped * mask
#   return sampler(Variable(torch.randn(num_train_samples, dim_z))).data, sampler

# X, sampler = sample_decoder_data(3)

Xtrain = sample_tied_bars_data(num_train_samples, 2)
Xtest = sample_tied_bars_data(num_test_samples, 2)

### Hardcode a known correct model in the case of linear group decoders
def hardcode_model():
  poop = 1
  baseline_val = 0.5

  # The encoder model
  for i in range(image_size):
    A = torch.zeros(image_size, image_size)
    A[i, :] = 1
    encoder.weight.data[i, :] = A.view(-1) / image_size / baseline_val / poop

    decoder.latent_to_group_maps[i].weight.data[:] = 0
    decoder.latent_to_group_maps[i].weight.data[0, i] = poop
    decoder.latent_to_group_maps[i].bias.data[:] = 0

    decoder.group_generators[i].weight.data[:, 0] = baseline_val * torch.ones(image_size)
    decoder.group_generators[i].bias.data[:] = 0

  encoder.bias.data[:] = 0

  # for i in range(image_size):
  #   A = torch.zeros(image_size)

# hardcode_model()
###

# lr = 1e0 / 32
# momentum = 0.0
# optimizer = torch.optim.SGD([
#   {'params': encoder.parameters(), 'lr': lr, 'momentum': momentum},
#   {'params': decoder.parameters(), 'lr': lr, 'momentum': momentum}
# ])

lr = 1e-2
optimizer = torch.optim.Adam([
  {'params': encoder.parameters(), 'lr': lr},
  {'params': decoder.parameters(), 'lr': lr}
])

def debug(data, ixs):
  fig, ax = plt.subplots(3, len(ixs), figsize=(12, 4))

  # True images
  for i, ix in enumerate(ixs):
    # X = torch.zeros(image_size, image_size)
    # X[i, :] = 0.5
    # Xvar = Variable(X.view(-1))

    Xvar = Variable(data[ix])
    ax[0, i].imshow(Xvar.data.view(image_size, image_size).numpy(), vmin=0, vmax=1)
    # ax[0, i].set_title('{:6.4f}'.format(encoder(Xvar)[2].data[0]))
    ax[0, i].axes.xaxis.set_ticks([])
    ax[0, i].axes.yaxis.set_ticks([])
    # ax[0, i].set_title(i)

  for i, ix in enumerate(ixs):
    # X = torch.zeros(image_size, image_size)
    # X[i, :] = 0.5
    # Xvar = Variable(X.view(-1))

    Xvar = Variable(data[ix])
    fX = encoder(Xvar)
    ax[1, i].bar(range(dim_z), fX.data.numpy())
    # ax[0, i].set_title('{:6.4f}'.format(encoder(Xvar)[2].data[0]))
    ax[1, i].axes.xaxis.set_ticks([])
    ax[1, i].axes.yaxis.set_ticks([])

  # Reconstructed images
  for i, ix in enumerate(ixs):
    # X = torch.zeros(image_size, image_size)
    # X[i, :] = 0.5
    # Xvar = Variable(X.view(-1))

    Xvar = Variable(data[ix])
    fX = decoder(encoder(Xvar)).view(image_size, image_size)
    ax[2, i].imshow(fX.data.numpy(), vmin=0, vmax=1)
    # loss = torch.sum(torch.pow(fX - Xvar.view(image_size, image_size), 2))
    # ax[2, i].set_title('{:6.4f}'.format(loss.data[0]))
    ax[2, i].axes.xaxis.set_ticks([])
    ax[2, i].axes.yaxis.set_ticks([])

  ax[0, 0].set_ylabel('true image')
  ax[1, 0].set_ylabel('z')
  ax[2, 0].set_ylabel('reconstructed')

  return fig

def debug2():
  plt.figure(figsize=(12, 4))
  plt.suptitle('zs decoded')
  for i in range(dim_z):
    plt.subplot(1, dim_z, i + 1)
    z = torch.zeros(dim_z)
    z[i] = 1
    plt.imshow(decoder(Variable(z)).view(image_size, image_size).data.numpy(), vmin=0, vmax=1)
    plt.title('Component {}'.format(i))

  plt.colorbar()

# def debug_sampler_vs_learned_weights(sampler):
#   plt.figure(figsize=(12, 4))

#   # See https://matplotlib.org/examples/color/colormaps_reference.html
#   cmap = 'bwr'
#   for j, m in enumerate(sampler.latent_to_group_maps):
#     plt.subplot(2, image_size, j + 1)
#     plt.imshow(torch.stack([m.weight.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
#     plt.title('group {}'.format(j))
#     # plt.xlabel('z_i')
#     plt.gca().xaxis.set_ticks(range(image_size))
#     plt.gca().yaxis.set_ticks([])

#   for j, m in enumerate(decoder.latent_to_group_maps):
#     plt.subplot(2, image_size, j + 1 + image_size)
#     plt.imshow(torch.stack([m.weight.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
#     # plt.title('group {}'.format(j))
#     plt.xlabel('z_i')
#     plt.gca().xaxis.set_ticks(range(image_size))
#     plt.gca().yaxis.set_ticks([])

#   plt.subplot(2, image_size, 1)
#   plt.ylabel('true weights')
#   plt.subplot(2, image_size, image_size + 1)
#   plt.ylabel('learned weights')

#   plt.suptitle('first layer weights, iter = {}, lambda = {}, lr = {}, momentum = {}\nNote that the z components may be permuted between the true and learned models.'.format(i, lam, lr, momentum))

def debug_incoming_weights():
  fig, ax = plt.subplots(1, image_size, figsize=(12, 4))

  # See https://matplotlib.org/examples/color/colormaps_reference.html
  cmap = 'bwr'
  for i, m in enumerate(decoder.latent_to_group_maps):
    ax[i].imshow(torch.stack([m.weight.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('group {}'.format(i))
    ax[i].set_xlabel('z_i')
    ax[i].axes.xaxis.set_ticks(range(dim_z))
    ax[i].axes.yaxis.set_ticks([])

  ax[0].set_ylabel('learned weights')

  return fig

def debug_outgoing_weights():
  fig, ax = plt.subplots(1, dim_z, figsize=(12, 4))

  # rows correspond to groups and cols correspond to z_i's
  col_norms = torch.stack([
    torch.sqrt(torch.sum(torch.pow(m.weight.data, 2), dim=0))
    for m in decoder.latent_to_group_maps
  ])

  # See https://matplotlib.org/examples/color/colormaps_reference.html
  cmap = 'bwr'
  for i in range(dim_z):
    ax[i].imshow(torch.stack([col_norms[:, i] for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
    ax[i].set_title('z_{}'.format(i))
    ax[i].set_xlabel('groups')
    ax[i].axes.xaxis.set_ticks(range(image_size))
    ax[i].axes.yaxis.set_ticks([])

  return fig

plot_interval = 2500

# Calculate the reconstruction loss on the given data
def reconstruction_loss(data):
  Xvar = Variable(data)
  reconstructed = decoder(encoder(Xvar))
  if torch.sum(torch.abs(reconstructed - reconstructed[0])).data[0] / Xvar.size(0) <= 1e-3:
    print('solution has collapsed!')

  residual = reconstructed - Xvar
  return torch.sum(torch.pow(residual, 2)) / Xvar.size(0)

for epoch in range(num_epochs):
  train_loss = reconstruction_loss(Xtrain)
  sparsity_penalty = lam * decoder.group_lasso_penalty()

  optimizer.zero_grad()
  train_loss.backward()
  optimizer.step()
  decoder.proximal_step(lr * lam)
  print('epoch', epoch)
  print('  reconstruction loss:', train_loss.data[0])
  print('  regularization:     ', sparsity_penalty.data[0])
  print('  combined loss:      ', (train_loss + sparsity_penalty).data[0])

  test_loss = reconstruction_loss(Xtest)
  print('  test reconstruction loss:', test_loss.data[0])

  if epoch % plot_interval == 0 and epoch > 0:
    test_loss = reconstruction_loss(Xtest)
    print('  test reconstruction loss:', test_loss.data[0])
    # print('### decoder.latent_to_group_maps parameters')
    # for m in decoder.latent_to_group_maps:
    #   print(m.weight)

    # debug([0, 1, 4, 7, 8, 9, 12, 25])
    debug(Xtrain, [0, 1, 2, 3, 4, 5, 6, 7])
    # plt.suptitle('Iteration {}, lambda = {}, lr = {}, momentum = {}'.format(i, lam, lr, momentum))
    plt.suptitle('Iteration {}, lambda = {}, lr = {}'.format(epoch, lam, lr))

    # debug2()

    debug_incoming_weights()
    plt.suptitle('incoming weights')

    debug_outgoing_weights()
    plt.suptitle('outgoing weights')

    plt.show()
