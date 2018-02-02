"""Weird Edward behavior here. Things go to NaN even when learning rate = 0."""

import edward as ed
from edward.models import Normal, Gamma

import tensorflow as tf

import torch
from lib.bars_data import sample_bars


ed.set_seed(0)
torch.manual_seed(0)

image_size = 8

def sample_bars_data(n):
  return torch.stack([
    sample_bars(
      torch.ones(image_size) / image_size * 2,
      torch.ones(image_size) / image_size * 0
    ).view(-1)
    for _ in range(n)
  ]).numpy()

num_samples = 1024
dim_z = image_size

# group input dim
m_g = 1

# regularization param
lam = 0.1

X = sample_bars_data(num_samples)

# MODEL
z = Normal(
  # None doesn't work here for some reason
  loc=tf.zeros([num_samples, dim_z]),
  scale=tf.ones([num_samples, dim_z]),
  name='z'
)
thetas = [
  Gamma(
    (m_g + 1) / 2 * tf.ones([dim_z]),
    (lam ** 2) / 2 * tf.ones([dim_z])
  )
  for _ in range(image_size)
]

# The mappings from z to the input to each group generator
Ws = [
  Normal(
    loc=tf.zeros([dim_z, m_g]),
    # Should we be concerned that tf.sqrt just works here with a Jacobian adjustment?
    # Epsilon required here in order to avoid NaNs
    scale=tf.expand_dims(tf.sqrt(thetas[i] + 1e-9), 1) * tf.ones([dim_z, m_g])
  )
  for i in range(image_size)
]

# The actual inputs to each group generator
group_inputs = [tf.matmul(z, W) for W in Ws]

# The group generators. The output from these is then mapped to the mean by a
# linear transformation and mapped to the scale by a linear followed by a
# softplus.
group_generators = [
  tf.layers.dense(group_input, 32, activation=tf.nn.relu)
  # group_input
  for group_input in group_inputs
]

# The actual observations
x = Normal(
  loc=tf.concat(
    [tf.layers.dense(hidden, image_size) for hidden in group_generators],
    axis=1
  ),
  scale=tf.concat(
    [tf.layers.dense(hidden, image_size, activation=tf.nn.softplus)
     for hidden in group_generators],
    axis=1
  )
)

# INFERENCE
x_ph = tf.placeholder(tf.float32, [None, image_size * image_size])
# hidden = tf.layers.dense(x_ph, 256, activation=tf.nn.relu)
hidden = x_ph
q_z = Normal(
  loc=tf.layers.dense(hidden, dim_z),
  scale=tf.layers.dense(hidden, dim_z, activation=tf.nn.softplus),
  name='q_z'
)
q_thetas = [
  Normal(
    loc=tf.Variable(tf.random_normal([dim_z])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim_z])))
  )
  for _ in range(image_size)
]
q_Ws = [
  Normal(
    loc=tf.Variable(tf.random_normal([dim_z, m_g])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([dim_z, m_g])))
  )
  for _ in range(image_size)
]

### OPTIMIZATION
variational_approxes = {z: q_z}
for i in range(image_size):
  variational_approxes[thetas[i]] = q_thetas[i]
  variational_approxes[Ws[i]] = q_Ws[i]

inference = ed.KLqp(variational_approxes, data={x: x_ph})
# optimizer = tf.train.AdamOptimizer(learning_rate=0)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0)
inference.initialize(n_samples=1, optimizer=optimizer, auto_transform=True)

tf.global_variables_initializer().run()

# import matplotlib.pyplot as plt

# def debug(data, ixs):
#   fig, ax = plt.subplots(3, len(ixs), figsize=(12, 4))

#   # True images
#   for i, ix in enumerate(ixs):
#     ax[0, i].imshow(data[ix].reshape(image_size, image_size), vmin=0, vmax=1)
#     # ax[0, i].set_title('{:6.4f}'.format(encoder(Xvar)[2].data[0]))
#     ax[0, i].axes.xaxis.set_ticks([])
#     ax[0, i].axes.yaxis.set_ticks([])
#     # ax[0, i].set_title(i)

#   posterior_means = q_z.mean().eval(feed_dict={x_ph: data[ixs]})
#   for i, ix in enumerate(ixs):
#     ax[1, i].bar(range(dim_z), posterior_means[i])
#     # ax[0, i].set_title('{:6.4f}'.format(encoder(Xvar)[2].data[0]))
#     ax[1, i].axes.xaxis.set_ticks([])
#     ax[1, i].axes.yaxis.set_ticks([])

#   # Reconstructed images
#   z_ph = tf.placeholder(tf.float32, [None, dim_z])
#   poop = ed.copy(x, {z: z_ph})
#   reconstructions = poop.eval(feed_dict={z_ph: posterior_means})
#   for i, ix in enumerate(ixs):
#     ax[2, i].imshow(reconstructions[i].reshape(image_size, image_size), vmin=0, vmax=1)
#     # loss = torch.sum(torch.pow(fX - Xvar.view(image_size, image_size), 2))
#     # ax[2, i].set_title('{:6.4f}'.format(loss.data[0]))
#     ax[2, i].axes.xaxis.set_ticks([])
#     ax[2, i].axes.yaxis.set_ticks([])

#   ax[0, 0].set_ylabel('true image')
#   ax[1, 0].set_ylabel('z')
#   ax[2, 0].set_ylabel('reconstructed')

#   return fig

# def debug2():
#   plt.figure(figsize=(12, 4))
#   plt.suptitle('zs decoded')
#   for i in range(dim_z):
#     plt.subplot(1, dim_z, i + 1)
#     z = torch.zeros(dim_z)
#     z[i] = 1
#     plt.imshow(decoder(Variable(z)).view(image_size, image_size).data.numpy(), vmin=0, vmax=1)
#     plt.title('Component {}'.format(i))

#   plt.colorbar()

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

# def debug_incoming_weights():
#   fig, ax = plt.subplots(1, image_size, figsize=(12, 4))

#   # See https://matplotlib.org/examples/color/colormaps_reference.html
#   cmap = 'bwr'
#   for i, m in enumerate(decoder.latent_to_group_maps):
#     ax[i].imshow(torch.stack([m.weight.data for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
#     ax[i].set_title('group {}'.format(i))
#     ax[i].set_xlabel('z_i')
#     ax[i].axes.xaxis.set_ticks(range(dim_z))
#     ax[i].axes.yaxis.set_ticks([])

#   ax[0].set_ylabel('learned weights')

#   return fig

# def debug_outgoing_weights():
#   fig, ax = plt.subplots(1, dim_z, figsize=(12, 4))

#   # rows correspond to groups and cols correspond to z_i's
#   col_norms = torch.stack([
#     torch.sqrt(torch.sum(torch.pow(m.weight.data, 2), dim=0))
#     for m in decoder.latent_to_group_maps
#   ])

#   # See https://matplotlib.org/examples/color/colormaps_reference.html
#   cmap = 'bwr'
#   for i in range(dim_z):
#     ax[i].imshow(torch.stack([col_norms[:, i] for _ in range(image_size)]).squeeze(), vmin=-0.5, vmax=0.5, cmap=cmap)
#     ax[i].set_title('z_{}'.format(i))
#     ax[i].set_xlabel('groups')
#     ax[i].axes.xaxis.set_ticks(range(image_size))
#     ax[i].axes.yaxis.set_ticks([])

#   return fig

n_epoch = 1000000
# debug_interval = 5000

for epoch in range(1, n_epoch + 1):
  print("Epoch: {0}".format(epoch))
  info_dict = inference.update(feed_dict={x_ph: X})
  avg_loss = info_dict['loss']# / num_samples
  print("-log p(x) <= {:0.4f}".format(avg_loss))

  # if epoch % debug_interval == 0:
  #   debug(X, [0, 1, 2, 3, 4, 5, 6, 7])
  #   plt.show()
