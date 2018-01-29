import edward as ed
# import numpy as np
import tensorflow as tf

from edward.models import Normal, Gamma

import torch
from bars_data import sample_bars


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

M = 1024
d = 8    # latent dimension

m_g = 16
lam = 1

X = sample_bars_data(M)

# MODEL
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
theta = Gamma((m_g + 1) / 2 * tf.ones([d]), (lam ** 2) / 2 * tf.ones([d]))
W = Normal(
  loc=tf.zeros([d, m_g]),
  # Should we be concerned that tf.sqrt just works here with a Jacobian adjustment?
  scale=tf.expand_dims(tf.sqrt(theta), 1) * tf.ones([d, m_g])
)
# hidden = tf.layers.dense(tf.matmul(z, W), 256, activation=tf.nn.relu)
hidden = tf.matmul(z, W)
x = Normal(
  loc=tf.layers.dense(hidden, image_size * image_size),
  scale=tf.layers.dense(hidden, image_size * image_size, activation=tf.nn.softplus)
)

# INFERENCE
x_ph = tf.placeholder(tf.float32, [M, image_size * image_size])
hidden = tf.layers.dense(x_ph, 256, activation=tf.nn.relu)
q_z = Normal(
  loc=tf.layers.dense(hidden, d),
  scale=tf.layers.dense(hidden, d, activation=tf.nn.softplus)
)
q_theta = Normal(
  loc=tf.Variable(tf.random_normal([d])),
  scale=tf.nn.softplus(tf.Variable(tf.random_normal([d])))
)
q_W = Normal(
  loc=tf.Variable(tf.random_normal([d, m_g])),
  scale=tf.nn.softplus(tf.Variable(tf.random_normal([d, m_g])))
)

inference = ed.KLqp({z: q_z, theta: q_theta, W: q_W}, data={x: x_ph})
optimizer = tf.train.AdamOptimizer()
inference.initialize(optimizer=optimizer)

tf.global_variables_initializer().run()

n_epoch = 1000
for epoch in range(1, n_epoch + 1):
  print("Epoch: {0}".format(epoch))
  info_dict = inference.update(feed_dict={x_ph: X})
  avg_loss = info_dict['loss'] / M
  print("-log p(x) <= {:0.3f}".format(avg_loss))
