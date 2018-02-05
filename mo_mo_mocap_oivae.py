"""Bayesian group lasso in a VAE setup on mocap data

* Point estimates on the sparse weight parameters.
* Proximal gradient steps are taken to handle the group lasso penalty.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

# Necessary for the torch.utils.data stuff.
import torchvision

from lib import mocap_data
from lib.distributions import Normal
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.oivae import NormalPriorTheta, OIVAE
from lib.utils import Lambda


torch.manual_seed(0)

dim_z = 16
group_input_dim = 4

prior_theta_scale = 1
lam = 1
lam_adjustment = 1

num_epochs = 1000000000
mc_samples = 1
batch_size = 64

# These are some walking sequences from subject 7.
trials = [
  (7, 1),
  (7, 2),
  (7, 3),
  (7, 4),
  (7, 5),
  (7, 6),
  (7, 7),
  (7, 8),
  (7, 9),
  (7, 10),
  (7, 11),
  (7, 12)
]

trials_data = [
  mocap_data.load_mocap_trial(subject, trial)
  for subject, trial in trials
]
joint_order, joint_dims, _ = trials_data[0]

# We remove the first three components since those correspond to root position
# in 3d space.
joint_dims['root'] = joint_dims['root'] - 3
Xraw = torch.FloatTensor(
  # Chain all of the different lists together across the trials
  list(itertools.chain(*[arr for _, _, arr in trials_data]))
)[:, 3:]

# Normalize each of the channels to be within [0, 1].
mins, _ = torch.min(Xraw, dim=0)
maxs, _ = torch.max(Xraw, dim=0)

# Some of these things aren't used, and we don't want to divide by zero
X = (Xraw - mins) / torch.clamp(maxs - mins, min=0.1)

dataloader = torch.utils.data.DataLoader(
  # TensorDataset is stupid. We have to provide two tensors.
  torch.utils.data.TensorDataset(X, torch.zeros(X.size(0))),
  batch_size=batch_size,
  shuffle=True
)

dim_x = X.size(1)
num_groups = len(joint_order)

# This value adjusts the impact of our learned variances in the sigma_net of
# `inference_net` below. Zero means that the model has no actual connection to
# the output and therefore the standard deviation defaults to the minimum. One
# means that we're learning the real model. This value is flipped to 1 after
# some number of iterations.
stddev_multiple = 0.1

# inference_net_log_stddev = Variable(
#   torch.log(1e-2 * torch.ones(dim_z)),
#   requires_grad=True
# )
inference_net = NormalNet(
  mu_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_x, dim_z)
  ),

  # Fixed standard deviation
  # sigma_net=Lambda(lambda x: 1e-3 * Variable(torch.ones(x.size(0), dim_z)))

  # Learned constant standard deviation
  # sigma_net=Lambda(
  #   lambda x: torch.exp(inference_net_log_stddev.expand(x.size(0), -1)) + 1e-3
  # )

  # Learned standard deviation as a function of the input
  sigma_net=torch.nn.Sequential(
    # inference_net_base,
    torch.nn.Linear(dim_x, dim_z),
    Lambda(torch.exp),
    Lambda(lambda x: x * stddev_multiple + 1e-3)
  )
)

def make_group_generator(output_dim):
  # Note that this Variable is NOT going to show up in `net.parameters()` and
  # therefore it is implicitly free from the ridge penalty/p(theta) prior.
  log_sigma = Variable(
    torch.log(1e-2 * torch.ones(output_dim)),
    requires_grad=True
  )
  return NormalNet(
    mu_net=torch.nn.Linear(group_input_dim, output_dim),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )

generative_net = BayesianGroupLassoGenerator(
  group_generators=[
    make_group_generator(joint_dims[joint])
    for joint in joint_order
  ],
  group_input_dim=group_input_dim,
  dim_z=dim_z
)

def debug_z_by_group_matrix():
  fig, ax = plt.subplots()
  W_col_norms = torch.sqrt(
    torch.sum(torch.pow(generative_net.Ws.data, 2), dim=2)
  )
  ax.imshow(W_col_norms.t().numpy(), aspect='equal')
  ax.set_ylabel('dimensions of z')
  ax.set_xlabel('group generative nets')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  ax.xaxis.set_ticks(np.arange(len(joint_order)))
  ax.xaxis.set_ticklabels(joint_order, rotation='vertical')
  plt.show()
  # plt.title('Connectivity between dimensions of z and group generator networks')

# lr = 1e-3
# optimizer = torch.optim.Adam([
#   {'params': inference_net.parameters(), 'lr': lr},
#   {'params': generative_net.parameters(), 'lr': lr}
# ])

# lr = 1e-4
# Ws_lr = 5e-3
# momentum = 0.9
# optimizer = torch.optim.SGD([
#   {'params': inference_net.parameters(), 'lr': lr, 'momentum': momentum},
#   # {'params': generative_net.parameters(), 'lr': lr, 'momentum': momentum}
#   {'params': generative_net.group_generators_parameters(), 'lr': lr, 'momentum': momentum},
#   {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
# ])

lr = 1e-3
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
  # {'params': [inference_net_log_stddev], 'lr': lr},
  {'params': generative_net.group_generators_parameters(), 'lr': lr},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr}
])

Ws_lr = 1e-4
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
])

prior_z = Normal(
  Variable(torch.zeros(1, dim_z)),
  Variable(torch.ones(1, dim_z))
)

vae = OIVAE(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_z=prior_z,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws]
)

plot_interval = 50000
elbo_per_iter = []
iteration = 0
for epoch in range(num_epochs):
  for Xbatch, _ in dataloader:
    if iteration > 500:
      stddev_multiple = 1

    info = vae.step(
      X=Variable(Xbatch),
      prox_step_size=Ws_lr * lam * lam_adjustment,
      mc_samples=mc_samples
    )
    elbo_per_iter.append(info['elbo'].data[0])

    if iteration % plot_interval == 0 and iteration > 0:
      debug_z_by_group_matrix()

      plt.figure()
      plt.plot(elbo_per_iter)
      plt.xlabel('iteration')
      plt.ylabel('ELBO')
      plt.title('ELBO per iteration. lam = {}'.format(lam))
      plt.show()

    print('epoch', epoch, 'iteration', iteration)
    print('  ELBO:', info['elbo'].data[0])
    print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
    print('    loglik_term      ', info['loglik_term'].data[0])
    print('    log p(theta)     ', info['logprob_theta'].data[0])
    print('    log p(W)         ', info['logprob_W'].data[0])

    iteration += 1

print('Outputting reconstructions AMC file...')
def poop():
  reconstructed = generative_net(inference_net(Variable(X)).mu).mu

poop()
