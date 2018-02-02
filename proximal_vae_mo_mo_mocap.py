"""Bayesian group lasso in a VAE setup on mocap data

* Point estimates on the sparse weight parameters.
* Proximal gradient steps are taken to handle the group lasso penalty.
"""

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from lib import mocap_data
from lib.distributions import Normal
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.proximal_vae import NormalPriorTheta, ProximalVAE
from lib.utils import KL_Normals, Lambda

torch.manual_seed(0)

Xraw = torch.FloatTensor(mocap_data.arr)
mins, _ = torch.min(Xraw, dim=0)
maxs, _ = torch.max(Xraw, dim=0)

# Some of these things aren't used, and we don't want to divide by zero
X = (Xraw - mins) / torch.clamp(maxs - mins, min=0.1)

dim_z = 16
dim_x = X.size(1)
num_groups = len(mocap_data.joint_order)
group_input_dim = 4

prior_theta_scale = 1
lam = 0.1
lam_adjustment = 1

num_epochs = 1000000000
mc_samples = 1
batch_size = 64

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
    make_group_generator(mocap_data.joint_dims[joint])
    for joint in mocap_data.joint_order
  ],
  group_input_dim=group_input_dim,
  dim_z=dim_z
)

prior_z = Normal(
  Variable(torch.zeros(batch_size, dim_z)),
  Variable(torch.ones(batch_size, dim_z))
)

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

lr = 1e-2
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

vae = ProximalVAE(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_z=prior_z,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws]
)

plot_interval = 5000
elbo_per_iter = []
for i in range(num_epochs):
  if i > 1000:
    stddev_multiple = 1

  # [batch_size, dim_x]
  Xvar = Variable(X[torch.randperm(X.size(0))[:batch_size]])

  info = vae.step(
    X=Xvar,
    prox_step_size=Ws_lr * lam * lam_adjustment,
    mc_samples=mc_samples
  )
  elbo_per_iter.append(info['elbo'].data[0])

  if i % plot_interval == 0 and i > 0:
    plt.figure()
    plt.plot(elbo_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO per iteration. lam = {}'.format(lam))
    plt.show()

  print('iter', i)
  print('  ELBO:', info['elbo'].data[0])
  print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
  print('    loglik_term      ', info['loglik_term'].data[0])
  print('    log p(theta)     ', info['logprob_theta'].data[0])
  print('    log p(W)         ', info['logprob_W'].data[0])
