import pickle as pkl
import mne
import numpy as np
import os
import torch
import torchvision

from mne.minimum_norm import apply_inverse_epochs
from scipy.sparse import csc_matrix, csr_matrix, diags
from torch.autograd import Variable

from lib import sps_info
from lib.distributions import Normal
from lib.label_util import load_hcpmmp1
from lib.meg_util import load_subject_data
from lib.models import BayesianGroupLassoGenerator, NormalNet
from lib.oivae import NormalPriorTheta, OIVAE
from lib.utils import KL_Normals, Lambda

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

cond_map = sps_info.condition_map

tmin = 1.55
tmax = 3.15

epochs_fmt = "All_55-sss_%s-boot-dscale_2-nsamples_5-seed_8675309-epo.fif"
cond = 'LL'

labels = load_hcpmmp1()
labels = [l for l in labels if '???' not in l.name]


subject_name = "eric_sps_04"
sps_dir = os.environ['SPS_DIR']
subjects_dir = mne.utils.get_subjects_dir()
epochs, inv, labels_morphed = load_subject_data(subject_name, sps_dir,
                                                subjects_dir, labels,
                                                epochs_fmt=epochs_fmt,
                                                sps_dir=sps_dir,
                                                tmin=tmin, tmax=tmax,
                                                cond=cond)
# compute inverse solutions
print("applying inverse operators")
ep_inv = apply_inverse_epochs(epochs, inv, lambda2=1.0 / 9, method='MNE')
ep_inv_ndarray = np.array([np.ascontiguousarray(ep.data.T) for ep in ep_inv])

# get vertex indices for each label
src = inv['src']

roiidx = list()
vertidx = list()
n_lhverts = len(src[0]['vertno'])
n_rhverts = len(src[1]['vertno'])
n_verts = n_lhverts + n_rhverts
offsets = {'lh': 0, 'rh': n_lhverts}

for li, lab in enumerate(labels):
  if isinstance(lab, mne.Label):
    comp_labs = [lab]
  elif isinstance(lab, mne.BiHemiLabel):
    comp_labs = [lab.lh, lab.rh]

  for clab in comp_labs:
    hemi = clab.hemi
    hi = 0 if hemi == 'lh' else 1

    lverts = clab.get_vertices_used(vertices=src[hi]['vertno'])

    # gets the indices in the source space vertex array, not the huge
    # array.
    # use `src[hi]['vertno'][lverts]` to get surface vertex indices to
    # plot.
    lverts = np.searchsorted(src[hi]['vertno'], lverts)
    lverts += offsets[hemi]
    vertidx.extend(lverts)
    roiidx.extend(li*np.ones(lverts.size, dtype=np.int))

num_labels = len(labels)
M = n_verts

# construct sparse L matrix
data = np.ones(len(vertidx), dtype=np.int)
L = csc_matrix((data, (vertidx, roiidx)), shape=(M, num_labels))

vertidx = np.array(vertidx)
roiidx = np.array(roiidx)

# how to get group structure from labels
# L[i,j] \in {0,1} indicates if vertex i is in label j
# vertidx[roiidx == label_ind] gets vertex indices for label `label_ind`

# number of vertices in each group
num_groups = num_labels
group_output_dims = np.bincount(roiidx).tolist()
ep_inv_stacked = np.reshape(ep_inv_ndarray, (-1, ep_inv_ndarray[0].shape[-1]))

# permuting vertices to be with those from same group, i.e.
# idx_sorted[:group_output_dims[0]] are all vertices that belong to label 0.
idx_sorted = vertidx[np.argsort(roiidx)]

ep_inv_stacked = ep_inv_stacked[:, idx_sorted]
dim_x = ep_inv_stacked.shape[-1]

ep_inv_stacked *= 1e9
ep_inv_stacked = ep_inv_stacked.astype(np.float32)

#raise RuntimeError("grab test set!")

# VAE stuff
torch.manual_seed(8675309)

# should be cmd line arg
dim_z = 10

# dimensionalty of input to group-generator network
group_input_dim = 1

prior_theta_scale = 1.
lam = 1
lam_adjustment = 1.

num_epochs = 200
mc_samples = 1
batch_size = 4096

ep_inv_tnsr = torch.from_numpy(ep_inv_stacked)

dataloader = torch.utils.data.DataLoader(
  torch.utils.data.TensorDataset(
    ep_inv_tnsr.cuda(),
    torch.zeros(ep_inv_tnsr.size(0))
  ),
  batch_size=batch_size,
  shuffle=True
)

# This value adjusts the impact of our learned variances in the sigma_net of
# `inference_net` below. Zero means that the model has no actual connection to
# the output and therefore the standard deviation defaults to the minimum. One
# means that we're learning the real model. This value is flipped to 1 after
# some number of iterations.
stddev_multiple = 0.1

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

def make_group_generator(group_output_dim):
  # Note that this Variable is NOT going to show up in `net.parameters()` and
  # therefore it is implicitly free from the ridge penalty/p(theta) prior.
  log_sigma = Variable(
    torch.log(1e-2 * torch.ones(group_output_dim).type(torch.cuda.FloatTensor if use_cuda else torch.FloatTensor)),
    requires_grad=True
  )
  return NormalNet(
    mu_net=torch.nn.Linear(group_input_dim, group_output_dim),
    sigma_net=Lambda(
      lambda x, log_sigma: torch.exp(log_sigma.expand(x.size(0), -1)) + 1e-3,
      extra_args=(log_sigma,)
    )
  )

generative_net = BayesianGroupLassoGenerator(
  group_generators=[make_group_generator(gs) for gs in group_output_dims],
  group_input_dim=group_input_dim,
  dim_z=dim_z
)

prior_z = Normal(
  Variable(torch.zeros(1, dim_z)), 
  Variable(torch.ones(1, dim_z))
)

if use_cuda:
  inference_net.cuda()
  generative_net.cuda()
  prior_z.mu = prior_z.mu.cuda()
  prior_z.sigma = prior_z.sigma.cuda()

lr = 1e-3
optimizer = torch.optim.Adam([
  {'params': inference_net.parameters(), 'lr': lr},
  # {'params': [inference_net_log_stddev], 'lr': lr},
  {'params': generative_net.group_generators_parameters(), 'lr': lr},
  {'params': [gen.sigma_net.extra_args[0] for gen in generative_net.group_generators], 'lr': lr}
])

Ws_lr = 1e-6
optimizer_Ws = torch.optim.SGD([
  {'params': [generative_net.Ws], 'lr': Ws_lr, 'momentum': 0}
])

vae = OIVAE(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_z=prior_z,
  prior_theta=NormalPriorTheta(prior_theta_scale),
  lam=lam,
  optimizers=[optimizer, optimizer_Ws]
)

print('starting opt')
plot_interval = 5000
elbo_per_iter = []
iteration = 0
for epoch in range(num_epochs):
  for Xbatch, _ in dataloader:
    if iteration > 1000:
      stddev_multiple = 1

    info = vae.step(
      X=Variable(Xbatch),
      prox_step_size=Ws_lr * lam * lam_adjustment,
      mc_samples=mc_samples
    )

    elbo_per_iter.append(info['elbo'].data[0])

    if iteration % plot_interval == 0 and iteration > 0:
      plt.figure()
      plt.plot(elbo_per_iter)
      plt.xlabel('iteration')
      plt.ylabel('ELBO')
      plt.title('ELBO per iteration. lam = {}'.format(lam))
      plt.show()

    print('epoch', epoch, 'iter', iteration)
    print('  ELBO:', info['elbo'].data[0])
    print('    -KL(q(z) || p(z))', -info['z_kl'].data[0])
    print('    loglik_term      ', info['loglik_term'].data[0])
    print('    log p(theta)     ', info['logprob_theta'].data[0])
    print('    log p(W)         ', info['logprob_W'].data[0])
    
    iteration += 1

# test log-lik on test set
#test_log_lik = 
#print("test log-lik: %.4f" % test_log_lik)

# save results for visualization script
# Ws is num_groups x dim_z x group_input_dim
#save_dict = {'test-log-lik': test_log_lik,
#             'Ws': generative_net.Ws,
#             'label_names': [l.name for l in labels]}
#
#outdir =
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#
#outfn = 
#
#with open(outfn, 'wb') as f:
#    pkl.dump(save_dict, f)
