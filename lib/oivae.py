import torch
from torch.autograd import Variable

from .distributions import Normal
from .utils import KL_Normals


class NormalPriorTheta(object):
  """A distribution that places a zero-mean Normal distribution on all of the
  `group_generators` in a BayesianGroupLassoGenerator."""

  def __init__(self, sigma):
    self.sigma = sigma

  def logprob(self, module):
    return sum(
      Normal(
        torch.zeros_like(param),
        self.sigma * torch.ones_like(param)
      ).logprob(param)
      for gen in module.group_generators
      for param in gen.parameters()
    )

class OIVAE(object):
  def __init__(
      self,
      inference_model,
      generative_model,
      prior_z,
      prior_theta,
      lam,
      optimizers
  ):
    self.inference_model = inference_model
    self.generative_model = generative_model
    self.prior_z = prior_z
    self.prior_theta = prior_theta
    self.lam = lam
    self.optimizers = optimizers

  def step(self, X, prox_step_size, mc_samples):
    batch_size = X.size(0)

    # [batch_size, dim_z]
    q_z = self.inference_model(X)

    # KL divergence is additive across independent joint distributions, so this
    # works appropriately.
    z_kl = KL_Normals(q_z, self.prior_z.expand_as(q_z)) / batch_size

    # [batch_size * mc_samples, dim_z]
    z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)
    Xrep = Variable(X.data.repeat(mc_samples, 1))
    loglik_term = (
      self.generative_model(z_sample).logprob(Xrep)
      / mc_samples
      / batch_size
    )

    # Prior over the weights of the group generative nets.
    logprob_theta = self.prior_theta.logprob(self.generative_model)

    # Prior over the first layer Ws in the generative model.
    logprob_W = -self.lam * self.generative_model.group_lasso_penalty()

    # Proximal gradient descent requires differentiating through only the
    # non-group lasso terms, hence the separation between the loss
    # (differentiated) and the ELBO (not differentiated).
    loss = -1.0 * (-z_kl + loglik_term + logprob_theta)
    elbo = -loss + logprob_W

    for opt in self.optimizers:
      opt.zero_grad()
    loss.backward()
    for opt in self.optimizers:
      opt.step()
    if self.lam > 0:
      self.generative_model.proximal_step(prox_step_size)

    return {
      'q_z': q_z,
      'z_kl': z_kl,
      'z_sample': z_sample,
      'loglik_term': loglik_term,
      'logprob_theta': logprob_theta,
      'logprob_W': logprob_W,
      'loss': loss,
      'elbo': elbo
    }
