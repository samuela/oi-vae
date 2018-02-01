import torch


# def softplus(x):
#   return torch.log(torch.exp(x) + 1)

# Stolen from upstream master
def split(tensor, split_size_or_sections, dim=0):
  """Splits the tensor into chunks.
  If ``split_size_or_sections`` is an integer type, then ``tensor`` will be
  split into equally sized chunks (if possible).
  Last chunk will be smaller if the tensor size along a given dimension
  is not divisible by ``split_size``.
  If ``split_size_or_sections`` is a list, then ``tensor`` will be split
  into ``len(split_size_or_sections)`` chunks with sizes in ``dim`` according
  to ``split_size_or_sections``.

  Arguments:
      tensor (Tensor): tensor to split.
      split_size_or_sections (int) or (list(int)): size of a single chunk or
      list of sizes for each chunk
      dim (int): dimension along which to split the tensor.
  """
  if dim < 0:
    dim += tensor.dim()
  dim_size = tensor.size(dim)

  if isinstance(split_size_or_sections, int):
    split_size = split_size_or_sections
    num_splits = (dim_size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - dim_size)

    def get_split_size(i):
      return split_size if i < num_splits - 1 else last_split_size
    return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i)))
                 for i in range(0, num_splits))

  else:
    if dim_size != sum(split_size_or_sections):
      raise ValueError("Sum of split sizes exceeds tensor dim")
    split_indices = [0] + split_size_or_sections
    split_indices = torch.cumsum(torch.Tensor(split_indices), dim=0)

    return tuple(
      tensor.narrow(int(dim), int(start), int(length))
      for start, length in zip(split_indices, split_size_or_sections))

class Bijection(object):
  def __init__(
      self,
      forward,
      inverse,
      forward_log_abs_det_jacobian,
      inverse_log_abs_det_jacobian
  ):
    self.forward = forward
    self.inverse = inverse
    self.forward_log_abs_det_jacobian = forward_log_abs_det_jacobian
    self.inverse_log_abs_det_jacobian = inverse_log_abs_det_jacobian

  def __call__(self, x):
    return self.forward(x)

def invert_bijection(f):
  return Bijection(
    forward=f.inverse,
    inverse=f.forward,
    forward_log_abs_det_jacobian=f.inverse_log_abs_det_jacobian,
    inverse_log_abs_det_jacobian=f.forward_log_abs_det_jacobian
  )

softplus = Bijection(
  forward=lambda x: torch.log(torch.exp(x) + 1),
  inverse=lambda y: torch.log(torch.exp(y) - 1),
  forward_log_abs_det_jacobian=lambda x: torch.sum(x - torch.log(torch.exp(x) + 1)),
  inverse_log_abs_det_jacobian=lambda y: torch.sum(y - torch.log(torch.abs(torch.exp(y) - 1)))
)
