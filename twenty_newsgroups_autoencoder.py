from collections import Counter

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset

from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
# from sklearn.feature_extraction.text import TfidfVectorizer

from lib.common import FirstLayerSparseDecoder


print('Loading 20 newsgroups...')
# newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
newsgroups = fetch_20newsgroups_vectorized(subset='all', remove=('headers', 'footers', 'quotes'))
ix = np.argsort(newsgroups.target)
Xfat = newsgroups.data[ix]
y = newsgroups.target[ix]
group_counter = Counter(y)
group_counts = [group_counter[i] for i in range(20)]

print('Fitting SVM for TF-IDF feature selection...')
# Smaller C = stronger sparsity penalty
lsvc = LinearSVC(loss='squared_hinge', penalty='l1', dual=False, C=0.01).fit(Xfat, y)
X_sel = SelectFromModel(lsvc, threshold=1e-3, prefit=True).transform(Xfat)
X = torch.from_numpy(X_sel.astype(np.float32).T.toarray())

print(X.size())

data_loader = torch.utils.data.DataLoader(
  TensorDataset(X, torch.arange(X.size(0))),
  batch_size=X.size(0),
  shuffle=True
)

torch.manual_seed(0)

dim_z = 64
num_epochs = 100000
lam = 0.0

encoder = torch.nn.Linear(X.size(1), dim_z)

def make_linear_decoder():
  group_input_dim = dim_z
  return FirstLayerSparseDecoder(
    [torch.nn.Linear(group_input_dim, i, bias=False) for i in group_counts],
    [group_input_dim] * 20,
    dim_z
  )

def make_nonlinear_decoder():
  group_input_dim = 1
  return FirstLayerSparseDecoder(
    [
      torch.nn.Sequential(
        torch.nn.Linear(group_input_dim, i),
        torch.nn.Tanh(),
        torch.nn.Linear(i, i),
        torch.nn.Tanh(),
        torch.nn.Linear(i, i)
      )
      for i in group_counts
    ],
    [group_input_dim] * 20,
    dim_z
  )

decoder = make_linear_decoder()
# decoder = make_nonlinear_decoder()

# lr = 1e-4
# optimizer = torch.optim.Adam([
#   {'params': encoder.parameters(), 'lr': lr},
#   {'params': decoder.parameters(), 'lr': lr}
# ])

lr = 1e-4
momentum = 0.9
optimizer = torch.optim.SGD([
  {'params': encoder.parameters(), 'lr': lr, 'momentum': momentum},
  {'params': decoder.parameters(), 'lr': lr, 'momentum': momentum}
])

# Calculate the reconstruction loss on the given data
def reconstruction_loss(data):
  Xvar = Variable(data)
  reconstructed = decoder(encoder(Xvar))
  if torch.sum(torch.abs(reconstructed - reconstructed[0])).data[0] / Xvar.size(0) <= 1e-3:
    print('solution has collapsed!')

  residual = reconstructed - Xvar
  return torch.sum(torch.pow(residual, 2)) / Xvar.size(0)

for epoch in range(num_epochs):
  for batch_idx, (data, target) in enumerate(data_loader):
    train_loss = reconstruction_loss(data)
    sparsity_penalty = lam * decoder.group_lasso_penalty()

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    decoder.proximal_step(lr * lam)
    print('epoch: {} [{}/{} ({:.0f}%)]'.format(
      epoch,
      batch_idx * len(data),
      len(data_loader.dataset),
      100. * batch_idx / len(data_loader)
    ))
    print('  reconstruction loss:', train_loss.data[0])
    print('  regularization:     ', sparsity_penalty.data[0])
    print('  combined loss:      ', (train_loss + sparsity_penalty).data[0])
