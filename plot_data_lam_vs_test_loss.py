import pickle

import matplotlib.pyplot as plt


nums_train_samples = [2, 4, 8, 16, 32, 64, 128, 256, 512]
epoch = 1000
lams = [0.0, 0.001, 0.01, 0.1, 1]
# lams = [0.0, 0.1]

res = pickle.load(open('poop.p', 'rb'))

plt.figure()
for j in range(len(lams)):
  test_losses = [res[(i, j)][0][epoch] for i in range(len(nums_train_samples))]
  # plt.plot(nums_train_samples, test_losses)
  plt.semilogx(nums_train_samples, test_losses, basex=2, marker='o')
plt.xlabel('num. training samples')
plt.ylabel('test reconstruction loss')
plt.legend(['lambda = {}'.format(lam) for lam in lams])
plt.title('epoch = {}'.format(epoch))
plt.show()
