import numpy as np
import matplotlib.pyplot as plt
lam=5
samples = np.random.exponential(lam, 10000)


def exponential_MLE(samples):
	lam_mle=np.mean(samples)
	return lam_mle


lam_mle=exponential_MLE(samples)

samples_mle=np.random.exponential(lam_mle, 10000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(samples,bins=50)
ax2.hist(samples_mle,bins=50)
plt.show()
