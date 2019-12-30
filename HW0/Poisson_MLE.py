import numpy as np
import matplotlib.pyplot as plt
lam = 5  
samples = np.random.poisson(lam, 10000)


def Poisson_MLE(samples):
	lam_mle=np.mean(samples)
	return lam_mle


lam_mle=Poisson_MLE(samples)

samples_mle=np.random.poisson(lam_mle, 10000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(samples,bins=50)
ax2.hist(samples_mle,bins=50)
plt.show()
