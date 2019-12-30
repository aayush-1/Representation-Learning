import numpy as np
import matplotlib.pyplot as plt
# Binomial
n, p = 50, .4
samples = np.random.binomial(n, p, 100000)
print(samples)
def binomial_MLE(samples,n):
	p_mle=np.mean(samples)/n
	n_mle=n
	return p_mle,n_mle


p_mle,n_mle=binomial_MLE(samples,n)
samples_mle=np.random.binomial(n_mle,p_mle, 100000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(samples,bins=50)
ax2.hist(samples_mle,bins=50)
plt.show()
