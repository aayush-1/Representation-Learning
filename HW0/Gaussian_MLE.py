import numpy as np
import matplotlib.pyplot as plt
m, var = 5, 2 
samples = np.random.normal(m, var, 10000)


def gaussian_MLE(samples):
	mean_mle=np.mean(samples)
	var_mle=np.sqrt(np.var(samples))
	return mean_mle,var_mle


m_mle,var_mle=gaussian_MLE(samples)

samples_mle=np.random.normal(m_mle, var_mle, 10000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(samples,bins=50)
ax2.hist(samples_mle,bins=50)
plt.show()


