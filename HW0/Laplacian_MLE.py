import numpy as np
import matplotlib.pyplot as plt
mu=0
b=10
samples = np.random.laplace(mu,b, 10000)

def sort(A):
	for i in range(len(A)): 
	      
	    # Find the minimum element in remaining  
	    # unsorted array 
	    min_idx = i 
	    for j in range(i+1, len(A)): 
	        if A[min_idx] > A[j]: 
	            min_idx = j 
	              
	    # Swap the found minimum element with  
	    # the first element         
	    A[i], A[min_idx] = A[min_idx], A[i] 
	return A

def Laplace_MLE(samples):
	sort(samples)
	n=len(samples)
	if n%2==0:
		mu= (samples[int(n/2)]+samples[int((n/2)+1)])/2
	else:
		mu= samples[(n+1)/2]
	b=sum(abs(samples-mu))/n
	return mu,b


mu_mle,b_mle=Laplace_MLE(samples)

samples_mle=np.random.laplace(mu_mle,b_mle, 10000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(samples,bins=50)
ax2.hist(samples_mle,bins=50)
plt.show()
