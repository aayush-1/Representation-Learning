import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt


#input data N*d
X1 = np.random.multivariate_normal([5,-2], [[2,0],[0,2]], 1000)
X2=np.random.multivariate_normal([1,6], [[4,0],[0,0.5]], 1000)
X=np.vstack((X1,X2))
per=np.random.permutation(X.shape[0])
X=X[per]
N=X.shape[0]



#initialization
K=2
samples_K=[]
for i in range(K):
	a=X[int(i*N/K):int((i+1)*N/K)];
	samples_K.append(a) # list

mean_K=[]
for i in range(K):
	a=np.mean(samples_K[i],axis=0)
	mean_K.append(a) # np array



variance_K=[]
for i in range(K):
	a=np.var(samples_K[i],axis=0)
	a=np.diagflat(a)
	variance_K.append(a) 


Pi_K=[samples_K[i].shape[0]/N for i in range(K)]


log_error_old=0
log_error=100
error=abs(log_error-log_error_old)


while(error>0.001):

	posterior=np.zeros((N,K))

	for k in range(K):
		for n in range(N):
			p1=Pi_K[k]*multivariate_normal.pdf(X[n],mean=mean_K[k],cov=variance_K[k])
			p2=np.sum(np.array([Pi_K[j]*multivariate_normal.pdf(X[n],mean=mean_K[j],cov=variance_K[j]) for j in range(K)]))
			posterior[n][k]=p1/p2


	N_k=[]
	for k in range(K):
		N_k.append(np.sum([posterior[i][k] for i in range(N)]))


	for k in range(K):
		mean_K[k]=(np.sum([posterior[i][k]*X[i] for i in range(N)],axis=0))/N_k[k]

	Pi_K=[N_k[i]/N for i in range(K)]

	for k in range(K):
		a=np.zeros((K,K))
		b=np.zeros((K,K))
		for n in range(N):
			x_u=np.array(X[n])-mean_K[k]
			for i in range(X.shape[1]):
				for j in range(X.shape[1]):
					a[i][j]=x_u[i]*x_u[j]*posterior[n][k];
			b=b+a
		variance_K[k]=b/N_k[k]
	print("mean = ",mean_K)

	log_error=0
	for n in range(N):
		log_error+=np.log10(np.sum(np.array([Pi_K[j]*multivariate_normal.pdf(X[n],mean=mean_K[j],cov=variance_K[j]) for j in range(K)])))
	error=abs(log_error-log_error_old)
	print("error = ",error)

	log_error_old=log_error+0.0


print("mean = ",mean_K)

plt.plot(X[:,0],X[:,1], 'o', color='g')
for i in range(K):
	print(mean_K[i])
	plt.plot(mean_K[i][0],mean_K[i][1],'d',color='b')

plt.show()




















