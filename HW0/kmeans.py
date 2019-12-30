import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt



# Data points are of format N*d 

# 3 colours image
image=plt.imread("aa.png")
X=np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2]))




K=8
epsilon=0.14
error=10
#Initialize centroids
a=np.random.randint(X.shape[0], size=K)
c=X[a,:]



c_old=X[a,:]

dist=np.zeros(K)

N=X.shape[0]
f=X.shape[1]


while error>epsilon:
	cluster=np.zeros(N)
	for i in range(N):
		for j in range(K):
			for k in range(f):
				dist[j]+=(c[j][k]-X[i][k])**2
			dist[j]=np.sqrt(dist[j])
		cluster[i]=np.argmin(dist)
	clusters=[]
	for j in range(K):
		clusters.append(X[cluster==j])

	for i in range(K):
		for j in range(f):
			if(clusters[i][:,j].shape[0]!=0):
				c[i][j]=np.mean(clusters[i][:,j])	
	error=0
	for i in range(K):
		err=0
		for j in range(f):
			err=err+(c[i][j]-c_old[i][j])**2
		err=np.sqrt(err)
		error=error+err
	for j in range(K):
		for k in range(f):
			c_old[j][k]=c[j][k]
	print('Error = ',error)


for i in range(N):
	for j in range(K):
		if(int(cluster[i]) == j):
			X[i]= c[j]
			break;



Xnew=np.reshape(X,(image.shape[0],image.shape[1],image.shape[2]))

for i in range(K):
	print("Centroid ",i," = " ,c[i,:])
for i in range(K):
	print("cluster ",i," = ",clusters[i].shape)


plt.imshow(Xnew)
plt.show()





		

		

