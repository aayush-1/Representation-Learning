import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt


# Data points are of format N*d 


# 3 colours image
image=plt.imread("aa.png")


print(image.shape)
X=np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2]))
print(X)
X1=np.mean(X,axis=0)
X1= X-X1

cov=np.matmul(np.transpose(X1),X1)/(X1.shape[0])

e_val,e_vec = np.linalg.eig(cov)

newdata=np.matmul(X1,e_vec)

print(newdata)

Xnew=np.reshape(newdata,(image.shape[0],image.shape[1],image.shape[2]))
plt.imshow(Xnew)
plt.show()
