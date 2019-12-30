import math
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2
    
def relu_dev(x):
    if x>0:
        return 1
    else:
        return 0.11

def relu(x):
    if(x>0):
        return x
    elif(x<=0):
        return 0


def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid_deriv(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm * (1. - sigm)


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def MLP(mlp_input):
    layers_MLP[0]=mlp_input
    for i in range(0,no_layers-1):
        layers_MLP[i+1]=np.matmul(np.transpose(layers_MLP[i]),weights_MLP[i][1:,:])+weights_MLP[i][0,:]
        layers_MLP[i+1]=sigmoid(layers_MLP[i+1])
    return layers_MLP


def MLP_backprop(y_train,p,learning_rate):
    x=0
    for i in range(no_layers-2,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_layer[i+1]))
            x=1
        else:
            dy=dx+0.0
            dx=np.zeros((size_layer[i+1]))

        if i==no_layers-2:
            dx=(-2)*(y_train-(layers_MLP[i+1]))
        else:
            for m in range(0,size_layer[i+1]):
                dx[m]+=np.dot(dy,weights_MLP[i+1][m+1,:]) - ((p)/(layers_MLP[i+1][m]))+ ((1-p)/(1-layers_MLP[i+1][m]))
        for m in range(size_layer[i]):
            weights_MLP[i][1+m,:]=weights_MLP[i][1+m,:]-learning_rate*dx*layers_MLP[i][m]*sigmoid_deriv(layers_MLP[i+1])
        weights_MLP[i][0,:]=weights_MLP[i][0,:]-learning_rate*dx*sigmoid_deriv(layers_MLP[i+1])



def train(x_train,y_train):
    epochs=1
    learning_rate=0.001
    error_epoch=[]
    accuracy_epoch=[]
    epoch_list=[]
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        acc=0
        for i in range(x_train.shape[0]):
            MLP(x_train[i,:])
            dec_loss=0
            # loss = latent loss + generation loss
            latent_loss=0
            p=0.1
            for j in range(y_train.shape[1]):
                dec_loss+=(y_train[i,j]-layers_MLP[no_layers-1][j])**2
            for j in range(size_layer[no_layers-2]):
                latent_loss+= p*np.log(p/layers_MLP[no_layers-2][j]) + (1-p)*np.log((1-p)/(1-layers_MLP[no_layers-2][j])) 
            error+= dec_loss + latent_loss
            print(i)
            print("dec_loss: ",dec_loss)
            print("latent_loss: ",latent_loss)
            MLP_backprop(y_train[i,:],p,learning_rate)
        print("error: ",error/x_train.shape[0])
        error_epoch.append(error/x_train.shape[0])
        #accuracy_epoch.append(acc/x_train.shape[0])
        print("error per epoch = ",error/x_train.shape[0])
        #print("accuracy per epoch = ",acc/x_train.shape[0])
        epoch_list.append(epoch+1)




def test(x_test):
    output=np.zeros((x_test.shape))
    for i in range(8):
        input1 = np.reshape(x_test[i], [14*14])
        output1=MLP(input1)
        output1=output1[no_layers-1]
        input1 = np.reshape(input1, [-1, 14,14])
        output[i]=output1
    imgs = np.concatenate([x_test, output])
    imgs = imgs.reshape((4, 4,14,14))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('Sparse_AutoEncoder_output.png')
    plt.show()

(x_tr, _), (x_te, _) = mnist.load_data()


x_train=np.zeros((x_tr.shape[0],14,14))

for i in range(x_tr.shape[0]):
    x_train[i]=cv2.resize(x_tr[i], (14, 14),interpolation=cv2.INTER_CUBIC)

x_test=np.zeros((x_te.shape[0],14,14))

for i in range(x_te.shape[0]):
    x_test[i]=cv2.resize(x_te[i], (14, 14),interpolation=cv2.INTER_CUBIC)





image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size*image_size])
x_test = np.reshape(x_test, [-1, image_size*image_size])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

perm=np.random.permutation(x_train.shape[0])
x_train=x_train[perm]



MLP_input_size=x_train.shape[1]  #depends on data
no_layers=1+2
size_layer=[]
size_layer.append(MLP_input_size)
size_layer.append(250)
output_size=x_train.shape[1] #depends on data
size_layer.append(output_size)

weights_MLP=[]
for i in range(0,no_layers-1):
    weights_MLP.append(np.random.normal(0,0.1,(size_layer[i]+1,size_layer[i+1])))
layers_MLP=[]
for i in range(0,no_layers):
    layers_MLP.append(np.zeros((size_layer[i])))



#TRAINING

# optimization method - gradient descent
train(x_train,x_train)

#np.save('weights',weights_MLP)
#TESTING

test(x_test[:8])
