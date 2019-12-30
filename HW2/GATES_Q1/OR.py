import math
import numpy as np
import matplotlib.pyplot as plt

#input - 1,2

    
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


def MLP(mlp_input,no_layers,size_layer,weights_MLP,layers_MLP):
    layers_MLP[0]=mlp_input
    for i in range(0,no_layers-1):
        layers_MLP[i+1]=np.matmul(np.transpose(layers_MLP[i]),weights_MLP[i][1:,:])+weights_MLP[i][0,:]
        layers_MLP[i+1]=np.tanh(layers_MLP[i+1])
        #for h in range(0,size_layer[i+1]):
        #	layers_MLP[i+1][h]=relu(layers_MLP[i+1][h])	
    layers_MLP[no_layers-1]=softmax(layers_MLP[no_layers-1])
    return layers_MLP


def MLP_backprop(y_train,no_layers,size_layer,weights_MLP,layers_MLP,learning_rate):
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
                for k in range(0,size_layer[i+2]):
                    dx[m]+=dy[k]*weights_MLP[i+1][m+1,k]
        for m in range(size_layer[i]):
            for k in range(size_layer[i+1]):
                weights_MLP[i][1+m,k]=weights_MLP[i][1+m,k]-learning_rate*dx[k]*layers_MLP[i][m]*tanh_deriv(layers_MLP[i+1][k])
        for k in range(size_layer[i+1]):
            weights_MLP[i][0,k]=weights_MLP[i][0,k]-learning_rate*dx[k]*tanh_deriv(layers_MLP[i+1][k])
        #learning_rate=learning_rate*0.1
    return weights_MLP

def train(x_train,y_train,MLP_input_size,no_hidden_layers,size_layer,weights_MLP,layers_MLP):
    epochs=10
    learning_rate=0.1
    error_epoch=[]
    accuracy_epoch=[]
    epoch_list=[]

    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        #if(epoch>7):
        #	learning_rate=0.01
        error=0
        acc=0
        for i in range(x_train.shape[0]):
            layers_MLP=MLP(x_train[i,:],no_layers,size_layer,weights_MLP,layers_MLP)
            d=0
            for j in range(y_train.shape[1]):
                d+=(y_train[i,j]-layers_MLP[no_layers-1][j])**2

            error+=d
            if(np.argmax(layers_MLP[no_layers-1])==np.argmax(y_train[i,:])):
                acc+=1
            weights_MLP=MLP_backprop(y_train[i,:],no_layers,size_layer,weights_MLP,layers_MLP,learning_rate)


        error_epoch.append(error/x_train.shape[0])
        accuracy_epoch.append(acc/x_train.shape[0])
        print("error per epoch = ",error/x_train.shape[0])
        print("accuracy per epoch = ",acc/x_train.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.savefig('OR_error.png')
    plt.show(f1)
    

    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.savefig('OR_accuracy.png')  
    plt.show(f2)
      




def test(input1):
	output=MLP(input1,no_layers,size_layer,weights_MLP,layers_MLP)

	print(input1," : ",np.argmax(output[no_layers-1]))




x_train=np.array([[0,0],[0,1],[1,0],[1,1]])
a=x_train
for i in range(10):
	x_train=np.vstack((x_train,a))

y_tr=np.array([0,1,1,1])
b=y_tr

for i in range(10):
	y_tr=np.hstack((y_tr,b))

y_train = np.zeros((y_tr.shape[0], 2))
y_train[np.arange(y_tr.shape[0]), y_tr] = 1

perm=np.random.permutation(x_train.shape[0])
x_train=x_train[perm]
y_train=y_train[perm]


MLP_input_size=x_train.shape[1]  #depends on data
no_layers=1+2
size_layer=[]
size_layer.append(MLP_input_size)
size_layer.append(2)
output_size=y_train[0].shape[0] #depends on data
print(output_size)
size_layer.append(output_size)

weights_MLP=[]
#weights_MLP.append(np.random.normal(0,0.1,(MLP_input_size+1, size_layer[1])))
for i in range(0,no_layers-1):
    weights_MLP.append(np.random.normal(0,0.1,(size_layer[i]+1,size_layer[i+1])))
#weights_MLP.append(np.random.normal(0,0.01,(size_layer[no_layers-2]+1,output_size)))
layers_MLP=[]
for i in range(0,no_layers):
    layers_MLP.append(np.zeros((size_layer[i])))


# optimization method - gradient descent
train(x_train,y_train,MLP_input_size,no_layers,size_layer,weights_MLP,layers_MLP)


test([0,0])
test([0,1])
test([1,0])
test([1,1])
