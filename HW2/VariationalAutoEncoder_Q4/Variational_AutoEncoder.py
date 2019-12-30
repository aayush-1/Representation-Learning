import math
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2


#somewhat good

def sigmoid_deriv(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm * (1. - sigm)


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm



def train(x_train,y_train):
	epochs=4
	learning_rate=0.008
	error_epoch=[]
	accuracy_epoch=[]
	epoch_list=[]
	for epoch in range(epochs):
		print('Training epoch: {}'.format(epoch + 1))
		error=0
		acc=0
		for i in range(x_train.shape[0]):
			encoder(x_train[i,:])
			# decoder vaala daal
			L=np.random.normal(0,1,(size_layer_enc[no_layers_enc-1]))
			z_input=layers_enc[no_layers_enc-1]+layers_std[0]*L
			decoder(z_input)
			dec_loss=0
			# loss = latent loss + generation loss
			latent_loss=0
			for j in range(y_train.shape[1]):
				dec_loss+=(y_train[i,j]-layers_dec[no_layers_dec-1][j])**2
			for j in range(size_layer_enc[no_layers_enc-1]):
				latent_loss+=0.5*(layers_enc[no_layers_enc-1][j]**2 + layers_std[0][j]**2 - 2*np.log(layers_std[0][j]) - 1)
			error+= dec_loss + latent_loss
			print(i)
			print("dec_loss: ",dec_loss)
			print("latent_loss: ",latent_loss)
			d_prev=decoder_backprop(y_train[i,:],learning_rate)
			dx=np.zeros((size_layer_dec[0]))
			for m in range(0,size_layer_dec[0]):
				for k in range(0,size_layer_dec[1]):
					dx[m]+=d_prev[k]*weights_dec[0][m+1,k]
			d_prev_mean=dx
			d_prev_std=dx*L
			encoder_backprop(d_prev_mean,d_prev_std,learning_rate)
		print(weights_enc)
		print(weights_dec)
		print(weights_std)
		error_epoch.append(error/x_train.shape[0])
		#accuracy_epoch.append(acc/x_train.shape[0])
		print("error per epoch = ",error/x_train.shape[0])
		#print("accuracy per epoch = ",acc/x_train.shape[0])
		epoch_list.append(epoch+1)




def encoder(enc_input):
	layers_enc[0]=enc_input
	for i in range(0,no_layers_enc-1):
		layers_enc[i+1]=np.matmul(np.transpose(layers_enc[i]),weights_enc[i][1:,:])+weights_enc[i][0,:]
		layers_enc[i+1]=sigmoid(layers_enc[i+1])
		if(i==no_layers_enc-2):
			layers_std[0]=np.matmul(np.transpose(layers_enc[i]),weights_std[0][1:,:])+weights_std[0][0,:]
			layers_std[0]=sigmoid(layers_std[0])


def decoder(dec_input):
	layers_dec[0]=dec_input
	for i in range(0,no_layers_dec-1):
		layers_dec[i+1]=np.matmul(np.transpose(layers_dec[i]),weights_dec[i][1:,:])+weights_dec[i][0,:]
		layers_dec[i+1]=sigmoid(layers_dec[i+1])




def decoder_backprop(y_train,learning_rate):
	x=0
	for i in range(no_layers_dec-2,-1,-1):
		a=weights_dec[i].shape[0]
		b=weights_dec[i].shape[1]
		if x==0:
			dx=np.zeros((size_layer_dec[i+1]))
			x=1
		else:
			dy=dx+0.0
			dx=np.zeros((size_layer_dec[i+1]))

		if i==no_layers_dec-2:
			dx=(-2)*(y_train-(layers_dec[i+1]))
		else:
			for m in range(0,size_layer_dec[i+1]):
				dx[m]=np.dot(dy*weights_dec[i+1][m+1,:])
		for m in range(size_layer_dec[i]):
			weights_dec[i][1+m,:]=weights_dec[i][1+m,:]-learning_rate*dx*layers_dec[i][m]*sigmoid_deriv(layers_dec[i+1])
		weights_dec[i][0,:]=weights_dec[i][0,:]-learning_rate*dx*sigmoid_deriv(layers_dec[i+1])
	return dx



def encoder_backprop(d_prev_mean,d_prev_std,learning_rate):
	for i in range(no_layers_enc-2,-1,-1):
		a=weights_enc[i].shape[0]
		b=weights_enc[i].shape[1]
		if i==no_layers_enc-2:
			dx_mean=np.zeros((size_layer_enc[i+1]))
			dx_std=np.zeros((size_layer_enc[i+1]))
		elif i==no_layers_enc-3:
			dy_mean=dx_mean+0.0
			dy_std=dx_std+0.0
			dx=np.zeros((size_layer_enc[i+1]))
		else:
			dy=dx_mean+0.0
			dx=np.zeros((size_layer_enc[i+1]))

		if i==no_layers_dec-2:
			dx_mean=d_prev_mean+layers_enc[no_layers_enc-1]
			dx_std=d_prev_std + layers_std[0] - 1/(layers_std[0])
		elif i==no_layers_dec-3:
			for m in range(0,size_layer_enc[i+1]):
				dx[m]=np.dot(dy_mean*weights_enc[i+1][m+1,:])+np.dot(dy_std*weights_std[0][m+1,:])
		else:
			for m in range(0,size_layer_enc[i+1]):
				dx[m]=np.dot(dy*weights_enc[i+1][m+1,:])
		if i==no_layers_enc-2: 
			for m in range(size_layer_enc[i]):
				weights_enc[i][1+m,:]=weights_enc[i][1+m,:]-learning_rate*dx_mean*layers_enc[i][m]*sigmoid_deriv(layers_enc[i+1])
				weights_std[0][1+m,:]=weights_std[0][1+m,:]-learning_rate*dx_std*layers_enc[i][m]*sigmoid_deriv(layers_std[0])
			weights_enc[i][0,:]=weights_enc[i][0,:]-learning_rate*dx_mean*sigmoid_deriv(layers_enc[i+1])
			weights_std[0][0,:]=weights_std[0][0,:]-learning_rate*dx_std*sigmoid_deriv(layers_std[0])

		else:
			for m in range(size_layer_enc[i]):
				weights_enc[i][1+m,:]=weights_enc[i][1+m,:]-learning_rate*dx*layers_enc[i][m]*sigmoid_deriv(layers_enc[i+1])
			weights_enc[i][0,:]=weights_enc[i][0,:]-learning_rate*dx*sigmoid_deriv(layers_enc[i+1])




def test(x_test):
	output=np.zeros((x_test.shape))
	for i in range(8):
		input1 = np.reshape(x_test[i], [14*14])
		encoder(input1)
		output1=layers_enc[no_layers_enc-1]+layers_std[0]*np.random.normal(0,1,(size_layer_enc[no_layers_enc-1]))
		decoder(output1)
		output1=layers_dec[no_layers_dec-1]
		output[i]=output1
	imgs = np.concatenate([x_test, output])
	imgs = imgs.reshape((4, 4,14,14))
	imgs = np.vstack([np.hstack(i) for i in imgs])
	plt.figure()
	plt.axis('off')
	plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
	plt.imshow(imgs, interpolation='none', cmap='gray')
	plt.savefig('Variational_AutoEncoder_output.png')
	plt.show()

def test_dec():
	output=np.zeros((16,196))
	for i in range(16):
		output1=np.random.normal(0,1,(size_layer_enc[no_layers_enc-1]))
		decoder(output1)
		output1=layers_dec[no_layers_dec-1]
		output[i]=output1
	imgs = output.reshape((4, 4,14,14))
	imgs = np.vstack([np.hstack(i) for i in imgs])
	plt.figure()
	plt.axis('off')
	plt.imshow(imgs, interpolation='none', cmap='gray')
	plt.savefig('Variational_AutoEncoder_output_dec.png')
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


# Encoder initialization:

enc_input_size=x_train.shape[1]  #depends on data
no_layers_enc=2
size_layer_enc=[]
size_layer_enc.append(enc_input_size)
size_layer_enc.append(150)

weights_enc=[]
weights_std=[]
for i in range(0,no_layers_enc-1):
	weights_enc.append(np.random.normal(0,1,(size_layer_enc[i]+1,size_layer_enc[i+1])))
	if(i==no_layers_enc-2):
		weights_std.append(np.random.normal(0,1,(size_layer_enc[i]+1,size_layer_enc[i+1])))
layers_enc=[]
layers_std=[]
for i in range(0,no_layers_enc):
	layers_enc.append(np.zeros((size_layer_enc[i])))
	if(i==no_layers_enc-1):
		layers_std.append(np.zeros((size_layer_enc[i])))


# Decoder initialization:

dec_input_size=150  #latent space
no_layers_dec=2
size_layer_dec=[]
size_layer_dec.append(dec_input_size)
size_layer_dec.append(196)

weights_dec=[]
for i in range(0,no_layers_dec-1):
	weights_dec.append(np.random.normal(0,1,(size_layer_dec[i]+1,size_layer_dec[i+1])))
layers_dec=[]
for i in range(0,no_layers_dec):
	layers_dec.append(np.zeros((size_layer_dec[i])))


train(x_train,x_train)
#np.save('weights_enc',weights_enc)
#np.save('weights_std',weights_std)
#np.save('weights_dec',weights_dec)
test(x_test[:8])
test_dec()












