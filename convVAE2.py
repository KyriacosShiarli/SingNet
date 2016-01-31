from preprocess import pickle_saver,pickle_loader
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from preprocess import map_to_range
from updates import momentum_update,adam,adadelta
from layers import one_d_conv_layer,hidden_layer,variational_gauss_layer,one_d_deconv_layer,one_d_conv_layer_fast,one_d_deconv_layer_fast
from collections import OrderedDict
from functions import relu
from sn_plot import plot_filters
from matplotlib import pyplot as plt
import time

#from sn_play import device_play

# REMBEMBER for convilutional layers. Input tensor = (batch,channels,xpixels,ypixels)
# xpixels is always 1. ypixels is the only one that changes because we have a 1D convolution in this case.

# Weight tensor - (in_channels,out_channels,filter_x,filter_y). filter_x = 1 since we are only 
#filtering in one direction. 

class convVAE(object):
	def __init__(self,dim_z,x_train,x_test):
		####################################### SETTINGS ###################################
		self.x_train = x_train;self.x_test = x_test;
		self.batch_size = 2.
		self.learning_rate = theano.shared(0.00005).astype(theano.config.floatX)
		self.momentum = 0.3
		self.performance = {"train":[],"test":[]}
		self.inpt = T.ftensor4(name='input')
		self.inpt.tag.test_value = x_train
		self.dim_z = dim_z
		self.generative_z = theano.shared(np.zeros([1,dim_z])).astype(theano.config.floatX)
		
		self.generative = False
		#self.y = T.matrix(name="y")
		self.in_filters = [20,20,20]
		self.filter_lengths = [256.,256.,256.]
		self.params = []
		#magic = 73888.
		magic = 51700.
		####################################### LAYERS ######################################

		self.layer1 = one_d_conv_layer(self.inpt,self.in_filters[0],1,self.filter_lengths[0],activation = T.nnet.softplus,param_names = ["W1",'b1'],pool=5.) 
		self.params+=self.layer1.params
		self.layer2 = one_d_conv_layer(self.layer1.output,self.in_filters[0],self.in_filters[0],self.filter_lengths[0],activation = T.nnet.softplus,param_names = ["W1",'b1'],pool=5.) 
		self.params+=self.layer2.params
		self.test = T.flatten(self.layer2.output,outdim = 2)
		self.latent_layer = variational_gauss_layer(self.test,magic,dim_z)
		self.params+=self.latent_layer.params
		self.latent_out = self.latent_layer.output
		self.hidden_layer = hidden_layer(self.latent_out,dim_z,magic)
		self.params+=self.hidden_layer.params
		self.hid_out = self.hidden_layer.output.reshape((self.inpt.shape[0],self.in_filters[-1],1,int(magic/self.in_filters[-1])))
		self.deconv1 = one_d_deconv_layer(self.hid_out,self.in_filters[2],self.in_filters[2],self.filter_lengths[2],pool=5.,param_names = ["W3",'b3'],activation=T.nnet.softplus,distribution=False)
		self.params+=self.deconv1.params
		self.deconv2 = one_d_deconv_layer(self.deconv1.output,1,self.in_filters[2],self.filter_lengths[2],pool=5.,param_names = ["W4",'b4'],activation=None,distribution=True)
		self.params+=self.deconv2.params
		self.last_layer = self.deconv2
		self.trunc_output = self.last_layer.output[:,:,:,:self.inpt.shape[3]]
		self.trunk_sigma =  self.last_layer.log_sigma[:,:,:,:self.inpt.shape[3]]
		################################### FUNCTIONS ######################################################
		self.get_latent_states = theano.function([self.inpt],self.latent_out)
		self.get_prior = theano.function([self.inpt],self.latent_layer.prior)
		# self.convolve1 = theano.function([self.inpt],self.layer1.output)
		# self.convolve2 = theano.function([self.inpt],self.layer2.output)
		# self.convolve3 = theano.function([self.inpt],self.test)
		# self.deconvolve1 = theano.function([self.inpt],self.deconv1.output)
		self.deconvolve2 = theano.function([self.inpt],self.deconv2.output)
		#self.sig_out = theano.function([self.inpt],T.flatten(self.trunk_sigma,outdim=2))
		self.output = theano.function([self.inpt],self.last_layer.output)
		self.generate_from_z = theano.function([self.inpt],self.trunc_output,givens = [[self.latent_out,self.generative_z]])
		self.cost = self.lower_bound()
		self.mse = self.MSE()
		#self.likelihood = self.log_px_z()
		self.get_cost = theano.function([self.inpt],[self.cost,self.mse])

		#self.get_likelihood = theano.function([self.layer1.inpt],[self.likelihood])
		self.derivatives = T.grad(self.cost,self.params)
		self.get_gradients = theano.function([self.inpt],self.derivatives)
		self.updates =adam(self.params,self.derivatives,self.learning_rate)
		self.train_model = theano.function(inputs = [self.inpt],outputs = self.cost,updates = self.updates)

	def lower_bound(self):
		sigma = T.mean(T.flatten(self.trunk_sigma,outdim=2))
		#sigma = 0.
		mu = T.flatten(self.trunc_output,outdim=2)
		inp = T.flatten(self.inpt,outdim=2)
		log_gauss =  0.5*np.log(2 * np.pi) + 0.5*sigma + 0.5 * ((inp - mu) / T.exp(sigma))**2.
		test2 = T.sum(log_gauss,axis=1)
		return T.mean(test2- self.latent_layer.prior)
	# #def log_px_z(self):
	# 	log_gauss = 0.5*np.log(2 * np.pi) + 0.5*self.trunk_sigma + 0.5 * ((self.inpt - self.trunc_output) / T.exp(self.trunk_sigma))**2
	# 	test = T.flatten(log_gauss,outdim=2)
	# 	test2 = T.sum(test,axis=1)
	# 	return T.mean(tes2)
	def MSE(self):
		#self.cost = T.mean(T.sum((self.y-self.fully_connected.output)**2))
		m = T.sum(T.flatten((self.inpt-self.trunc_output)**2,outdim=2),axis=1)
		return T.mean(m- self.latent_layer.prior)
	def iterate(self):
		print self.batch_size;
		num_minibatches = int(np.ceil(self.x_train.shape[0]/self.batch_size))
		print num_minibatches
		for i in range(num_minibatches):
			out = []
			if i != num_minibatches-1:
				minibatch_x = self.x_train[i*self.batch_size:(i+1) * self.batch_size,:,:,:]
				#minibatch_y = self.y_train[i*batch_length:(i+1) * batch_length,:]
			else:
				minibatch_x = self.x_train[i*self.batch_size:,:,:,:]
				#minibatch_y = self.y_train[i*batch_length:,:]
			out.append(self.train_model(minibatch_x))
		out1 = self.get_cost(minibatch_x)
		print out1
		self.performance["train"].append(np.mean(out))
	def validate(self):
		performance["test"].append(self.get_cost(self.x_test,self.y_test))
		print "Validation cost",performance['test'][-1]
	def generate(self,z = None):
		if z == None:
			z = np.zeros([1,self.dim_z])
		else:
			z = np.array([z])
		self.generative_z.set_value(z)
		inp = np.zeros([1,self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3]])
		return np.squeeze(self.generate_from_z(inp))



if __name__ == "__main__":
	data= pickle_loader("sound/instruments.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
	# map inputs from 0 to 1
	# patch and shape for the CNN
	mean = np.mean(data)
	std = np.std(data)
	#data = (data -mean)/std
	data = data.reshape(data.shape[0],1,1,data.shape[1])
	data = data+1
	dim_z = 200
	#enable interactive plotting

	# train and test
	#idx = np.random.permutation(data.shape[0])
	x_train = data[:10,:,:,:];x_test = data
	net = convVAE(dim_z,x_train,x_test)
	out = net.convolve3(net.x_train)
	iterations = 400
	disc = 1.01
	f1 = plt.figure(1)
	print "F1",f1.number
	print "params",len(net.params)
	prev = net.params[6].get_value()
	no_of_filters = 3
	for i in range(iterations):
		net.iterate()
		print "Prior",net.get_prior(net.x_train)
		#plot_filters(net.params[12],f1,interactive=True)
		diff = prev - net.params[6].get_value()
		#print diff
		print prev
		
		rows = 1;columns = 3;

		
		print "ITERATION",i
		if i==10:	
			net.learning_rate.set_value(net.learning_rate.get_value()/1.2)
		if i%3==0:	
			plt.clf()
			net.learning_rate.set_value(net.learning_rate.get_value()/disc)
			# for i in range(no_of_filters):
			# 	f1.add_subplot(rows,columns,i+1)
			# 	plt.plot(prev)
			# 	plt.pause(0.01)	
			# 	plt.draw()

			disc = disc
		print net.performance['train'][-1]
		prev = net.params[6].get_value()
	pdb.set_trace()
	#device_play(net,sample_rate,duration = 1000)
	#use np.squeeze to remove redundant dimentions.
	#out = out.reshape(out.shape[0],out.shape[1],out.shape[3])
	#print out2.shape



