from preprocess import pickle_saver,pickle_loader
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from preprocess import map_to_range
from updates import momentum_update,adam
from layers import one_d_conv_layer,hidden_layer,variational_gauss_layer,one_d_deconv_layer
from collections import OrderedDict
#from sn_play import device_play

# REMBEMBER for convilutional layers. Input tensor = (batch,channels,xpixels,ypixels)
# xpixels is always 1. ypixels is the only one that changes because we have a 1D convolution in this case.

# Weight tensor - (in_channels,out_channels,filter_x,filter_y). filter_x = 1 since we are only 
#filtering in one direction. 

class convVAE(object):
	def __init__(self,dim_z,x_train,x_test):
		####################################### SETTINGS ###################################
		self.x_train = x_train;self.x_test = x_test;
		self.batch_proportion = 0.5
		self.learning_rate = theano.shared(0.00005)
		self.momentum = 0.9
		self.performance = {"train":[],"test":[]}
		self.inpt = T.tensor4(name='input')
		self.dim_z = dim_z
		self.generative_z = theano.shared(np.zeros([1,dim_z]))
		
		self.generative = False
		#self.y = T.matrix(name="y")
		self.in_filters = [32,32,32]
		self.filter_lengths = [40.,40.,40]
		self.params = []

		####################################### LAYERS ######################################
		self.layer1 = one_d_conv_layer(self.inpt,self.in_filters[0],1,self.filter_lengths[0],param_names = ["W1",'b1'],pool=2)
		self.params+=self.layer1.params
		self.test = T.flatten(self.layer1.output,outdim = 2)
		self.latent_layer = variational_gauss_layer(T.flatten(self.layer1.output,outdim = 2),63392.,dim_z)
		self.params+=self.latent_layer.params
		self.latent_out = self.latent_layer.output
		self.hidden_layer = hidden_layer(self.latent_layer.output,dim_z,63392.)
		self.params+=self.hidden_layer.params
		self.hid_out = self.hidden_layer.output.reshape((self.inpt.shape[0],self.in_filters[-1],1,int(63392./self.in_filters[-1])))
		self.deconv1 = one_d_deconv_layer(self.hid_out,1,self.in_filters[2],self.filter_lengths[2],pool=2,param_names = ["W4",'b4'],activation=None,distribution=True)
		self.params+=self.deconv1.params
		self.last_layer = self.deconv1
		self.trunc_output = self.last_layer.output[:,:,:,:self.inpt.shape[3]]
		self.trunk_sigma =  self.last_layer.log_sigma[:,:,:,:self.inpt.shape[3]]
		################################### FUNCTIONS ######################################################
		self.get_latent_states = theano.function([self.inpt],self.latent_out)
		self.convolve1 = theano.function([self.layer1.inpt],self.layer1.output)
		self.convolve3 = theano.function([self.layer1.inpt],self.test)
		self.deconvolve3 = theano.function([self.layer1.inpt],self.latent_layer.prior)
		self.output = theano.function([self.layer1.inpt],self.last_layer.output)
		self.generate_from_z = theano.function([self.inpt],self.trunc_output,givens = [[self.latent_out,self.generative_z]])
		self.cost = self.lower_bound()
		self.mse = self.MSE()
		print "gothere"
		self.get_cost = theano.function([self.layer1.inpt],[self.cost,self.mse])
		self.derivatives = T.grad(self.cost,self.params)
		self.get_gradients = theano.function([self.layer1.inpt],self.derivatives)
		self.updates = adam(self.params,self.derivatives,self.learning_rate)
		self.train_model = theano.function(inputs = [self.inpt],outputs = self.cost,updates = self.updates)

	def lower_bound(self):
		log_gauss =  0.5*np.log(2 * np.pi) + 0.5*self.trunk_sigma + 0.5 * ((self.inpt - self.trunc_output) / T.exp(self.trunk_sigma))**2
		test = T.flatten(log_gauss,outdim=2)
		test2 = T.sum(test,axis=1)
		return T.mean(test2- self.latent_layer.prior)
	def log_px_z(self):
		log_gauss = 0.5*np.log(2 * np.pi) + 0.5*self.trunk_sigma + 0.5 * ((self.inpt - self.trunc_output) / T.exp(self.trunk_sigma))**2
		test = T.flatten(log_gauss,outdim=2)
		test2 = T.sum(test,axis=1)
		return T.mean(test2)
	def MSE(self):
		#self.cost = T.mean(T.sum((self.y-self.fully_connected.output)**2))
		m = T.flatten((self.inpt-self.trunc_output)**2,outdim=2)
		return T.mean(m)
	def iterate(self):
		batch_length = np.floor(self.x_train.shape[0]*self.batch_proportion)
		num_minibatches = int(np.ceil(1/self.batch_proportion))
		for i in range(num_minibatches):
			out = []
			if i != num_minibatches-1:
				minibatch_x = self.x_train[i*batch_length:(i+1) * batch_length,:,:,:]
				#minibatch_y = self.y_train[i*batch_length:(i+1) * batch_length,:]
			else:
				minibatch_x = self.x_train[i*batch_length:,:,:,:]
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
	data_dictionary = pickle_loader("sound/sine_mixture.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
	data = data_dictionary["data"]
	sample_rate = data_dictionary["sample rate"]
	print "SAMLE",sample_rate
	# map inputs from 0 to 1
	# patch and shape for the CNN
	mean = np.mean(data)
	std = np.std(data)
	#data = (data -mean)/std
	data = data.reshape(data.shape[0],1,1,data.shape[1])
	dim_z = 10
	# train and test
	idx = np.random.permutation(data.shape[0])
	x_train = data[:20,:];x_test = data
	x_train = x_train*2.
	net = convVAE(dim_z,x_train,x_test)
	out1 = net.deconvolve3(x_train)
	pdb.set_trace()
	out2 = net.generate()


	iterations = 1000

	disc = 1.08
	for i in range(iterations):
		net.iterate()
		print "ITERATION",i
		print net.generate().shape
		print net.learning_rate.get_value()
		if i%50==0:
			net.learning_rate.set_value(net.learning_rate.get_value()/disc)
		print net.performance['train'][-1]
	pdb.set_trace()
	device_play(net,sample_rate,duration = 1000)
	#use np.squeeze to remove redundant dimentions.
	#out = out.reshape(out.shape[0],out.shape[1],out.shape[3])
	#print out2.shape



