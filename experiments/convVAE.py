import sys
import os
from os.path import pardir
print pardir
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.split(os.path.split(current)[0])[0]
core = parent+"/core/"
helpers = parent+"/helpers/"
sys.path.append(core)
sys.path.append(helpers)
from preprocess import pickle_saver,pickle_loader
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from preprocess import map_to_range
from updates import momentum_update,adam,adadelta
from layers import one_d_conv_layer,hidden_layer,variational_gauss_layer,one_d_deconv_layer,one_d_conv_layer_fast,one_d_deconv_layer_fast
from layers import batchnorm
from layers import dropout
from collections import OrderedDict
from theano.tensor.signal import downsample as ds
from functions import relu
from sn_plot import plot_filters
import sn_play as sound_write
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import time

#from sn_play import device_play

# REMBEMBER for convilutional layers. Input tensor = (batch,channels,xpixels,ypixels)
# xpixels is always 1. ypixels is the only one that changes because we have a 1D convolution in this case.

# Weight tensor - (in_channels,out_channels,filter_x,filter_y). filter_x = 1 since we are only 
#filtering in one direction. 

class convVAE(object):
	def __init__(self,dim_z,x_train,x_test,diff=None,magic=5000):
		####################################### SETTINGS ###################################
		self.x_train = x_train;self.x_test = x_test;
		self.diff=diff
		self.batch_size = 100.
		self.learning_rate = theano.shared(np.float32(0.0008))
		self.momentum = 0.3
		self.performance = {"train":[],"test":[]}
		self.inpt = T.ftensor4(name='input')
		self.df = T.fmatrix(name='differential')
		self.dim_z = dim_z
		self.generative_z = theano.shared(np.float32(np.zeros([1,dim_z])))
		self.activation =relu
		self.generative = False
		self.out_distribution=False
		#self.y = T.matrix(name="y")
		self.in_filters = [5,5,5]
		self.filter_lengths = [10.,10.,10.]
		self.params = []
		#magic = 73888.
		self.magic =magic

		self.dropout_symbolic = T.fscalar()
		self.dropout_prob = theano.shared(np.float32(0.0))
		####################################### LAYERS ######################################
		# LAYER 1 ##############################
		self.conv1 = one_d_conv_layer(self.inpt,self.in_filters[0],1,self.filter_lengths[0],param_names = ["W1",'b1']) 
		self.params+=self.conv1.params
		self.bn1 = batchnorm(self.conv1.output)
		self.nl1 = self.activation(self.bn1.X)
		self.maxpool1 = ds.max_pool_2d(self.nl1,[3,1],st=[2,1],ignore_border = False).astype(theano.config.floatX)
		self.layer1_out = dropout(self.maxpool1,self.dropout_symbolic)
		#self.layer1_out = self.maxpool1
		# LAYER2 ################################
		self.flattened = T.flatten(self.layer1_out,outdim = 2)
		# Variational Layer #####################
		self.latent_layer = variational_gauss_layer(self.flattened,self.magic,dim_z)
		self.params+=self.latent_layer.params
		self.latent_out = self.latent_layer.output
		# Hidden Layer #########################
		self.hidden_layer = hidden_layer(self.latent_out,dim_z,self.magic)
		self.params+=self.hidden_layer.params
		self.hid_out = dropout(self.activation(self.hidden_layer.output).reshape((self.inpt.shape[0],self.in_filters[-1],int(self.magic/self.in_filters[-1]),1)),self.dropout_symbolic)
		# Devonvolutional 1 ######################
		self.deconv1 = one_d_deconv_layer(self.hid_out,1,self.in_filters[2],self.filter_lengths[2],pool=2.,param_names = ["W3",'b3'],distribution=False)
		self.params+=self.deconv1.params
		#self.nl_deconv1 = dropout(self.activation(self.deconv1.output),self.dropout_symbolic)
		self.tanh_out = self.deconv1.output
		self.last_layer = self.deconv1

		if self.out_distribution==True:
			self.trunk_sigma =  self.last_layer.log_sigma[:,:,:self.inpt.shape[2],:]
		self.trunc_output = self.tanh_out[:,:,:self.inpt.shape[2],:]
		
		################################### FUNCTIONS ######################################################
		self.get_latent_states = theano.function([self.inpt],self.latent_out,givens=[[self.dropout_symbolic,self.dropout_prob]])
		#self.prior_debug = theano.function([self.inpt],[self.latent_out,self.latent_layer.mu_encoder,self.latent_layer.log_sigma_encoder,self.latent_layer.prior])
		#self.get_prior = theano.function([self.inpt],self.latent_layer.prior)
		#self.convolve1 = theano.function([self.inpt],self.layer1_out)
		#self.convolve2 = theano.function([self.inpt],self.layer2_out)
		self.output = theano.function([self.inpt],self.trunc_output,givens=[[self.dropout_symbolic,self.dropout_prob]])
		self.get_flattened = theano.function([self.inpt],self.flattened,givens=[[self.dropout_symbolic,self.dropout_prob]])
		#self.deconvolve1 = theano.function([self.inpt],self.deconv1.output)
		#self.deconvolve2 = theano.function([self.inpt],self.deconv2.output)
		#self.sig_out = theano.function([self.inpt],T.flatten(self.trunk_sigma,outdim=2))
		self.output = theano.function([self.inpt],self.trunc_output,givens=[[self.dropout_symbolic,self.dropout_prob]])
		#self.generate_from_z = theano.function([self.inpt],self.trunc_output,givens = [[self.latent_out,self.generative_z]])
		self.generate_from_z = theano.function([self.inpt],self.trunc_output,givens = [[self.dropout_symbolic,self.dropout_prob],[self.latent_out,self.generative_z]])
		
		self.cost = self.MSE()
		self.mse = self.MSE()
		#self.likelihood = self.log_px_z()
		#self.get_cost = theano.function([self.inpt],[self.cost,self.mse])

		#self.get_likelihood = theano.function([self.layer1.inpt],[self.likelihood])
		self.derivatives = T.grad(self.cost,self.params)
		#self.get_gradients = theano.function([self.inpt],self.derivatives)
		self.updates =adam(self.params,self.derivatives,self.learning_rate)
		#self.updates =momentum_update(self.params,self.derivatives,self.learning_rate,self.momentum)
		self.train_model = theano.function(inputs = [self.inpt,self.df],outputs = self.cost,updates = self.updates,givens=[[self.dropout_symbolic,self.dropout_prob]])

	def lower_bound(self):
		mu = T.flatten(self.trunc_output,outdim=2)
		inp = T.flatten(self.inpt,outdim=2)
		if self.out_distribution==True:
			sigma = T.mean(T.flatten(self.trunk_sigma,outdim=2))
		else:
			sigma=0	
		#log_gauss =  0.5*np.log(2 * np.pi) + 0.5*sigma + 0.5 * ((inp - mu) / T.exp(sigma))**2.
		log_gauss = T.sum(0.5*np.log(2 * np.pi) + 0.5*sigma + 0.5 * ((inp - mu) / T.exp(sigma))**2.,axis=1)
		return T.mean(log_gauss-self.latent_layer.prior)
	# #def log_px_z(self):
	# 	log_gauss = 0.5*np.log(2 * np.pi) + 0.5*self.trunk_sigma + 0.5 * ((self.inpt - self.trunc_output) / T.exp(self.trunk_sigma))**2
	# 	test = T.flatten(log_gauss,outdim=2)
	# 	test2 = T.sum(test,axis=1)
	# 	return T.mean(tes2)
	def MSE(self):
		#self.cost = T.mean(T.sum((self.y-self.fully_connected.output)**2))
		m = T.sum(T.flatten((self.inpt-self.trunc_output)**2,outdim=2)*self.df,axis=1)
		return T.mean(4*m- self.latent_layer.prior)
	def iterate(self):
		print self.batch_size;
		num_minibatches = int(np.ceil(self.x_train.shape[0]/self.batch_size))
		print num_minibatches
		for i in range(num_minibatches):
			out = []
			idxx = np.random.randint(0,self.x_train.shape[0],self.batch_size)
			if i != num_minibatches:
				minibatch_x = self.x_train[idxx,:,:,:]
				if self.diff==None:
					df_x=np.ones([minibatch_x.shape[0],minibatch_x.shape[2]]).astype(np.float32)
				else:
					df_x = self.diff[idxx,:]
				#minibatch_y = self.y_train[i*batch_length:(i+1) * batch_length,:]
			else:
				minibatch_x = self.x_train[i*self.batch_size:,:,:,:]
				#minibatch_y = self.y_train[i*batch_length:,:]
			out.append(self.train_model(minibatch_x,df_x))
		#out1 = self.get_cost(minibatch_x)
		self.performance["train"].append(np.mean(out))
	def validate(self):
		performance["test"].append(self.get_cost(self.x_test,self.y_test))
		print "Validation cost",performance['test'][-1]
	def generate(self,z = None):
		if z == None:
			z = np.zeros([1,self.dim_z])
		else:
			z = np.array(z)
		self.generative_z.set_value(np.float32(z))
		inp =np.zeros([z.shape[0],self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3]])
		return np.squeeze(self.generate_from_z(np.float32(inp)))



if __name__ == "__main__":
	data_dir = parent+"/data/"
	plot_dir = current+"/plots/"
	sound_dir = current+"/sound/"
	experiment_name = "vae"

	data= pickle_loader(data_dir+"sines.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
	data = np.ndarray.astype(data, np.float32)

	data2 = data[:,1:]
	data1 = data[:,:-1]

	dif = np.hstack([1+np.absolute((data1-data2)/2),np.ones([data.shape[0],1]).astype(np.float32)])

	# map inputs from 0 to 1
	# patch and shape for the CNN
	mean = np.mean(data)
	std = np.std(data)
	sample_rate = 880
	#data = (data -mean)/std
	data = data.reshape(data.shape[0],1,data.shape[1],1)
	dim_z = 5
	#enable interactive plotting

	# train and test
	x_train = data[:20,:,:,:];x_test = data
	dif = dif[:,:]
	# Discover the magic number
	net = convVAE(dim_z,x_train,x_test)
	get_magic = net.get_flattened(net.x_train[:2,:,:,:])
	# actual nework
	net1 = convVAE(dim_z,x_train,x_test,diff = dif,magic = get_magic.shape[1])
	net2 = convVAE(dim_z,x_train,x_test,diff=None,magic = get_magic.shape[1])
	print "magic_value", net.magic
	iterations = 100
	disc = 1.01
	for i in range(iterations):
		net1.iterate()
		net2.iterate()
		rows = 1;columns = 3;	
		print "ITERATION",i
		print net2.performance['train'][-1]

	net1.dropout_prob.set_value(np.float32(0.0))
	net2.dropout_prob.set_value(np.float32(0.0))

	path_array,original_array = sound_write.path_write(net2,880,duration = 0.2,data_points = 10)
	wav.write("sound/"+experiment_name+"_paths.wav",sample_rate,path_array)
	wav.write("sound/"+experiment_name+"_paths_original.wav",sample_rate,original_array)

	random,reconstruction,original = sound_write.sample_write(net1,880,duration = 10)
	wav.write("sound/"+experiment_name+"_random.wav",sample_rate,random)
	wav.write("sound/"+experiment_name+"_reconstruction.wav",sample_rate,reconstruction)
	wav.write("sound/"+experiment_name+"_reconstruction_original.wav",sample_rate,original)

	ou1 = net1.output(x_train[:50])
	ou2 = net2.output(x_train[:50])
	for i in range(ou2.shape[0]):
		if i <100:
			plt.figure()
			plt.plot(ou1[i,0,:,0],color = "g")
			plt.plot(ou2[i,0,:,0],color = "b")
			plt.plot(x_train[i,0,:,0],color = "r")
			plt.savefig(plot_dir+str(i)+"_compare.png",cmap=plt.cm.binary)
	pickle_saver(net2.params,"model_no_boost.pkl")


	f = open("readme","w")
	f.write("SingVAE parameters for experiment: \n------ \n \n")
	f.write("Number of filters:  "+ str([net1.in_filters]) +"\n")
	f.write("Filter lengths:  "+str([net1.filter_lengths])+"\n")
	f.write("Latend variables:  "+str(dim_z)+"\n")
	f.write("Iterations:  "+str(iterations)+"\n")
	f.write("Sample rate:  "+str(sample_rate)+"\n")
	f.close()

	#device_play(net,sample_rate,duration = 1000)
	#use np.squeeze to remove redundant dimentions.
	#out = out.reshape(out.shape[0],out.shape[1],out.shape[3])
	#print out2.shape



