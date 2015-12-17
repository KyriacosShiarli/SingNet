import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from theano.tensor.signal import downsample as ds
from theano.tensor import shared_randomstreams as t_random
def relu(x):
    return T.switch(x<0, 0, x)
def det_x(a_11,a_12,a_21,a_22):
	#symbolic determinant of 2 by 2 matrix
	return a_11*a_22 - a_12*a_21

		#self. = theano.function([inpt], output)

UNPOOL =2
POOL = 1

class hidden_layer(object):
	def __init__(self,inpt,in_dim,out_dim,activation = relu,param_names = ["W","b"]):
		self.inpt = inpt
		self.out_dim = out_dim
		self.in_dim = in_dim
		self.activation = activation
		self.param_names = param_names
		self.initialise()
	def initialise(self):
		rng = np.random.RandomState(23455)
		inpt = self.inpt
		w_shp = (self.in_dim,self.out_dim)
		w_bound = np.sqrt(self.in_dim*self.out_dim)
		W = theano.shared( np.asarray(
        rng.uniform(
                low=-0.001,
                high=0.001,
                size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0])

		b_shp = (self.out_dim,)
		b = theano.shared(np.asarray(
            np.zeros(self.out_dim),
            dtype=inpt.dtype), name =self.param_names[1])
		if self.activation!=None:
			self.output = self.activation(T.dot(self.inpt,W) + b)
		else:
			self.output = T.dot(self.inpt,W) + b
		self.params = [W,b]

class variational_gauss_layer(object):
	def __init__(self,inpt,in_dim,out_dim,param_names = ["W","b"]):
		self.inpt = inpt
		self.out_dim = out_dim
		self.in_dim = in_dim
		self.param_names = param_names
		self.eps = t_random.RandomStreams().normal((inpt.shape[0],out_dim,))
		self.initialise()
		
	def initialise(self):
		rng = np.random.RandomState(23455)
		inpt = self.inpt
		w_shp = (self.in_dim,self.out_dim)
		w_bound = np.sqrt(self.out_dim)
		W_mu = theano.shared( np.asarray(
        rng.normal(0.,0.001,size=w_shp),
            dtype=inpt.dtype), name ='w_post_sign')

		b_shp = (self.out_dim,)
		b_mu = theano.shared(np.asarray(
            np.zeros(self.out_dim),
            dtype=inpt.dtype), name ='b_post_mu')
		W_sigma = theano.shared( np.asarray(
        rng.normal(0.,0.001,size=w_shp),
            dtype=inpt.dtype), name ='w_post_sigm')

		b_sigma = theano.shared(np.asarray(
            np.zeros(self.out_dim),
            dtype=inpt.dtype), name ='b_post_sigm')        #Find the hidden variable z
		self.mu_encoder = T.dot(self.inpt,W_mu) +b_mu
		self.log_sigma_encoder =0.5*(T.dot(self.inpt,W_sigma) + b_sigma)
		self.output = self.mu_encoder +T.exp(self.log_sigma_encoder)*self.eps
		self.prior =  0.5* T.sum(1 + 2*self.log_sigma_encoder - self.mu_encoder**2 - T.exp(self.log_sigma_encoder)**2,axis = 1)
		self.params = [W_mu,b_mu,W_sigma,b_sigma]




class one_d_conv_layer(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,activation = relu,param_names = ["W","b"],pool = 1):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.activation = activation
		self.inpt =inpt
		self.param_names = param_names
		self.pool = pool

		print "POOL",pool
		self.initialise()
	def initialise(self):
		activation = self.activation
		rng = np.random.RandomState(23455)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		w_shp = (self.no_of_filters,self.in_channels, 1., self.filter_length)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.001,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = False)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow = True)
		conv_out = conv.conv2d(inpt, W,subsample=(1,1),border_mode = "valid")
		if activation!=None:
			output = self.activation(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
		else:
			output = conv_out + b.dimshuffle('x', 0, 'x', 'x')
		self.params = [W,b]
		self.output = ds.max_pool_2d(output,[1,int(self.pool)],ignore_border = False)

class one_d_deconv_layer(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,activation = T.nnet.softplus,param_names = ["W","b"],pool =1,distribution = False):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.activation = activation
		self.inpt =inpt
		self.param_names = param_names
		self.pool = pool
		self.distribution=distribution
		self.initialise()
	def initialise(self):
		activation = self.activation
		rng = np.random.RandomState(235)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		w_shp = (self.no_of_filters,self.in_channels, 1., self.filter_length)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.001,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = True)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow = True)


		upsampled = self.inpt.repeat(int(self.pool),axis = 3)
		conv_out = conv.conv2d(upsampled, W,subsample=(1,1),border_mode = "full")
		self.params = [W,b]
		if self.distribution==True:
			W_sigma = theano.shared(value = np.asarray(
	        rng.normal(0.,0.001,size=w_shp),
	            dtype=inpt.dtype), name ='lik_sigma',borrow = True)
			b_sigma = theano.shared(value = np.asarray(
	            rng.uniform(low=-.0, high=.0, size=b_shp),
	            dtype=inpt.dtype), name ='b_sigm',borrow = True)
			#self.output =conv_out + b.dimshuffle('x', 0, 'x', 'x')
			conv_out_sigma = conv.conv2d(upsampled, W_sigma,subsample=(1,1),border_mode = "full")
			self.log_sigma = conv_out_sigma + b_sigma.dimshuffle('x', 0, 'x', 'x')
			self.params +=[W_sigma,b_sigma]
		if activation!=None:
			self.output = self.activation(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
		else:
			self.output = conv_out + b.dimshuffle('x', 0, 'x', 'x')
		
		