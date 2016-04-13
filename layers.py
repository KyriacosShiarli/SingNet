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
		self.param_names = param_names
		self.initialise()
	def initialise(self):
		rng = np.random.RandomState(23455)
		inpt = self.inpt
		w_shp = (self.in_dim,self.out_dim)
		w_bound = np.sqrt(self.in_dim*self.out_dim)
		W = theano.shared( np.asarray(
        rng.uniform(
                low=-0.01,
                high=0.01,
                size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0])

		b_shp = (self.out_dim,)
		b = theano.shared(np.asarray(
            np.zeros(self.out_dim),
            dtype=inpt.dtype), name =self.param_names[1])
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
        rng.normal(0.,0.01,size=w_shp),
            dtype=inpt.dtype), name ='w_post_mu')

		b_shp = (self.out_dim,)
		b_mu = theano.shared(np.asarray(
            np.zeros(self.out_dim),
            dtype=inpt.dtype), name ='b_post_mu')
		W_sigma = theano.shared( np.asarray(
        rng.normal(0.,0.01,size=w_shp),
            dtype=inpt.dtype), name ='w_post_sigm')

		b_sigma = theano.shared(np.asarray(
            np.zeros(self.out_dim),
            dtype=inpt.dtype), name ='b_post_sigm')        #Find the hidden variable z
		self.mu_encoder = T.dot(self.inpt,W_mu) +b_mu
		self.log_sigma_encoder =0.5*(T.dot(self.inpt,W_sigma) + b_sigma)
		self.output =self.mu_encoder +T.exp(self.log_sigma_encoder)*self.eps.astype(theano.config.floatX)
		self.prior =  0.5* T.sum(1 + 2*self.log_sigma_encoder - self.mu_encoder**2 - T.exp(2*self.log_sigma_encoder),axis=1).astype(theano.config.floatX)
		self.params = [W_mu,b_mu,W_sigma,b_sigma]




class one_d_conv_layer(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,param_names = ["W","b"],pool = 1,border_mode=[0,0]):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.inpt =inpt
                self.border_mode = border_mode
		self.param_names = param_names
		print "POOL",pool
		self.initialise()
	def initialise(self):
		rng = np.random.RandomState(23455)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		w_shp = (self.no_of_filters,self.in_channels, self.filter_length,1.)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.01,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = False)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow = True)
		self.output = conv.conv2d(inpt, W,subsample=(1,1),border_mode = "full") + b.dimshuffle('x', 0, 'x', 'x')
		self.params = [W,b]

class one_d_conv_layer_fast(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,activation = relu,param_names = ["W","b"],pool = 1):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.activation = activation
		self.inpt =inpt.dimshuffle(0,3,2,1)
		self.param_names = param_names
		self.pool = pool
		print "POOL",pool
		self.initialise()
	def initialise(self):
		activation = self.activation
		rng = np.random.RandomState(23455)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		w_shp = (self.no_of_filters,self.in_channels, self.filter_length,1.)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.01,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = False)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow =False)
		conv_out = conv.conv2d(inpt, W.dimshuffle(0,3,2,1),subsample=(1,1),border_mode = "valid")
		if activation!=None:
			output = self.activation(conv_out+ b.dimshuffle('x', 0, 'x', 'x'))
		else:
			output = conv_out + b.dimshuffle('x', 0, 'x', 'x')
		self.params = [W,b]
		self.output = ds.max_pool_2d(output,[int(self.pool),1],ignore_border = False).astype(theano.config.floatX)


class one_d_deconv_layer_zero(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,param_names = ["W","b"],pool =1,distribution = False):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.inpt =inpt
		self.param_names = param_names
		self.pool = pool
		self.distribution=distribution
		self.initialise()
	def initialise(self):
		rng = np.random.RandomState(235)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		w_shp = (self.no_of_filters,self.in_channels,self.filter_length,1.)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.01,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = True)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow = True)
		zeros = T.zeros_like(????????) # It should do: create array of size (batch, channels in, length_signal, upsample_factor-1) with zeros in it 
		upsampled = T.flatten(T.concatenate((a,b),axis=3),outdim=3)[:,:,:,None]
		conv_out = conv.conv2d(upsampled, W,subsample=(1,1),border_mode = "full")
		self.params = [W,b]
		if self.distribution==True:
			W_sigma = theano.shared(value = np.asarray(
	        rng.normal(0.,0.01,size=w_shp),
	            dtype=inpt.dtype), name ='lik_sigma',borrow = True)
			b_sigma = theano.shared(value = np.asarray(
	            rng.uniform(low=-.0, high=.0, size=b_shp),
	            dtype=inpt.dtype), name ='b_sigm',borrow = True)
			#self.output =conv_out + b.dimshuffle('x', 0, 'x', 'x')
			conv_out_sigma = conv.conv2d(upsampled, W_sigma,subsample=(1,1),border_mode = "full")
			self.log_sigma = conv_out_sigma + b_sigma.dimshuffle('x', 0, 'x', 'x')
			self.params +=[W_sigma,b_sigma]
		self.output = conv_out + b.dimshuffle('x', 0, 'x', 'x').astype(theano.config.floatX)


class one_d_deconv_layer(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,param_names = ["W","b"],pool =1,distribution = False):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.inpt =inpt
		self.param_names = param_names
		self.pool = pool
		self.distribution=distribution
		self.initialise()
	def initialise(self):
		rng = np.random.RandomState(235)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		w_shp = (self.no_of_filters,self.in_channels,self.filter_length,1.)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.01,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = True)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow = True)
		upsampled = self.inpt.repeat(int(self.pool),axis = 2)
		conv_out = conv.conv2d(upsampled, W,subsample=(1,1),border_mode = "full")
		self.params = [W,b]
		if self.distribution==True:
			W_sigma = theano.shared(value = np.asarray(
	        rng.normal(0.,0.01,size=w_shp),
	            dtype=inpt.dtype), name ='lik_sigma',borrow = True)
			b_sigma = theano.shared(value = np.asarray(
	            rng.uniform(low=-.0, high=.0, size=b_shp),
	            dtype=inpt.dtype), name ='b_sigm',borrow = True)
			#self.output =conv_out + b.dimshuffle('x', 0, 'x', 'x')
			conv_out_sigma = conv.conv2d(upsampled, W_sigma,subsample=(1,1),border_mode = "full")
			self.log_sigma = conv_out_sigma + b_sigma.dimshuffle('x', 0, 'x', 'x')
			self.params +=[W_sigma,b_sigma]
		self.output = conv_out + b.dimshuffle('x', 0, 'x', 'x').astype(theano.config.floatX)


class one_d_deconv_layer_fast(object):
	def __init__(self,inpt,no_filters,in_channels,filter_length,activation = T.nnet.softplus,param_names = ["W","b"],pool =1,distribution = False):
		self.no_of_filters = no_filters
		self.in_channels = in_channels
		self.filter_length = filter_length
		self.activation = activation
		self.inpt =inpt.dimshuffle(0,3,2,1)
		self.param_names = param_names
		self.pool = pool
		self.distribution=distribution
		self.initialise()
	def initialise(self):
		activation = self.activation
		rng = np.random.RandomState(235)
		inpt = self.inpt
		# initialise layer 1 weight vector. 
		#w_shp = (self.no_of_filters, 1.,self.in_channels, self.filter_length)
		w_shp = (self.no_of_filters, self.in_channels, self.filter_length,1.)
		w_bound = np.sqrt(self.in_channels* self.filter_length)
		W = theano.shared(value = np.asarray(
        rng.normal(0.,0.001,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = True)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow = True)
		upsampled = self.inpt.repeat(int(self.pool),axis = 2)
		conv_out = conv.conv2d(upsampled, W.dimshuffle(0,3,2,1),subsample=(1,1),border_mode = "full")
		conv_out = conv_out[:,:,:,int(self.in_channels-1):-int(self.in_channels-1)]
		self.params = [W,b]
		if self.distribution==True:
			W_sigma = theano.shared(value = np.asarray(
	        rng.normal(0.,0.001,size=w_shp),
	            dtype=inpt.dtype), name ='lik_sigma',borrow = True)
			b_sigma = theano.shared(value = np.asarray(
	            rng.uniform(low=-.0, high=.0, size=b_shp),
	            dtype=inpt.dtype), name ='b_sigm',borrow = True)
			#self.output =conv_out + b.dimshuffle('x', 0, 'x', 'x')
			conv_out_sigma = conv.conv2d(upsampled, W_sigma.dimshuffle(0,3,2,1),subsample=(1,1),border_mode = "full",)
			conv_out_sigma = conv_out_sigma[:,:,:,int(self.in_channels-1):-int(self.in_channels-1)]
			self.log_sigma = conv_out_sigma + b_sigma.dimshuffle('x', 0, 'x', 'x')
			self.params +=[W_sigma,b_sigma]
		if activation!=None:
			self.output = self.activation(conv_out + b.dimshuffle('x', 0, 'x', 'x')).astype(theano.config.floatX)
		else:
			self.output = conv_out + b.dimshuffle('x', 0, 'x', 'x').astype(theano.config.floatX)

class batchnorm(object):
	def __init__(self,X,g=None,b=None,u=None,s=None,a=1.,e=1e-8):
		self.g = g
		self.b = b
		self.u = u
		self.s = s
		self.a = a
		self.e = e
		self.X = X
		self.params=[]
		self.initialise()
	def initialise(self):
		if self.X.ndim == 4:
			if self.u is not None and self.s is not None:
				b_u = self.u.dimshuffle('x',0,'x','x')
				b_s = self.s.dimshuffle('x',0,'x','x')
			else:
				b_u = T.mean(self.X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
				b_s = T.mean(T.sqr(self.X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
			if self.a != 1:
				b_u = (1. - self.a)*0. + self.a*b_u
				b_s = (1. - self.a)*1. + self.a*b_s
			output = (self.X - b_u) / T.sqrt(b_s + self.e)
			if self.g is not None and self.b is not None:
				self.X = self.X*self.g.dimshuffle('x', 0, 'x', 'x') + self.b.dimshuffle('x', 0, 'x', 'x')
				self.params.append(g);self.params.append(b)
		elif self.X.ndim == 2:
			if self.u is None and self.s is None:
				self.u = T.mean(self.X, axis=0)
				self.s = T.mean(T.sqr(self.X - self.u), axis=0)
			if self.a != 1:
				self.u = (1. - self.a)*0. + self.a*self.u
				self.s = (1. - self.a)*1. + self.a*self.s
		 	self.X = (self.X - self.u) / T.sqrt(self.s + self.e)
			if self.g is not None and self.b is not None:
				self.X = self.X*self.g + self.b
				self.params.append(g);self.params.append(b)
		else:
			raise NotImplementedError
		#return self.X	

def dropout(X, p):
	srng = t_random.RandomStreams()
	retain_prob = 1 - p
	X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
	X /= retain_prob
	return X.astype(theano.config.floatX)
		
		
