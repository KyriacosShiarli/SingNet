import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from theano.tensor.signal import downsample as ds
from theano.tensor import shared_randomstreams as t_random
theano.sandbox.cuda.dnn import dnn_conv as conv
def relu(x):
    return T.switch(x<0, 0, x)
def det_x(a_11,a_12,a_21,a_22):
	#symbolic determinant of 2 by 2 matrix
	return a_11*a_22 - a_12*a_21

		#self. = theano.function([inpt], output)

UNPOOL =2
POOL = 1

class one_d_conv_layer_cuda(object):
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
        rng.normal(0.,0.001,size=w_shp),
            dtype=inpt.dtype), name =self.param_names[0],borrow = False)
		b_shp = (self.no_of_filters,)
		b = theano.shared(value = np.asarray(
            rng.uniform(low=-.0, high=.0, size=b_shp),
            dtype=inpt.dtype), name =self.param_names[1],borrow =False)
		conv_out = conv(inpt, W.dimshuffle(0,3,2,1),border_mode="valid")
		if activation!=None:
			output = self.activation(conv_out+ b.dimshuffle('x', 0, 'x', 'x'))
		else:
			output = conv_out + b.dimshuffle('x', 0, 'x', 'x')
		self.params = [W,b]
		self.output = ds.max_pool_2d(output,[int(self.pool),1],ignore_border = False).astype(theano.config.floatX)

class one_d_deconv_layer_cuda(object):
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
		conv_out = conv(upsampled, W.dimshuffle(0,3,2,1),border_mode = (self.filter_length/2.,0))
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
	

		
		