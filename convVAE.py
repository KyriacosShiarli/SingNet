from preprocess import pickle_saver,pickle_loader
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from preprocess import map_to_range
from layers import one_d_conv_layer,hidden_layer,variational_gauss_layer
from collections import OrderedDict

# REMBEMBER for convilutional layers. Input tensor = (batch,channels,xpixels,ypixels)
# xpixels is always 1. ypixels is the only one that changes because we have a 1D convolution in this case.

# Weight tensor - (in_channels,out_channels,filter_x,filter_y). filter_x = 1 since we are only 
#filtering in one direction. 

class convVAE(object):
	def __init__(self,dim_z,x_train,x_test):
		self.x_train = x_train;self.x_test = x_test;
		self.batch_proportion = 0.1
		self.learning_rate = theano.shared(0.00005)
		self.momentum = 0.9
		self.performance = {"train":[],"test":[]}
		self.inpt = T.tensor4(name='input')
		self.dim_z = dim_z
		#self.y = T.matrix(name="y")
		self.in_filters = [32,32,32]
		self.strides = [1.,1.,1.]
		self.filter_lengths = [40.,40.,40]
		self.params = []
		self.layer1 = one_d_conv_layer(self.inpt,self.in_filters[0],1,self.strides[0],self.filter_lengths[0],param_names = ["W1",'b1'],pool=None)
		self.params+=self.layer1.params
		self.layer2 = one_d_conv_layer(self.layer1.output,self.in_filters[1],self.in_filters[0],self.strides[1],self.filter_lengths[1],pool=None,param_names = ["W2",'b2'])
		self.params+=self.layer2.params
		self.layer3 = one_d_conv_layer(self.layer2.output,self.in_filters[2],self.in_filters[1],self.strides[2],self.filter_lengths[2],pool=None,param_names = ["W3",'b3'])
		self.params+=self.layer3.params
		self.test = T.flatten(self.layer3.output,outdim = 2)

		self.latent_layer = variational_gauss_layer(T.flatten(self.layer3.output,outdim = 2),124288,dim_z)
		self.params+=self.latent_layer.params
		self.hidden_layer = hidden_layer(self.latent_layer.output,dim_z,124288)
		self.params+=self.hidden_layer.params
		self.hid_out = self.hidden_layer.output.reshape((self.inpt.shape[0],self.in_filters[-1],1,int(124288/self.in_filters[-1])))
		self.deconv1 = one_d_conv_layer(self.hid_out,self.in_filters[1],self.in_filters[2],self.strides[2],self.filter_lengths[2],pool=None,param_names = ["W4",'b4'],border_mode = 'full')
		self.params+=self.deconv1.params
		self.deconv2 = one_d_conv_layer(self.deconv1.output,self.in_filters[0],self.in_filters[1],self.strides[1],self.filter_lengths[1],pool=None,param_names = ["W4",'b4'],border_mode = 'valid')
		self.params+=self.deconv2.params
		self.deconv3 = one_d_conv_layer(self.deconv2.output,1,self.in_filters[0],self.strides[0],self.filter_lengths[0],activation = None,param_names = ["W4",'b4'],pool=-3,border_mode = 'valid',distribution=False)
		self.params+=self.deconv3.params
		self.last_layer = self.deconv3
		self.trunc_output = self.deconv3.output[:,:,:,:self.inpt.shape[3]]
		#self.trunk_sigma =  self.deconv3.log_sigma[:,:,:,:self.inpt.shape[3]]
		# Replace this with variational layer
		#self.fully_connected = hidden_layer(T.flatten(self.layer3.output,outdim=2),735.,9.,param_names = ["W4",'b4'])
		#self.params+=self.fully_connected.params
		self.get_prior = theano.function([self.inpt],self.latent_layer.prior)
		self.convolve1 = theano.function([self.layer1.inpt],self.layer1.output)
		self.convolve2 = theano.function([self.layer1.inpt],self.layer2.output)
		self.convolve3 = theano.function([self.layer1.inpt],self.test)
		self.deconvolve3 = theano.function([self.layer1.inpt],self.trunc_output)
		self.deconvolve2 = theano.function([self.layer1.inpt],self.deconv2.output)
		self.output = theano.function([self.layer1.inpt],self.last_layer.output)
		self.cost = self.MSE()
		self.mse = self.MSE()
		print "gothere"
		self.get_cost = theano.function([self.layer1.inpt],[self.cost,self.mse])
		self.derivatives = T.grad(self.cost,self.params)
		self.get_gradients = theano.function([self.layer1.inpt],self.derivatives)
		self.updates = self.momentum_update()
		self.train_model = theano.function(inputs = [self.inpt],outputs = self.cost,updates = self.updates)

	def lower_bound(self):
		log_gauss =  0.5*np.log(2 * np.pi) + 0.5*self.trunk_sigma + 0.5 * ((self.inpt - self.trunc_output) / T.exp(self.trunk_sigma))**2
		test = T.flatten(log_gauss,outdim=2)
		test2 = T.sum(test,axis=1)
		return T.mean(test2- self.latent_layer.prior)
	def log_px_z(self):
		#TODO THERE IS A BUG HERE!!!
		
		log_gauss = 0.5*np.log(2 * np.pi) + 0.5*self.trunk_sigma + 0.5 * ((self.inpt - self.trunc_output) / T.exp(self.trunk_sigma))**2
		test = T.flatten(log_gauss,outdim=2)
		test2 = T.sum(test,axis=1)
		return T.mean(test2)
	def q_approx(self):
		return T.mean(-self.latent_layer.prior)
	def MSE(self):
		#self.cost = T.mean(T.sum((self.y-self.fully_connected.output)**2))
		m = T.flatten((self.inpt-self.trunc_output)**2,outdim=2)
		return T.mean(m)

	def adam(self,beta1=0.01, beta2=0.001,epsilon=1e-8, gamma=1-1e-8):
	    updates = []
	    i = theano.shared(np.float32(1))  # HOW to init scalar shared?
	    i_t = i + 1.
	    fix1 = 1. - (1. - beta1)**i_t
	    fix2 = 1. - (1. - beta2)**i_t
	    beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED
	    learning_rate_t = self.learning_rate * (T.sqrt(fix2) / fix1)

	    for param_i, g in zip(self.params, self.derivatives):
	        m = theano.shared(
	            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
	        v = theano.shared(
	            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

	        m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t to use beta1_t
	        v_t = (beta2 * g**2) + ((1. - beta2) * v)
	        g_t = m_t / (T.sqrt(v_t) + epsilon)
	        param_i_t = param_i - (learning_rate_t * g_t)

	        updates.append((m, m_t))
	        updates.append((v, v_t))
	        updates.append((param_i, param_i_t) )
	    updates.append((i, i_t))
	    return updates
	def adadelta(self, rho=0.95, epsilon=1e-6):
	    """ Adadelta updates
	    Scale learning rates by a the ratio of accumulated gradients to accumulated
	    step sizes, see [1]_ and notes for further description.
	    Parameters
	    ----------
	    loss_or_grads : symbolic expression or list of expressions
	        A scalar loss expression, or a list of gradient expressions
	    params : list of shared variables
	        The variables to generate update expressions for
	    learning_rate : float or symbolic scalar
	        The learning rate controlling the size of update steps
	    rho : float or symbolic scalar
	        Squared gradient moving average decay factor
	    epsilon : float or symbolic scalar
	        Small value added for numerical stability
	    Returns
	    -------
	    OrderedDict
	        A dictionary mapping each parameter to its update expression
	    Notes
	    -----
	    rho should be between 0 and 1. A value of rho close to 1 will decay the
	    moving average slowly and a value close to 0 will decay the moving average
	    fast.
	    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
	    work for multiple datasets (MNIST, speech).
	    In the paper, no learning rate is considered (so learning_rate=1.0).
	    Probably best to keep it at this value.
	    epsilon is important for the very first update (so the numerator does
	    not become 0).
	    Using the step size eta and a decay factor rho the learning rate is
	    calculated as:
	    .. math::
	       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
	       \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
	                             {\sqrt{r_t + \epsilon}}\\\\
	       s_t &= \\rho s_{t-1} + (1-\\rho)*g^2
	    References
	    ----------
	    .. [1] Zeiler, M. D. (2012):
	           ADADELTA: An Adaptive Learning Rate Method.
	           arXiv Preprint arXiv:1212.5701.
	    """
	    updates = OrderedDict()

	    for param, grad in zip(self.params, self.derivatives):
	        value = param.get_value(borrow=True)
	        # accu: accumulate gradient magnitudes
	        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
	                             broadcastable=param.broadcastable)
	        # delta_accu: accumulate update magnitudes (recursively!)
	        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
	                                   broadcastable=param.broadcastable)

	        # update accu (as in rmsprop)
	        accu_new = rho * accu + (1 - rho) * grad ** 2
	        updates[accu] = accu_new

	        # compute parameter update, using the 'old' delta_accu
	        update = (grad * T.sqrt(delta_accu + epsilon) /
	                  T.sqrt(accu_new + epsilon))
	        updates[param] = param - self.learning_rate * update

	        # update delta_accu (as accu, but accumulating updates)
	        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
	        updates[delta_accu] = delta_accu_new

	    return updates


	def momentum_update(self):
		lr = self.learning_rate
		updates = []
		for param,der in zip(self.params,self.derivatives):
			velocity = theano.shared(param.get_value()*0.,broadcastable = param.broadcastable)
			# velocity should be negative
			updates.append((param,param + lr*velocity))
			updates.append((velocity,self.momentum*velocity - (1-self.momentum)*lr*der))
		updates.append((self.learning_rate,self.learning_rate*1.0009))
		return updates
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

if __name__ == "__main__":
	data_dictionary = pickle_loader("sound/sine_mixture.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
	pdb.set_trace() 
	data = data_dictionary["data"]
	sample_rate = data_dictionary["sample rate"]
	# map inputs from 0 to 1
	# patch and shape for the CNN
	mean = np.mean(data)
	std = np.std(data)
	#data = (data -mean)/std
	data = data.reshape(data.shape[0],1,1,data.shape[1])
	dim_z = 10
	# train and test
	idx = np.random.permutation(data.shape[0])
	x_train = data;x_test = data
	net = convVAE(dim_z,x_train,x_test)
	out1 = net.convolve3(net.x_train[:1,:])
	pdb.set_trace()

	iterations = 1000
	for i in range(iterations):
		net.iterate()
		print "ITERATION",i
		if i%50==0:
			net.learning_rate/=4.
		print net.performance['train'][-1]
	out_labels = net.output(x_train)
	print out_labels[0,:]
	pdb.set_trace()
	#use np.squeeze to remove redundant dimentions.
	#out = out.reshape(out.shape[0],out.shape[1],out.shape[3])
	#print out2.shape



