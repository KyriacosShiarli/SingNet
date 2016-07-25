import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pdb
from theano.tensor.signal import downsample as ds
from theano.tensor import shared_randomstreams as t_random

def zeros_patch(loop_number,output,original,pool):
	idx = loop_number*pool
	subtensor = output[:,:,idx.astype("int32"),:]
	value = original[:,:,loop_number,:]
	return T.set_subtensor(subtensor, value)

def zeros_unpool(inpt,pool):
	output = T.zeros_like(inpt.repeat(int(pool),axis=2)).astype(theano.config.floatX)
	loop_number = inpt.shape[2]

	result, updates = theano.scan(fn=zeros_patch,
	                              outputs_info=output,
	                              sequences=T.arange(loop_number),
	                              non_sequences=[inpt,pool])
	return result[-1]