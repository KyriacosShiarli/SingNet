def relu(x):
    return T.switch(x<0, 0, x)
def det_x(a_11,a_12,a_21,a_22):
	#symbolic determinant of 2 by 2 matrix
	return a_11*a_22 - a_12*a_21

		#self. = theano.function([inpt], output)
