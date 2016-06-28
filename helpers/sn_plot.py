import matplotlib.pyplot as plt
import pdb
import numpy as np

def plot_reconstructed(data,reconstructions,figure_handle,grid_shape = None,interactive = False):
	#plots reconstructed signals on top of the originals
	#Input shape: N*D
	plt.figure(figure_handle.number)
	if interactive == True:
		#plt.ion()
		plt.clf()
	#Plot Shapes. if shape is given 
	if grid_shape !=None:
		subplot_dims = grid_shape	
	elif reconstructions.shape[0]==1:
		subplot_dims = [1,1]
	elif reconstructions.shape[0]%2 ==0:
		subplot_dims  = [2,reconstructions.shape[0]/2]
	else:
		subplot_dims = [3,reconstructions.shape[0]/2]

	assert data.shape[1] == reconstructions.shape[1] , "dimentions not right %s,%s" % (data.shape[1], reconstructions.shape[1])

	#new_data = data[:reconstructions.shape[0],:] # Just in case the the input data is not the dame dimention as the reconstructions
	for i, (d, mu) in enumerate(zip(data,reconstructions)):
		figure_handle.add_subplot(subplot_dims[0],subplot_dims[1],i+1)
		plt.plot(d)
		plt.plot(mu)
	if interactive == True:
		#plt.ioff()
		plt.draw()


def plot_filters(weight_tensor,figure_handle,interactive=False):
	weight_values = weight_tensor.get_value()[:,0,0,:]
	no_of_filters = 15
	plt.figure(figure_handle.number)
	print "OTHER",figure_handle.number
	rows = np.floor(np.sqrt(no_of_filters))
	columns = np.ceil(no_of_filters/float(rows))
	print "FUNCTION HANDLE",figure_handle
	if interactive == True:
		plt.clf()
	for i in range(no_of_filters):
		figure_handle.add_subplot(rows,columns,i+1)
		plt.plot(weight_values[i,:])
	plt.pause(0.01)
	if interactive == True:
		plt.draw()
	else:
		plt.show()


def plot_params(params,directory):
	if directory[-1]!="/":
		directory +="/"
	for n,param in enumerate(params):
		value = param.get_value()
		#print value.shape
		sq = np.squeeze(value)
		print sq.shape
		if len(value.shape)==4:
			# loop though number of filters
			for i in range(sq.shape[0]):
				f = plt.figure()
				plt.plot(sq[i,:])
				# first number is parameter set from lower to higher layer.
				# second number is filter number withn that parameter set.
				name = str(n)+"_"+str(i)
				f.savefig(directory+name+".png",)











             
          

