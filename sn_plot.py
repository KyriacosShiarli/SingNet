import matplotlib.pyplot as plt


def plot_reconstructed(data,reconstructions,figure_handle,grid_shape = None,interactive = False):
	#plots reconstructed signals on top of the originals
	#Input shape: N*D
	plt.figure(figure_handle.number)
	print "HANDLE",figure_handle.number
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
	print plt.gcf
	for i, (d, mu) in enumerate(zip(data,reconstructions)):
		figure_handle.add_subplot(subplot_dims[0],subplot_dims[1],i+1)
		plt.plot(d)
		plt.plot(mu)
	if interactive == True:
		#plt.ioff()
		plt.draw()



             
          

