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
import numpy as np
import pdb
from preprocess import map_to_range
from sn_plot import plot_filters,plot_params
import sn_play as sound_write
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import time
from convVAE import convVAE

#from sn_play import device_play

# REMBEMBER for convilutional layers. Input tensor = (batch,channels,xpixels,ypixels)
# xpixels is always 1. ypixels is the only one that changes because we have a 1D convolution in this case.

# Weight tensor - (in_channels,out_channels,filter_x,filter_y). filter_x = 1 since we are only 
#filtering in one direction. 


if __name__=="__main__":

	############## ALL THE BOOKEEPING #########################

	data_dir = parent+"/data/"
	plot_dir = current+"/plots/"
	sound_dir = current+"/sound/"
	model_dir = current+"/models/"
	weight_plot_dir = current+"/plots/weights"


	data= pickle_loader(data_dir+"sines.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
	data = np.ndarray.astype(data, np.float32)

	data2 = data[:,1:]
	data1 = data[:,:-1]

	dif = np.hstack([1+np.absolute((data1-data2)/2),np.ones([data.shape[0],1]).astype(np.float32)])

	mean = np.mean(data)
	std = np.std(data)
	sample_rate = 880
	#data = (data -mean)/std
	data = data.reshape(data.shape[0],1,data.shape[1],1)
	dim_z = 5

	# train and test
	x_train = data[:,:,:,:];x_test = data
	dif = dif[:,:]
	# Discover the magic number
	net = convVAE(dim_z,x_train,x_test)
	get_magic = net.get_flattened(net.x_train[:2,:,:,:])
	# actual nework
	#net1 = convVAE(dim_z,x_train,x_test,diff = dif,magic = get_magic.shape[1])
	net = convVAE(dim_z,x_train,x_test,diff=None,magic = get_magic.shape[1])

	# load the parameters from the model:
	params = pickle_loader(model_dir+"model.pkl")

	for p in params:
		net.params.set_value(p.get_value())


	# Writing paths
	path_array,original_array = sound_write.path_write(net,sample_rate,duration = 3.,data_points = 10)
	wav.write("sound/"+experiment_name+"_paths_rewrite.wav",sample_rate,path_array)
	wav.write("sound/"+experiment_name+"_paths_original_rewrite.wav",sample_rate,original_array)
	# Reconstruction sounds and random latent configurations
	random,reconstruction,original = sound_write.sample_write(net,sample_rate,duration = 10)
	wav.write("sound/"+experiment_name+"_random_rewrite.wav",sample_rate,random)
	wav.write("sound/"+experiment_name+"_reconstruction_rewrite.wav",sample_rate,reconstruction)
	wav.write("sound/"+experiment_name+"_reconstruction_original_rewrite.wav",sample_rate,original)
	# Weights
	plot_params(net.params,weight_plot_dir)
	#ou1 = net1.output(x_train[:50])
	idxx = np.random.randint(0,x_train.shape[0],50)
	# Reconstruction images
	ou2 = net.output(x_train[idxx])
	for i in range(ou2.shape[0]):
		if i <100:
			plt.figure()
	#		plt.plot(ou1[i,0,:,0],color = "g")
			plt.plot(ou2[i,0,:,0],color = "b")
			plt.plot(x_train[i,0,:,0],color = "r")
			plt.savefig(plot_dir+str(i)+"_compare.png",cmap=plt.cm.binary)

			
	# The model hyper parameters
	f = open("readme","w")
	f.write("SingVAE parameters for experiment: \n------ \n \n")
	f.write("Number of filters:  "+ str([net1.in_filters]) +"\n")
	f.write("Filter lengths:  "+str([net1.filter_lengths])+"\n")
	f.write("Latend variables:  "+str(dim_z)+"\n")
	f.write("Iterations:  "+str(iterations)+"\n")
	f.write("Sample rate:  "+str(sample_rate)+"\n")
	f.close()



