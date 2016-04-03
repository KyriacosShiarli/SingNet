import scipy.io.wavfile
import numpy as np
import cPickle as pickle
import pdb
import matplotlib.pyplot as plt

def pickle_saver(to_be_saved,full_directory):
	with open(full_directory,'wb') as output:
		pickle.dump(to_be_saved,output,-1)

def pickle_loader(full_directory):
	with open(full_directory,'rb') as input:
		return pickle.load(input)

def map_to_range(data,input_range,output_range,from_data = False):
	if from_data==True:
		input_range =(np.amin(data),np.amax(data),)
	data = np.array(data)
	grad = (np.amax(output_range)-np.amin(output_range))/(np.amax(input_range)-np.amin(input_range))
	offset = np.amin(output_range) - np.amin(input_range)*grad
	data = data*grad+offset
	return data


def map_to_range_symmetric(data,input_range,output_range,from_data = False):
	# this function maps from one symmetric range to another.
	# e.g. -1,1 to -32768,32768. useful for playback
	if from_data==True:
		input_range =(np.amin(data),np.amax(data),)
	data = map(float,data)
	grad = (np.amax(output_range)-np.amin(output_range))/(np.amax(input_range)-np.amin(input_range))
	data_temp = np.copy(data)*grad
	return data_temp

def load_and_split(directory,duration):
	rate,all_data= scipy.io.wavfile.read(directory+".wav")
	sample_length =rate*duration
	out_data = map_to_range_symmetric(all_data[:,0],[-32767. ,32767. ],[-1,1])
	num_datapoints = np.floor(out_data.shape[0]/sample_length)
	rem = np.floor(out_data.shape[0]%sample_length)
	out_data = np.reshape(out_data[:-rem],(num_datapoints,sample_length))
	data={"data":out_data,"sample rate":rate}
	pickle_saver(data,directory+".pkl")

def load_and_split_samples(directory,samples):
	rate,all_data= scipy.io.wavfile.read(directory+".wav")
	pdb.set_trace()

	all_data = map_to_range_symmetric(all_data,[-32767. ,32767. ],[-1,1])
	num_datapoints = np.floor(all_data.shape[0]/samples)
	rem = np.floor(all_data.shape[0]%samples)
	pdb.set_trace()
	data= np.reshape(all_data,(num_datapoints,samples))
	pickle_saver(data,directory+".pkl")


def load_pure_tone_data():

	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/A.wav")
	all_data = map_to_range(data,[-32767. ,32767. ],[-1,1])
	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/B.wav")
	all_data = np.vstack((all_data,map_to_range(data,[-32767. ,32767. ],[-1,1])))
	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/E.wav")
	all_data = np.vstack((all_data,map_to_range(data,[-32767. ,32767. ],[-1,1])))
	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/D.wav")
	all_data = np.vstack((all_data,map_to_range(data,[-32767. ,32767. ],[-1,1])))
	pickle_saver(all_data,"sound/puretone_data.pkl")

if __name__ == "__main__":
	load_and_split_samples("sound/sines",88)