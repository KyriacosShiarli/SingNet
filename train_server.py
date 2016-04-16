from convVAE2_basic import convVAE
import numpy as np
import pdb

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from preprocess import pickle_saver,pickle_loader
from sn_play import sample_write



data= pickle_loader("sound/sines.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
data = np.ndarray.astype(data, np.float32)
data2 = data[:,1:]
data1 = data[:,:-1]
dif = np.hstack([1+np.absolute((data1-data2)/2),np.ones([data.shape[0],1]).astype(np.float32)])
# patch and shape for the CNN
#data = (data -mean)/std
data = data.reshape(data.shape[0],1,data.shape[1],1)
dim_z = 200
sample_rate = 880
#enable interactive plotting

# train and test
#idx = np.random.permutation(data.shape[0])
x_train = data[:2];x_test = data
dif = dif[:2]
# Discover the magic number
net = convVAE(dim_z,x_train,x_test)
get_magic = net.get_flattened(net.x_train[:2,:,:,:])
# actual nework
net1 = convVAE(dim_z,x_train,x_test,diff = dif,magic = get_magic.shape[1])
net2 = convVAE(dim_z,x_train,x_test,diff=None,magic = get_magic.shape[1])
iterations = 2
for i in range(iterations):
    net1.iterate()
    net2.iterate()
    print "ITERATION",i
    print net2.performance['train'][-1]

net1.dropout_prob.set_value(np.float32(0.0))
net2.dropout_prob.set_value(np.float32(0.0))
pickle_saver(net1.params,"models/net_boost_home_basic.pkl")
pickle_saver(net2.params,"models/net_no_boost_basic.pkl")

plot_size = 100
idxx = np.random.randint(0,x_train.shape[0],plot_size)
ou1 = net1.output(x_train[idxx])
ou2 = net2.output(x_train[idxx])

sample_write(net1,sample_rate,name ="1_layer_boost" ,duration=15,sample_for_latents=20)
sample_write(net2,sample_rate,name ="1_layer_no_boost" ,duration=15,sample_for_latents=20)
results = {"original":x_train[idxx],"boosted":ou1,"no_boost":ou2}
pickle_saver(results,"images/results_for_figures.pkl")
for i in range(ou2.shape[0]):
    #plt.ioff()
    fig=plt.figure()
    plt.plot(ou1[i,0,:,0],color = "g")
    plt.plot(ou2[i,0,:,0],color = "b")
    plt.plot(x_train[i,0,:,0],color = "r")
    fig.savefig("images/"+str(i)+"_compare_20epoch.png",cmap=plt.cm.binary)
