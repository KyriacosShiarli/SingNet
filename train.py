from convVAE2 import *
import pdb




data = pickle_loader("sound/instruments.pkl") # data dictionary contains a "data" entry and a "sample rate" entry


# map inputs from 0 to 1
# patch and shape for the CNN
data = data.reshape(data.shape[0],1,1,data.shape[1])
dim_z = 10
# train and test
idx = np.random.permutation(data.shape[0])
x_train = data[:10,:];x_test = data

#mean = np.mean(x_train,axis = 0)
#std = np.std(x_train,axis = 0)
#print mean,std
#x_train = (x_train -mean)

net = convVAE(dim_z,x_train,x_test)
#out1 = net.get_cost(net.x_train[:3,:])
out1 = net.convolve3(x_train)
pdb.set_trace()
iterations = 1200
for i in range(iterations):
	net.iterate()
	print "ITERATION",i
	if i%50==0:
		net.learning_rate/=2.
	print net.performance['train'][-1]
out_labels = net.output(x_train)
print out_labels[0,:]
pdb.set_trace()
#use np.squeeze to remove redundant dimentions.
#out = out.reshape(out.shape[0],out.shape[1],out.shape[3])
#print out2.shape