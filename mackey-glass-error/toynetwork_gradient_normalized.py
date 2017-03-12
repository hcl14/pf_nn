#http://iamtrask.github.io/2015/07/12/basic-python-network/ 
import csv
import numpy as np
import matplotlib.pyplot as plt

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
#X = np.array([[0,0,1],
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])
                
#y = np.array([[0],
#			[1],
#			[1],
#			[0]])

def row_to_arrays(col,l1_num,l2_num):  # we store w0, w1 as single column to iterate easily
    w0 = col[0:l1_num*l2_num].reshape((l1_num,l2_num))
    w1 = col[l1_num*l2_num:(l1_num*l2_num+l2_num)].reshape((l2_num,1))
    return w0,w1

np.random.seed(1)

# read treaining data
f = open('training.csv', 'rt')
#reader = csv.reader(f)
result = [row for row in csv.reader(f, delimiter=',')]
X1 = np.array(result).astype('float64')
X1 = X1/np.max(X1)

print X1.shape

num_batches = np.int(X1.shape[1]-1)

print 'batches found: ', num_batches

# randomly initialize our weights with mean 0
#syn0 = 2*np.random.random((3,4)) - 1
#syn1 = 2*np.random.random((4,1)) - 1


# Layers: 6, 3, 1
# No. of weights = 6*3 + 3*1 = 18 + 3 = 21

l1_num = 19 # neurons in first layer
l2_num = 4 # neurons in second layer
w = 2*np.random.random((l1_num*l2_num+l2_num,1)) - 1 # we store w0 and w1 as single column to iterate easily
w_1 = None # stored chosen models



for j in xrange(6000):
    
    # define the vector of cumulative error over all batches 
    l2_cumulative_errors = 0
    # gradients for every batch
    l_grad = np.zeros((l1_num*l2_num+l2_num,num_batches-l1_num))
    
    # iterate batches and propagate models on them
    for batch_count in range(num_batches-l1_num): # 0.. num_batches-l1_num -1 
        
        # select current batch
        l0 = X1[0,batch_count:(batch_count+l1_num)].astype('float64')
        y = X1[0,(batch_count+l1_num)].astype('float64')
        y = y.reshape((1,1))
        l0 = l0.reshape((1,l1_num))
        
        w0,w1 = row_to_arrays(w[:,0],l1_num,l2_num)
        # Feed forward through layers 0, 1, and 2
        l1 = nonlin(np.dot(l0,w0))
        l2 = nonlin(np.dot(l1,w1))
        
        layer2_error = y - l2
        l2_cumulative_errors += layer2_error**2
        
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = layer2_error*nonlin(l2,deriv=True)
        
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer1_error = l2_delta.dot(w1.T)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = layer1_error * nonlin(l1,deriv=True)
        
        
        l_grad[(l1_num*l2_num):(l1_num*l2_num+l2_num),batch_count] = (l1.T.dot(l2_delta)).ravel()
        l_grad[0:(l1_num*l2_num),batch_count] = (l0.T.dot(l1_delta)).ravel()
        
    #update weights according to the mean gradient on all batches
    mean_grad = np.mean(l_grad,axis = 1)
    
    w0,w1 = row_to_arrays(w[:,0],l1_num,l2_num)
    w1 += (mean_grad[(l1_num*l2_num):(l1_num*l2_num+l2_num)]).reshape((l2_num,1))
    w0 += (mean_grad[0:(l1_num*l2_num)]).reshape((l1_num,l2_num))
    if (j% 500) == 0:
        print "mean l2 error per batch:", np.round((l2_cumulative_errors+0.0)/(num_batches-l1_num),decimals=5)
    


print "Output After Training:"

# Read validation data
f = open('validation.csv', 'rt')
result = [row for row in csv.reader(f, delimiter=',')]
X2 = np.array(result).astype('float64')
X2 = X2/np.max(X2)
print 'validation points found: ', X2.shape[1]



# propagate to get result:


#prepare predictions set
predictions = np.zeros(X2.shape[1])

# first values from testing set
inputs = (X1[0,(X1.shape[1]-l1_num):(X1.shape[1])].astype('float64')).reshape((1,l1_num))
#print inputs
# start predicting on validation data
for batch_count in range(X2.shape[1]):
#for batch_count in range(25):
    w0,w1 = row_to_arrays(w[:,0],l1_num,l2_num)
    l1 = nonlin(np.dot(inputs,w0))
    l2 = nonlin(np.dot(l1,w1))
    # add prediction to set
    predictions[batch_count] = l2
    # add prediction and shift input set
    inputs = np.hstack((inputs,l2))
    inputs = np.delete(inputs, 0)
    inputs = inputs.reshape((1,l1_num))

#print np.round(l2,decimals=3)

#print 'real: ', y, 'err: ', np.sum((y - l2)**2)


#plot data
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
plt.plot(range(X1.shape[1]),X1.ravel(),c='black', label='Data')
plt.plot(range(X1.shape[1]),X1.ravel(),'.',c='black', label='Data')
plt.plot(range(X1.shape[1],(X1.shape[1]+X2.shape[1])),X2.ravel(),c='blue', label='Data')
plt.plot(range(X1.shape[1],(X1.shape[1]+X2.shape[1])),X2.ravel(),'.',c='blue', label='Data')
plt.plot(range(X1.shape[1],(X1.shape[1]+X2.shape[1])),predictions,c='green', label='Data')
plt.plot(range(X1.shape[1],(X1.shape[1]+X2.shape[1])),predictions,'.',c='green', label='Data')

plt.show()