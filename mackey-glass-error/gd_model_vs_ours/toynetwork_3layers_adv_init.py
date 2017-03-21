#http://iamtrask.github.io/2015/07/12/basic-python-network/ 
import csv
import numpy as np
import matplotlib.pyplot as plt

def nonlin(x,deriv=False):
	if(deriv==True):
	    return 1.0 - np.tanh(x)**2

	return np.tanh(x)
 
# Relu
#def nonlin(x,deriv=False):
#	if(deriv==True):
#	    return 1*(x>0)
#
#	return x * (x>0)

def row_to_arrays(col,l1_num,l2_num,l3_num):  # we store w0, w1 as single column to iterate easily
    w0 = col[0:l1_num*l2_num].reshape((l1_num,l2_num))
    w1 = col[l1_num*l2_num:(l1_num*l2_num+l2_num*l3_num)].reshape((l2_num,l3_num))
    w2 = col[(l1_num*l2_num+l2_num*l3_num):(l1_num*l2_num+l2_num*l3_num+l3_num)].reshape((l3_num,1))
    return w0,w1,w2

def get_layers(col,l1_num,l2_num,l3_num):
    w0 = col[0:l1_num*l2_num,:]
    w1 = col[l1_num*l2_num:(l1_num*l2_num+l2_num*l3_num),:]
    w2 = col[(l1_num*l2_num+l2_num*l3_num):(l1_num*l2_num+l2_num*l3_num+l3_num),:]
    return w0,w1,w2

    
#X = np.array([[0,0,1],
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])
                
#y = np.array([[0],
#			[1],
#			[1],
#			[0]])

#np.random.seed(60000)
np.random.seed(30)

# read treaining data
f = open('training.csv', 'rt')
#reader = csv.reader(f)
result = [row for row in csv.reader(f, delimiter=',')]
X1 = np.array(result).astype('float64')
X1 = X1/1.5

print X1.shape

num_batches = np.int(X1.shape[1]-1)

print 'batches found: ', num_batches

# randomly initialize our weights with mean 0
#syn0 = 2*np.random.random((3,4)) - 1
#syn1 = 2*np.random.random((4,1)) - 1

l1_window = 20 #total moving window
#subset = [0,6,12,18]
#l1_num = 4 # neurons in first layer
#l2_num = 6 # neurons in second layer
#l3_num = 3 #neurons for third layer

subset = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
l1_num = 20 # neurons in first layer
l2_num = 20 # neurons in second layer
l3_num = 20 #neurons for third layer


num_particles = 1 # we're performing gradient descent here

# initializing weights according to the number of input connections and CLT
w = 2*np.zeros((l1_num*l2_num+l2_num*l3_num+l3_num,num_particles))

n_1 = l1_num
n_2 = l1_num*l2_num
n_3 = l2_num*l3_num
n_4 = l3_num

range_i = np.sqrt(6)/np.sqrt(n_1+n_2) # just for convenience of editing

w[0:l1_num*l2_num,:] = (2*np.random.random((l1_num*l2_num,num_particles)) - 1)*range_i

range_i = np.sqrt(6)/np.sqrt(n_2+n_3)

w[l1_num*l2_num:(l1_num*l2_num+l2_num*l3_num),:] = (2*np.random.random((l2_num*l3_num,num_particles)) - 1)*range_i

range_i = np.sqrt(6)/np.sqrt(n_3+n_4)

w[(l1_num*l2_num+l2_num*l3_num):(l1_num*l2_num+l2_num*l3_num+l3_num),:] = (2*np.random.random((l3_num,num_particles)) - 1)*range_i




epochs = 100000

for j in xrange(1,epochs):
    # Feed forward through layers 0, 1, and 2
    #l0 = X
    #l1 = nonlin(np.dot(l0,syn0))
    #l2 = nonlin(np.dot(l1,syn1))
    
    # how much did we miss the target value?
    #l2_error = y - l2
    
    w0,w1,w2 = get_layers(w,l1_num,l2_num,l3_num)
    
    
    l3_deltas = np.zeros((num_batches-l1_window,w2.shape[0]))
    l2_deltas = np.zeros((num_batches-l1_window,w1.shape[0]))
    l1_deltas = np.zeros((num_batches-l1_window,w0.shape[0]))
    
    layer3_error = 0
    
    for batch_count in range(num_batches-l1_window): # 0.. num_batches-l1_num -1 
        # select current batch                
        # select moving window
        l0 = X1[0,batch_count:(batch_count+l1_window)].astype('float64')
        #select 1,7,13,19
        l0 = l0[subset]
        
        y = X1[0,(batch_count+l1_window)].astype('float64')
        y = y.reshape((1,1))
        l0 = l0.reshape((1,l1_num))
        
        # compute error for current num_particles models
        layer3_error = np.zeros(1)
        l1, l2 = None,None
        #for i in range(num_particles): # just for compatibility, we have 1 there
        w0,w1,w2 = row_to_arrays(w[:,0],l1_num,l2_num,l3_num)
        l1 = nonlin(np.dot(l0,w0))
        l2 = nonlin(np.dot(l1,w1))
        l3 = nonlin(np.dot(l2,w2))
        # calculating error with regularization
        layer3_error = y - l3 # + lambda_r*(1.0/16)*np.sum(abs(w[:,i]))  # regularization either does nothing or destroys everything! The model tends then to [0.5,0.5,0.5,0.5]
        
            
        w0,w1,w2 = row_to_arrays(w[:,0],l1_num,l2_num,l3_num)
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l3_delta = layer3_error*nonlin(l3,deriv=True)
        
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer2_error = l3_delta.dot(w2.T)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        
        
        l2_delta = layer2_error * nonlin(l2,deriv=True)
        
        layer1_error = l2_delta.dot(w1.T)
        
        l1_delta = layer1_error * nonlin(l1,deriv=True)
        
        
        l3_deltas[batch_count,:] = l2.T.dot(l3_delta).ravel()
        l2_deltas[batch_count,:] = l1.T.dot(l2_delta).ravel()
        l1_deltas[batch_count,:] = l0.T.dot(l1_delta).ravel()
    
    if (j% 100) == 0:
            print "Error:" + str(np.mean(np.abs(layer3_error)))+ " "+str(j)
    
    w0,w1,w2 = get_layers(w,l1_num,l2_num,l3_num)
    

    w0 += np.mean(l1_deltas,axis = 0).reshape((w0.shape[0],1))
    w1 += np.mean(l2_deltas,axis = 0).reshape((w1.shape[0],1))
    w2 += np.mean(l3_deltas,axis = 0).reshape((w2.shape[0],1))


print "Output After Training:"

# Read validation data
f = open('validation.csv', 'rt')
result = [row for row in csv.reader(f, delimiter=',')]
X2 = np.array(result).astype('float64')
X2 = X2/1.5
print 'validation points found: ', X2.shape[1]


# select index that correspond to the smallest error and
# propagate to get result:
model_ind = 0 #np.argmin(l2_error_chosen)

#prepare predictions set
predictions = np.zeros(X2.shape[1])

# first values from testing set
inputs = (X1[0,(X1.shape[1]-l1_window):(X1.shape[1])].astype('float64')).reshape((1,l1_window))
print inputs
# start predicting on validation data
for batch_count in range(X2.shape[1]):
#for batch_count in range(25):
    w0,w1,w2 = row_to_arrays(w[:,model_ind],l1_num,l2_num,l3_num)
    
    l0 = inputs[0,subset].reshape((1,l1_num))
    
    l1 = nonlin(np.dot(l0,w0))
    l2 = nonlin(np.dot(l1,w1))
    l3 = nonlin(np.dot(l2,w2))
    # add prediction to set
    predictions[batch_count] = l3
    # add prediction and shift input set
    inputs = np.hstack((inputs,l3))
    inputs = np.delete(inputs, 0)
    inputs = inputs.reshape((1,l1_window))

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