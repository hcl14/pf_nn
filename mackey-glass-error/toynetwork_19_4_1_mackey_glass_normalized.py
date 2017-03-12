import csv
import numpy as np
import matplotlib.pyplot as plt

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
def row_to_arrays(col,l1_num,l2_num):  # we store w0, w1 as single column to iterate easily
    w0 = col[0:l1_num*l2_num].reshape((l1_num,l2_num))
    w1 = col[l1_num*l2_num:(l1_num*l2_num+l2_num)].reshape((l2_num,1))
    return w0,w1

#X = np.array([[0,0,1],
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])
                
#y = np.array([[0],
#			[1],
#			[1],
#			[0]])

# read treaining data
f = open('training.csv', 'rt')
#reader = csv.reader(f)
result = [row for row in csv.reader(f, delimiter=',')]
X1 = np.array(result).astype('float64')
X1 = X1/1.5

print X1.shape

num_batches = np.int(X1.shape[1]-1)

print 'batches found: ', num_batches

# choose one first window
#X = X1[0,0:6].reshape((1,6)).astype('float64')
#y = X1[0,6].astype('float64')


np.random.seed(1) # this random seed corresponds to the correct convergence, otherwise the model is very likely to end in local minima

# Sampling: initialize 100 different weight vectors from continuous uniform distribution (100 models)

# randomly initialize our weights with mean 0 for 100 models

# Layers: 6, 3, 1
# No. of weights = 6*3 + 3*1 = 18 + 3 = 21

l1_num = 19 # neurons in first layer
l2_num = 4 # neurons in second layer
w = 2*np.random.random((l1_num*l2_num+l2_num,100)) - 1 # we store w0 and w1 as single column to iterate easily
w_1 = None # stored chosen models

imp = 0

epochs = 20

eps1prev = 0 # previous values to avoid being stuck
eps2prev = 0
errprev = 0 
stucked = 0 # how much we stuck

for j in xrange(epochs):
    # Feed forward through layers 0, 1, and 2
    
    
    #lambda_r = 0.05 # regularization parameter
    
    
    # propagate this 100 models on every batch element
    # and then choose the one that gives minimal cumulative l2 error for all the batches
    
    # define the vector of cumulative error over all batches for all 100 models
    l2_cumulative_errors = np.zeros(100)
    # single batch l2_error for 100 models
    l2_error = np.zeros(100)
    
    # apply this 100 models over all batches
    for batch_count in range(num_batches-l1_num): # 0.. num_batches-l1_num -1 
        # select current batch
        l0 = X1[0,batch_count:(batch_count+l1_num)].astype('float64')
        y = X1[0,(batch_count+l1_num)].astype('float64')
        y = y.reshape((1,1))
        l0 = l0.reshape((1,l1_num))

        # compute error for current 100 models
        l2_error = np.zeros(100)
        l1, l2 = None,None
        for i in range(100):
            w0,w1 = row_to_arrays(w[:,i],l1_num,l2_num)
            l1 = nonlin(np.dot(l0,w0))
            l2 = nonlin(np.dot(l1,w1))
            # calculating error with regularization
            l2_error[i] = np.sum((y - l2)**2) # + lambda_r*(1.0/16)*np.sum(abs(w[:,i]))  # regularization either does nothing or destroys everything! The model tends then to [0.5,0.5,0.5,0.5]
        
        # add cumulative error
        l2_cumulative_errors += l2_error
    
    # for further simplicity
    l2_error = l2_cumulative_errors
    
    # Importance
    
    #imp = 1.0/l2_error 
    imp = np.divide(1.0, l2_error, where=l2_error!=0) # trying to avoid warning here, needs to be optimized
    
    # Resampling
    
    # approach 1: get area in ND around weighted ones
    
    # indices where l2 error == 0 should be taken immediately (need some more simple and robust procedure here)
    indices = np.where(l2_error==0)[0]
    # if there is no such ones
    if indices.size == 0:
        # then select indices of the 5% weights with largest likelihood 
        indices = np.argpartition(-imp, int(0.05*100))[:int(0.05*100)] 
    
    # choose these models
    w_1 = np.take(w,indices,axis=1)
    # extract their errors
    l2_error_chosen = np.take(l2_error,indices)
    
    # bootstrap with noise to create new sample of 100 elements located around these chosen
    
    # bootstrap - sample indices with replacement
    indices2 = np.random.choice(w_1.shape[1],100)
    # take elements
    w = np.take(w_1,indices2,axis=1)
    
    # add uniform noises to the wights (different training speed for different layers taken into account)
    
    #C1 = 0.1 # works like charm, but takes 60 iterations
    #C2 = 0.2
    
    # min is adequate measure (we need to pay most attention to the closest ones), also it allows us to stop when we encounter zero
    err = np.min(l2_error_chosen)
    
    C1 = 0.2*err # The closer we are to the solution, the lesser are the steps. Gives significant speed improvement!
    C2 = 0.4*err # But works a bit weird sometimes (l2 error shifts instantly to the either side, weights explode, no convergence in rare occasions)
    
    #first layer
    eps1 = C1*abs(np.amax(w[0:l1_num*l2_num,:],axis=1)-np.amin(w[0:l1_num*l2_num,:],axis=1)) * (100**(-1.0/l1_num*l2_num))   # weights still explode with these parameters sometimes, if there are very different weight values on the layer
    #second layer
    eps2 = C2*abs(np.amax(w[l1_num*l2_num:(l1_num*l2_num+l2_num),:],axis=1)-np.amin(w[l1_num*l2_num:(l1_num*l2_num+l2_num),:],axis=1)) * (100**(-1.0/l2_num)) # if the weights across the row are approximately the same, algorithm stucks even if error is >> 0 and needs to be revitalized by adding extra noise
    
    #threshold values to avoid weight explosion
    eps1[eps1 > 50] = 50
    eps2[eps2 > 50] = 50
    
    #revitalizing stuck training
    delta_eps1 = abs(eps1 - eps1prev)
    delta_eps2 = abs(eps2 - eps2prev) #Will keepthese to maybe take into account when only one layer stops training sometimes
    delta_errprev = abs(err - errprev)
    
    eps1prev = eps1  # saving previous values
    eps2prev = eps2
    errprev = err
    
    #if (delta_errprev < 0.01*err): # empirical, when we need to push out network. Such step 0.01 is already too slow for particle filter 
    #    print "stuck, adding +10% noise"
    #    stucked += 1
    #    eps1 = np.abs(np.min(w[0:18,:],axis = 1)*0.1*stucked)
    #    eps2 = np.abs(np.min(w[18:21,:],axis = 1)*0.1*stucked)
    #else:
    #    stucked = 0
    
    #print "*",max(eps1)
    #print "*",max(eps2)
    
    
    if j<(epochs-1): # to avoid noising at the end
        
        w[0:l1_num*l2_num,:] = np.add(w[0:l1_num*l2_num,:], np.multiply(np.random.random((l1_num*l2_num,100)),2*eps1.reshape((l1_num*l2_num,1))) - eps1.reshape((l1_num*l2_num,1)))
        w[l1_num*l2_num:(l1_num*l2_num+l2_num),:] = np.add(w[l1_num*l2_num:(l1_num*l2_num+l2_num),:], np.multiply(np.random.random((l2_num,100)),2*eps2.reshape((l2_num,1))) - eps2.reshape((l2_num,1)))
    
    print 'l2_error: ', err, '( MSE per batch: ', np.round((err+0.0)/num_batches,decimals=3), ')'
    

print "Output After Training:"

# Read validation data
f = open('validation.csv', 'rt')
result = [row for row in csv.reader(f, delimiter=',')]
X2 = np.array(result).astype('float64')
X2 = X2/1.5
print 'validation points found: ', X2.shape[1]


# select index that correspond to the smallest error and
# propagate to get result:
model_ind = np.argmin(l2_error_chosen)

#prepare predictions set
predictions = np.zeros(X2.shape[1])

# first values from testing set
inputs = (X1[0,(X1.shape[1]-l1_num):(X1.shape[1])].astype('float64')).reshape((1,l1_num))
print inputs
# start predicting on validation data
for batch_count in range(X2.shape[1]):
#for batch_count in range(25):
    w0,w1 = row_to_arrays(w_1[:,model_ind],l1_num,l2_num)
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