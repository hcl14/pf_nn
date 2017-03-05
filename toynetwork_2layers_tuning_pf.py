
import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
def row_to_arrays(col):  # we store w0, w1 as single column to iterate easily
    w0 = col[0:12].reshape((3,4))
    w1 = col[12:16].reshape((4,1))
    return w0,w1

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

#np.random.seed(1) # this random seed corresponds to the correct convergence, otherwise the model is very likely to end in local minima

# Sampling: initialize 100 different weight vectors from continuous uniform distribution (100 models)

# randomly initialize our weights with mean 0
#w0 = 2*np.random.random((3,4)) - 1
#w1 = 2*np.random.random((4,1)) - 1

w = 2*np.random.random((16,100)) - 1 # we store w0 and w1 as single column to iterate easily
w1 = None # stored chosen models

imp = 0

epochs = 25

eps1prev = 0 # previous values to avoid being stuck
eps2prev = 0
errprev = 0 
stucked = 0 # how much we stuck

for j in xrange(epochs):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    
    #lambda_r = 0.05 # regularization parameter
    
    l2_error = np.zeros(100)
    l1, l2 = None,None
    for i in range(100):
        w0,w1 = row_to_arrays(w[:,i])
        l1 = nonlin(np.dot(l0,w0))
        l2 = nonlin(np.dot(l1,w1))
        # calculating error with regularization
        l2_error[i] = np.sum((y - l2)**2) # + lambda_r*(1.0/16)*np.sum(abs(w[:,i]))  # regularization either does nothing or destroys everything! The model tends then to [0.5,0.5,0.5,0.5]
    
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
    w1 = np.take(w,indices,axis=1)
    # extract their errors
    l2_error_chosen = np.take(l2_error,indices)
    
    # bootstrap with noise to create new sample of 100 elements located around these chosen
    
    # bootstrap - sample indices with replacement
    indices2 = np.random.choice(w1.shape[1],100)
    # take elements
    w = np.take(w1,indices2,axis=1)
    
    # add uniform noises to the wights (different training speed for different layers taken into account)
    
    #C1 = 0.1 # works like charm, but takes 60 iterations
    #C2 = 0.2
    
    # min is adequate measure (we need to pay most attention to the closest ones), also it allows us to stop when we encounter zero
    err = np.min(l2_error_chosen)
    
    C1 = 16*err # The closer we are to the solution, the lesser are the steps. Gives significant speed improvement!
    C2 = 8*err # But works a bit weird sometimes (l2 error shifts instantly to the either side, weights explode, no convergence in rare occasions)
    
    #first layer
    eps1 = C1*abs(np.amax(w[0:12,:],axis=1)-np.amin(w[0:12,:],axis=1)) * (100**(-1.0/12))   # weights still explode with these parameters sometimes, if there are very different weight values on the layer
    #second layer
    eps2 = C2*abs(np.amax(w[12:16,:],axis=1)-np.amin(w[12:16,:],axis=1)) * (100**(-1.0/4)) # if the weights across the row are approximately the same, algorithm stucks even if error is >> 0 and needs to be revitalized by adding extra noise
    
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
    
    if (delta_errprev < 0.01*err): # empirical, when we need to push out network. Such step 0.01 is already too slow for particle filter 
        print "stuck, adding +10% noise"
        stucked += 1
        eps1 = np.abs(np.min(w[0:12,:],axis = 1)*0.1*stucked)
        eps2 = np.abs(np.min(w[12:16,:],axis = 1)*0.1*stucked)
    else:
        stucked = 0
    #print "*",max(eps1)
    #print "*",max(eps2)
    
    
    if j<(epochs-1): # to avoid noising at the end
        
        w[0:12,:] = np.add(w[0:12,:], np.multiply(np.random.random((12,100)),2*eps1.reshape((12,1))) - eps1.reshape((12,1)))
        w[12:16,:] = np.add(w[12:16,:], np.multiply(np.random.random((4,100)),2*eps2.reshape((4,1))) - eps2.reshape((4,1)))
    
    print 'l2_error: ', np.min(l2_error_chosen)
    

print "Output After Training:"
# select index that correspond to the smallest error and
# propagate to get result:
w0,w1 = row_to_arrays(w1[:,np.argmin(l2_error_chosen)])
l1 = nonlin(np.dot(l0,w0))
l2 = nonlin(np.dot(l1,w1))

print np.round(l2,decimals=3)