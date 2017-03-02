
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

imp = 0

epochs = 25

for j in xrange(epochs):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    
    # lambda_r = 0.05 # regularization parameter
    
    l2_error = np.zeros(100)
    l1, l2 = None,None
    for i in range(100):
        w0,w1 = row_to_arrays(w[:,i])
        l1 = nonlin(np.dot(l0,w0))
        l2 = nonlin(np.dot(l1,w1))
        # calculating error with regularization
        l2_error[i] = np.sum((y - l2)**2) #+ lambda_r*np.sum(abs(w[:,i]))  # regularization either does nothing or destroys everything! The model tends then to [0.5,0.5,0.5,0.5]
    
    # Importance
    
    imp = 1.0/l2_error # lower error gets bigger weight
    
    # Resampling
    
    # approach 1: get area in ND around weighted ones
    
    # select indices of the 5% weights witj largest likelihood
    indices = np.argpartition(-imp, int(0.05*100))[:int(0.05*100)] 
    
    # choose these models
    w1 = np.take(w,indices,axis=1)
    
    # bootstrap with noise to create new sample of 100 elements located around these chosen
    
    # bootstrap - sample indices with replacement
    indices2 = np.random.choice(w1.shape[1],100)
    # take elements
    w = np.take(w1,indices2,axis=1)
    # add uniform noises to the wights (different training speed for different layers taken into account)
    
    #C1 = 0.1 # works like charm, but takes 60 iterations
    #C2 = 0.2
    C1 = 1*l2_error[indices[0]] # The closer we are to the solution, the lesser are the steps. Gives significant speed improvement!
    C2 = 2*l2_error[indices[0]] # But works a bit weird sometimes (l2 error shifts instantly to the either side, weights explode, no convergence in rare occasions)
    
    #first layer
    eps1 = C1*abs(np.max(w[0:12,:])-np.min(w[0:12,:])) * (100**(-1.0/12))
    #second layer
    eps2 = C2*abs(np.max(w[12:16,:])-np.min(w[12:16,:])) * (100**(-1.0/4))
    
    print 'l2_error[indices[0]], eps1, eps2: ', l2_error[indices[0]], ', ', eps1, ', ', eps2
    
    if j<(epochs-1): # to avoid noising at the end
        
        w[0:12,:] = np.add(w[0:12,:], (2*eps1)*np.random.random((12,100)) - eps1)
        w[12:16,:] = np.add(w[12:16,:], (2*eps2)*np.random.random((4,100)) - eps2)
    

print "Output After Training:"
# select index that correspond to the smallest error:
ind = np.argmax(imp)
# propagate to get result:
w0,w1 = row_to_arrays(w[:,ind])
l1 = nonlin(np.dot(l0,w0))
l2 = nonlin(np.dot(l1,w1))

print np.round(l2,decimals=3)