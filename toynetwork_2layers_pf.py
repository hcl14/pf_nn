
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

np.random.seed(1) # this random seed corresponds to the correct convergence, otherwise the model is very likely to end in local minima

# Sampling: initialize 100 different weight vectors from continuous uniform distribution (100 models)

# randomly initialize our weights with mean 0
#w0 = 2*np.random.random((3,4)) - 1
#w1 = 2*np.random.random((4,1)) - 1

w = 2*np.random.random((16,100)) - 1 # we store w0 and w1 as single column to iterate easily

imp = 0

for j in xrange(40):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    
    
    l1_error = np.zeros(100)
    l1, l2 = None,None
    for i in range(100):
        w0,w1 = row_to_arrays(w[:,i])
        l1 = nonlin(np.dot(l0,w0))
        l2 = nonlin(np.dot(l1,w1))
        # calculating error
        l1_error[i] = np.sum(abs(y - l2))
    
    # Importance
    
    imp = 1.0/l1_error # lower error gets bigger weight
    
    # Resampling
    
    # approach 1: get area in ND around weighted ones
    
    # select indices of the largest 5% weights
    indices = np.argpartition(-imp, int(0.05*100))[:int(0.05*100)] 
    
    # choose these models
    w1 = np.take(w,indices,axis=1)
    
    # bootstrap with noise to create new sample of 100 elements located around these chosen
    
    # bootstrap - sample indices with replacement
    indices2 = np.random.choice(w1.shape[1],100)
    # take elements
    w = np.take(w1,indices2,axis=1)
    # add uniform noise
    eps = 1 #abs(np.max(w)-np.min(w))/4
    
    print 'l1_error[indices[0]], eps: ', l1_error[indices[0]], ', ', eps
    
    w = np.add(w, (2*eps)*np.random.random((16,100)) - eps)
    

print "Output After Training:"
# select index that correspond to the smallest error:
ind = np.argmax(imp)
# propagate to get result:
w0,w1 = row_to_arrays(w[:,ind])
l1 = nonlin(np.dot(l0,w0))
l2 = nonlin(np.dot(l1,w1))

print np.round(l2,decimals=3)