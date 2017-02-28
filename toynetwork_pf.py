import numpy as np


# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]])#.T 

# Sampling: initialize 100 different weight vectors from continuous uniform distribution (100 models)

# initialize weights randomly on [-1,1] with mean 0
w = 2*np.random.random((3,100)) - 1

imp = 0


for iter in xrange(10):
    # forward pass for all these weights samples
    l0 = X
    l1 = np.zeros((X.shape[0],100))
    
    l1_error = np.zeros(100)
    for i in range(100):
        l1[:,i] = nonlin(np.dot(l0,w[:,i]))
        # calculating error
        l1_error[i] = np.sum(abs(y - l1[:,i]))
    
    # Importance
    
    imp = 1.0/l1_error # lower error gets bigger weight
    
    # Resampling
    
    # approach 1: get area in ND around weighted ones
    
    # select indices of the largest 5% weights
    indices = np.argpartition(-imp, int(0.05*100))[:int(0.05*100)] # http://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
    
    # choose these models
    w1 = np.take(w,indices,axis=1)
    
    # bootstrap with noise to create new sample of 100 elements located around these chosen
    
    # bootstrap - sample indices with replacement
    indices2 = np.random.choice(w1.shape[1],100)
    # take elements
    w = np.take(w1,indices2,axis=1)
    # add uniform noise
    eps = abs(np.max(w)-np.min(w))/4
    
    #print 'l1[99]: ', l1[:,99]
    print 'l1_error[indices[0]], eps:  ', l1_error[indices[0]], ',', eps
    
    w = np.add(w, (2*eps)*np.random.random((3,100)) - eps)
    
print 'output after training:'

# select index that correspond to the smallest error:
ind = np.argmax(imp)
# select result:
l1_ind = l1[:,ind]
# select weights:
# w_ind = 

print np.round(l1_ind,decimals=3)
    

    