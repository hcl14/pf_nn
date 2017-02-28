import numpy as np

from matplotlib import pyplot as plt

import scipy.stats

from scipy.interpolate import interp1d

# This realization is very inefficient and is needed only to demonstrate approach

# Functions for sampling from 1-D ECDF: -----------------------------------------------------------------------

def indicator_kernel_ECDF(x,estimation_points,estimation_weights,weightsum): # stepwise ECDF
    return (1.0/weightsum)*np.sum(estimation_weights*(estimation_points<=x))


def ECDF_points(q,weights): # stepwise CDF ready to interpolate
    
    qe = q.copy()
    
    
    # here we are evaluating CDF on the estimation_points. We could do that in the function indicator_kernel_ECDF already to avoid re-computing many times, but let's have the general approach here, maybe will use some another grid there in the future
    
    pe = indicator_kernel_ECDF(qe[0],qe,weights,np.sum(weights))
    
    for i in range(1,qe.size):
        pe = np.append(pe,indicator_kernel_ECDF(qe[i],qe,weights,np.sum(weights))) # somewhy vectorizing function gives wrong results!
        
    # Fixind ends to avoid linear continuation in the wrong direction
    
    # 1 on the right end at the distance = 0.5*dist between last two points
    
    dist = abs(qe[qe.size-1]-qe[qe.size-2])/2.0
    
    pe = np.append(pe,1)
    qe = np.append(qe,qe[qe.size-1]+dist)
    
    #pe = np.append(pe,1)
    #qe = np.append(qe,qe[qe.size-1]+dist)
    
    # 0 on the left end 
    
    dist = abs(qe[0]-qe[1])/2.0
    
    pe = np.hstack((0,pe))
    qe = np.hstack((qe[0]-dist,qe))
    
    pe = np.hstack((0,pe))
    qe = np.hstack((qe[0]-dist,qe))

    return qe, pe


def ECDF_interpolated(q,weights,num=200): # interpolated ECDF. Very inefficient way to go!
    # updated points
    qe,pe = ECDF_points(q,weights)
    
    # linear interpolation
    f = interp1d(qe, pe,kind='linear')
    
    #vector version
    f_v = np.vectorize(f)
    
    # values from interpolated function calculated on the grid
    f_arg = np.linspace(qe[0], qe[-1], num)
    f_val = f_v(f_arg)
    
    return f, f_v, f_arg, f_val

def draw_one_sample(f_arg,f_val): # draw one sample from interpolated 1-D marginal CDF given by grid (f_arg,f_val)
    q = np.random.uniform(0,1)
    closest_f_value_idx = find_closest(f_val, q)
    return f_arg[closest_f_value_idx]
    
def find_closest(A, target): # Finding the nearest value and return the index of array in Python
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


# -----------------------------------------------------------------------


# Function for sampling from gaussian copula --------------------------------------

# http://stats.stackexchange.com/questions/7515/what-are-some-techniques-for-sampling-two-correlated-random-variables

def sample_copula1(m_arg,m_val): # m is a matrix of ECDFs' values
    
    corrmatrix = np.corrcoef(m_val) # correlation matrix
    
    # print 'corrmatrix', corrmatrix
    
    sample_vect = np.random.multivariate_normal(np.zeros(m_val.shape[0]),corrmatrix) # sample from multivariate normal CDF with this correlation matrix
    
    scipy.stats.norm.cdf(sample_vect) # correlated normal CDF values, distributed on [0,1] now
    
    samples = np.zeros(m_val.shape[0])
    
    # now sample from marginal ECDFs
    for i in range(m_val.shape[0]):
           samples[i] = draw_one_sample(np.ravel(m_arg[i, :]), np.ravel(m_val[i, :]))
    
    return samples

# -----------------------------------------------------------------------





if __name__ == "__main__":
    
    weights = np.array([0.1,0.45,0.3,0.3,0.7,0.4,0.4,0.3]) # points weights. They are the same for two samples, as in our NN we will have these weights for different Xi vectors (a sample from unknown multivariate distribution). In our 2D example these will be 2-element rows of a matrix (q,q1). So, as long as we're building marginals across two axes, we take q and q1 as these marginals, and each coordinate is taken with the corresponding weight 
    #weights = np.array([1,1,1,1,1,1,1])
    
    # evaluate first marginal cdf 
    q = np.sort(np.array([1,5,8,12,9,4,15,0.5]))
    
   
    f, f_v, f_arg, f_val = ECDF_interpolated(q,weights)
    
    # draw sample
    print draw_one_sample(f_arg,f_val)
    
    # plot 
    
    
    #plt.step(qe, pe, lw=2, label='Empirical CDF')
    
    plt.plot(f_arg, f_val, '.', label='Empirical interpolated marginal CDF')
    
    plt.show()

    #create second marginal 
    
    q1 = np.sort(np.array([2,15,18,1,29,4,20,7]))
    
    
    
    f1, f_v1, f_arg1, f_val1 = ECDF_interpolated(q1,weights)
    
    # attempt to do copula sampling, preparing input matrices with grid values
    
    m_arg = np.matrix((f_arg,f_arg1)) 
    m_val = np.matrix((f_val,f_val1))
    
    # show sampled element
    print sample_copula1(m_arg,m_val)