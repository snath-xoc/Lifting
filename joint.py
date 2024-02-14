## Pair grounding given two lifted objects

import numpy as np
from scipy import stats
from sklearn import covariance as skcovariance
import time
from tqdm import tqdm

class pair_ground(object):
    
    '''
    Pair lift two lifted objects by sampling in the same neighbourhood and applying copula jitters
    
    Input
    -----
    
    Lifted object 1 (considered the pointer to which neghbourhood to sample from
    
    Lifted object 2 (dependent variable on object 1)
    
    References
    ----------
    
    For copula jitters implementation: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018WR024446
    
    '''
    
    def __init__(self, lift1, lift2):
            
            self.lift1 = lift1
            self.lift2 = lift2

    def cov(self, a, b):

        return ((a * b).sum() - a.sum() * b.sum() / a.size) / (a.size - 1)
        
    
    def calc_conditional_cov(self, d_sampled_raw, labels, **kwargs):
        
        d_prop = {}
        start_time = time.time()
        for reg in d_sampled_raw.keys():
            
            d_prop[reg] = []
            
            if len(self.lift2.d[reg])!=self.lift2.max_iter[reg]+2:

                        self.lift2.compress_wavelet_coeff(reg, **kwargs)

            for counter,label in enumerate(labels):
                
                d_ind = d_sampled_raw[reg][counter]
                d_dep = self.lift2.d[reg][-1][label,:]
                
                mean = np.mean(d_dep, axis=0)

                sigma11 = skcovariance.EmpiricalCovariance().fit(d_ind[:,1:]).precision_
                sigma22 = np.cov(\
                                 d_dep[:,1:], rowvar = False)
                
                sigma21 = []
                for i_x in np.arange(1,d_dep.shape[1]):
                    
                    sigma21.append(np.array([self.cov(\
                                                     d_dep[:,i_x], d_ind[:,i_y])
                                            for i_y in np.arange(1,d_ind.shape[1])])
                                  )
                
                sigma21 = np.stack(sigma21)
                sigma12 = sigma21.T
                try:
                    cov = sigma22 - (sigma21@sigma11@sigma12)

                except:
                    sigma11_temp = np.ones_like(sigma22)
                    sigma11_temp[0,1] = sigma11
                    sigma11_temp[1,0] = sigma11
                    cov = sigma22 - (sigma21@sigma11_temp@sigma12)
        
                d_prop[reg].append([mean,cov])
                
        print('Time taken to calculate conditional covariance matrix ----%s secs----'%(time.time()-start_time))
                
        return d_prop

    def ground_over_regions(self, Y, n_samp = 100, progress = False, parallel = True, **kwargs):
        
        grounded_grid_1, D_sampled_raw, labels = self.lift1.ground_over_regions(Y[0], n_samp = n_samp, progress = progress, 
                                       parallel = parallel, output_dsampled=True, compute_neighbourhood = True, **kwargs)
        
        d_prop = self.calc_conditional_cov(D_sampled_raw, labels)

        try:
            grounded_grid_2 = self.lift2.ground_over_regions(Y[1], n_samp = n_samp, progress = progress, 
                                       parallel = parallel, d_prop = d_prop, labels_ind = np.empty(len(labels)), output_dsampled = False, 
                                                         compute_neighbourhood = False, **kwargs)
        except:
            print('Exited')
            return grounded_grid_1, d_prop
        
       
        return grounded_grid_1, grounded_grid_2
        
