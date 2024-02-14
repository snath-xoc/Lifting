## Reconstruction for emulations
## Step 1: do regional response
## Step 2: perform grounding

import joblib
import numpy as np
import load_data
import xarray as xr
import time
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import joint

def load_estimator(dir_est, model, gam_arch='exp', weighted = False):
    
    lifted = joblib.load(dir_est+model+'/mean_response/lifted.pkl')
    
    if gam_arch=='spline': ## for keeping conventions
                                
        if weighted:
            est = joblib.load(dir_est+model+'/mean_response/gam_reglift_wgt.pkl')
            residuals = joblib.load(dir_est+model+'/mean_response/gam_residuals_reglift_wgt.pkl')
            residuals_gp = joblib.load(dir_est+model+'/mean_response/gam_residuals_lifted_wgt.pkl')
        else:
            est = joblib.load(dir_est+model+'/mean_response/gam_reglift.pkl')
            residuals = joblib.load(dir_est+model+'/mean_response/gam_residuals_reglift.pkl')
            residuals_gp = joblib.load(dir_est+model+'/mean_response/residuals_lifted.pkl')
    else:

        if weighted:
            est = joblib.load(dir_est+model+'/mean_response/gam_'+gam_arch+'reglift_wgt.pkl')
            residuals = joblib.load(dir_est+model+'/mean_response/gam_'+gam_arch+'_residuals_reglift_wgt.pkl')
            residuals_gp = joblib.load(dir_est+model+'/mean_response/'+gam_arch+'_residuals_lifted_wgt.pkl')
        else:
            est = joblib.load(dir_est+model+'/mean_response/gam_'+gam_arch+'_reglift.pkl')
            residuals = joblib.load(dir_est+model+'/mean_response/gam_'+gam_arch+'_residuals_reglift.pkl')
            residuals_gp = joblib.load(dir_est+model+'/mean_response/'+gam_arch+'_residuals_lifted.pkl')
            
            
    return est, residuals, residuals_gp, lifted

def predict_mean_regional_response(est, var, GMT):
  
    mr = est.predict(GMT) + var
    
    return mr
    
def store_netcdf(emulations, time_beg = '1850-01', time_end = '2101-01'):
        
    lon_pc, lat_pc, wgt, wgt_l, idx_l = load_data.get_meta_data()
    emu_temp = np.zeros([emulations.shape[0], emulations.shape[1], 72, 144])
    emu_temp[:,:,idx_l] = emulations

    lon = np.arange(-178.75, 180, 2.5)
    lat = np.arange(-88.75, 90, 2.5)
    times = np.arange(time_beg, time_end, np.timedelta64(1,'Y'), dtype='datetime64')

    emu_nc = xr.Dataset(
                 {"y": (["time", "emu", "lat", "lon"], np.swapaxes(emu_temp, 0, 1)),
                                     },        
                  coords={
                           "emu":(["emu"], np.arange(emulations.shape[0])),
                           "time": (["time"], times),
                           "lon": (["lon"], lon),
                           "lat": (["lat"], lat),
                                        }
    )
    
    emu_nc.lat.attrs['axis']='Y'
    emu_nc.lat.attrs['standard_name']='lat'
    emu_nc.lat.attrs['long_name']='latitude'
    emu_nc.lat.attrs['units']='degrees_north'

    emu_nc.lon.attrs['axis']='X'
    emu_nc.lon.attrs['standard_name']='lon'
    emu_nc.lon.attrs['long_name']='longitude'
    emu_nc.lon.attrs['units']='degrees_east'

    return emu_nc

def get_regional_mr(residuals, i_mon, est, GMT, lifted,
                       n_emu_mr):
    
    mean = np.mean(residuals[:,i_mon,:], axis = 0)
    cov = np.cov(residuals[:,i_mon,:], rowvar = False)
    innov = np.random.multivariate_normal(mean, cov, size = (n_emu_mr, GMT.shape[0]))


    Y_all = [predict_mean_regional_response(est[region][i_mon], 
                                            innov[:,:,int(region)], GMT) for region in lifted[0].pairs.keys()]
    
    return Y_all

def single_ground(Y_all, residuals_gp, i_mon, GMT, lifted,
                       n_emu_mr, n_emu_lift):
    
    
    monthly_emulations = np.zeros([n_emu_mr*n_emu_lift, GMT.shape[0], residuals_gp.shape[-1]])

    for mr_emu in range(n_emu_mr):

        Y_mr_emu = [Y_all[reg][mr_emu] for reg in range(len(Y_all))]

        monthly_emulations[\
                   mr_emu*n_emu_lift:mr_emu*n_emu_lift+n_emu_lift,:,:\
                  ] = lifted[i_mon].ground_over_regions(Y_mr_emu, n_samp = n_emu_lift, parallel=False)
        
    return [monthly_emulations]
        
        

def pair_ground(Y_all, residuals_gp, i_mon, GMT,
                lifted, n_emu_mr, n_emu_lift):
    
    Y_all_1, Y_all_2 = Y_all
    residuals_gp_1, residuals_gp_2 = residuals_gp
    lifted_1, lifted_2 = lifted
    
    pair_ground = joint.pair_ground(lifted_1[i_mon],lifted_2[i_mon])
    
    monthly_emulations_1 = np.zeros([n_emu_mr*n_emu_lift, GMT.shape[0], residuals_gp_1.shape[-1]])
    monthly_emulations_2 = np.zeros([n_emu_mr*n_emu_lift, GMT.shape[0], residuals_gp_1.shape[-1]])
    
    for mr_emu in range(n_emu_mr):

        Y_mr_emu_1 = [Y_all_1[reg][mr_emu] for reg in range(len(Y_all_1))]
        Y_mr_emu_2 = [Y_all_2[reg][mr_emu] for reg in range(len(Y_all_2))]

        grounded_grid_1, grounded_grid_2 = pair_ground.ground_over_regions([Y_mr_emu_1, Y_mr_emu_2],
                                                                              parallel = False, progress = False)
        monthly_emulations_1[\
                   mr_emu*n_emu_lift:mr_emu*n_emu_lift+n_emu_lift,:,:\
                  ] = grounded_grid_1
        
        monthly_emulations_2[\
                   mr_emu*n_emu_lift:mr_emu*n_emu_lift+n_emu_lift,:,:\
                  ] = grounded_grid_2
        
    return [monthly_emulations_1, monthly_emulations_2]



def loop_through_months(residuals, residuals_gp, i_mon, ests, GMT, lifted,
                       n_emu_mr, n_emu_lift):
    
    pair = False
    
    if len(lifted)>1:
        
        pair = True
    
    Y_all_list = []
    
    for i_est, est in enumerate(ests):
        
        Y_all_list.append(get_regional_mr(residuals[i_est], i_mon, est, GMT, lifted[i_est],
                       n_emu_mr)
    
                         )
    #print(len(Y_all_list),len(Y_all_list[0]), len(lifted))
    if pair:

        return pair_ground(Y_all_list, residuals_gp, i_mon, GMT,
            lifted, n_emu_mr, n_emu_lift)
    
    else:
        
        return single_ground(Y_all[0], residuals_gp[0], i_mon, GMT, lifted[0],
                       n_emu_mr, n_emu_lift)



def create_emulations(dir_est, dir_out_emu, i_mon, model, GMT, scenario, var ='tas', n_emu_mr=10, n_emu_lift=100, 
                      time_beg = '2015-01', time_end = '2101-01', **kwargs):
    
    '''
    
    Function to create emulations
    
    Input:
    ------
    
    dir_est: string
             location to estimator i.e., pattern scaling module for regional temperatures
    
    i_mon: int
           month for which to create emulations
    
    model: string
           ESM model name
           
    GMT: nd.array (time,)
         Global Mean Temperature trajectory
         
    scenario: string
              for naming purposes, scenario name
              
    var: list or string
         variables to conduct emulations over, if len>1 then joint emulation conducting via pair grounding
         
    nr_emu_mr: int
               number of mean regional response emulations to generate
               
    n_emu_lift: int
                number of spatially resolved emulations to ground to
    
    time_beg: string
              YYYY-MM format for beginning year, for purpose of creation of NETCDF4 file
              
    time_end: string
              similar to time_beg but end year
               
    Note:
    -----
    
    Stores files under dir_est+model+dir_out_emu
    
    '''        
    
    if type(dir_est)==list and type(dir_out_emu)!=list:
        
        print('Estimator directory and output directory are not the same data type')
        
        return
    
    if isinstance(dir_est, list) and isinstance(dir_out_emu, list):
        
        if len(dir_est)!=len(dir_out_emu):
            
            print('Estimator directory and output directory lists are not the same length')
            
            return
    
        for d_est, d_out_emu in zip(dir_est, dir_out_emu):
            
            if not os.path.exists(d_est+model+d_out_emu):

                os.makedirs(d_est+model+d_out_emu)
                
    else:
        
        if not os.path.exists(dir_est+model+dir_out_emu):

                os.makedirs(dir_est+model+dir_out_emu)
                
        dir_est = [dir_est]
        dir_out_emu = [dir_out_emu]
                
                
    if not isinstance(var, list):
        
        var = [var]
    
    if len(dir_est)!=len(var):
        
        print('Expected a separate directory for each variable, however len(dir_est)!=len(var)')
    
    all_ests = []
    all_residuals = []
    all_residuals_gp = []
    all_lifted = []
    
    for d_est in dir_est:
        
        est, residuals, residuals_gp, lifted = load_estimator(d_est, model, **kwargs)
    
        residuals = residuals.reshape(-1,12,residuals.shape[-1])
        residuals_gp = residuals_gp.reshape(-1,12,residuals_gp.shape[-1])
        
        all_ests.append(est)
        all_residuals.append(residuals)
        all_residuals_gp.append(residuals_gp)
        all_lifted.append(lifted)

    start_time = time.time()
    
    #monthly_emulations = Parallel(n_jobs=12, pre_dispatch = 12)(delayed(loop_through_months)(\
    #                                                                     all_residuals, all_residuals_gp, i_mon, all_ests,
    #                                                                     GMT, all_lifted, n_emu_mr, n_emu_lift)
    #                                         for i_mon in range(12)
    #                                        )
    
    monthly_emulations = loop_through_months(all_residuals, all_residuals_gp, i_mon, all_ests,
                                                                        GMT, all_lifted, n_emu_mr, n_emu_lift)
                                 
        
    
    print("--- %s seconds --- for %i emulations" % (time.time() - start_time,n_emu_mr*n_emu_lift))
    
    start_time = time.time()
    
    for i_var, v in enumerate(var):
        
        emulations = monthly_emulations[i_var]

        
        nc_file = store_netcdf(emulations.reshape(n_emu_mr*n_emu_lift, -1, all_residuals_gp[i_var].shape[-1]),
                              time_beg = time_beg, time_end = time_end)
 
        nc_file.to_netcdf(\
                          dir_est[i_var]+model+dir_out_emu[i_var]+v+'_%iemus_'%(n_emu_mr*n_emu_lift)+\
                          model+'_'+scenario+'_%imon_g025.nc'%i_mon
                         )
        
    print("stored file in --- %s seconds ---" % (time.time() - start_time))

    return 
    
    

    