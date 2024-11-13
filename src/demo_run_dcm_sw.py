import argparse
from pydcm.models.pdcm import pdcm_fmri_priors_new
from pydcm.models.hemodynamic import *
from pydcm.io import loadmat
import numpy as np
from pydcm.spm.nlsi import spm_nlsi_GN
from pydcm.spm.integrate import spm_int_IT
import copy
import warnings
warnings.filterwarnings('ignore')

def main(args):
    
    SPM = loadmat(args.data_root+'SPM.mat')
    Y = SPM['Y']
    U = SPM['U']   
    
    if args.do_rescaling:
        scale   = np.max(Y['y']) - np.min(Y['y'])
        scale   = 4/max(scale,4)
    else:
        scale = 1
    Y['y']     = Y['y']*scale
    Y['scale'] = scale
    
    if args.Y_dt is not None:
        Y_dt = args.Y_dt
    else:
        Y_dt = 1
    vals = int(U['u'].shape[0]/Y['y'].shape[0])
    slide_len = int(args.stride * Y_dt)
    wind_len = int(args.window_len * Y_dt)
    u_wind_len = int(wind_len * vals)
    u_slide_len = int(slide_len * vals)
    
    A = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])
        
    B = np.array([[[0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]],
    
           [[0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]]])
        
    C = np.array([[0, 0, 1],
              [0, 0, 0],
              [0, 0, 0]])
    
    D = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])
        
    pE, pC, x, _ = pdcm_fmri_priors_new(A, B, C, D, {'decay':1})
    
    window = 0
    Y_actual = copy.deepcopy(Y)
    U_actual = copy.deepcopy(U)
    params_dict = {}
    pC_old = copy.deepcopy(pC)
    
    while int(window*slide_len)+wind_len <= Y_actual['y'].shape[0]:
        print('\n')
        print('window ' + str(window))
        Y['y'] = Y_actual['y'][int(window*slide_len):int(window*slide_len)+wind_len,:]
        Y['X0'] = Y_actual['X0'][int(window*slide_len):int(window*slide_len)+wind_len,:]
        U['u'] = U_actual['u'][int(window*u_slide_len):int(window*u_slide_len)+u_wind_len,:]
        M = {}
        B0      = 3
        TE      = 0.04
        nr      = Y['y'].shape[1]
        M['delays'] = np.ones((1,nr))*Y['dt']/2 
        M['TE']    = TE
        M['B0']    = B0
        M['m']     = nr
        M['n']     = 6         
        M['l']     = nr
        M['N']     = 64
        M['dt']    = U['dt']
        M['ns']    = Y['y'].shape[0]
        M['x']     = x
        M['IS']    = 'spm.integ_IT'
        
        M['f']   = 'fx_fmri_pdcm_new'
        M['g']   = 'gx_all_fmri'
        M['Tn']  = []                     
        M['Tc']  = []
        M['Tv']  = []
        M['Tm']  = []
        
        n = nr
        
        M['pE'] = pE
        M['pC'] = pC
        M['Nmax'] = args.Nmax
        
        Ep,Cp,Eh,F,L,dFdp,dFdpp = spm_nlsi_GN(M, U, Y)
        
        y, latent = spm_int_IT(Ep, M, U, nargout=2)
        
        pE = copy.deepcopy(Ep)
        if args.update_connectivity_only:
            _, pC_updated, _, _ = pdcm_fmri_priors_new(A, B, C, D, {'decay':1, 'other_neuronal_priors_cov':False, 'other_hemodynamic_priors_cov':False})
            if args.use_prev_pC:
                pC = copy.deepcopy((pC_updated!=0)*Cp)
            else:
                pC = copy.deepcopy(pC_updated)
        else:
            if args.use_prev_pC:
                pC = copy.deepcopy((pC_old!=0)*Cp)
            else:
                pC = copy.deepcopy(pC_old)
        x = latent[:, :, int(slide_len)]
        params_dict.update({'window'+str(window):[Ep, Cp]})
        
        window += 1
        
    return params_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--Nmax', type=int, help='Maximum Iterations')
    parser.add_argument('--data_root', default='pydcm/', type=str, help='Root directory of the data')
    parser.add_argument('--stride', default=1, type=int, help='Stride')
    parser.add_argument('--window_len', type=int, help='Window Length')
    parser.add_argument('--update_connectivity_only', type=bool, help='Update connectivity only or not')
    parser.add_argument('--use_prev_pC', type=bool, help='Use previous covariance matrix')
    parser.add_argument('--Y_dt', type=int, help='Sampling Frequency')
    parser.add_argument('--do_rescaling', default=True, type=bool, help='Rescaling of data')
    args = parser.parse_args()
    params_dict = main(args)