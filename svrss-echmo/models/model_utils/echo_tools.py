# Toolkit for complex numbers
# zlw@20240810
import torch
import os
import numpy as np
import scipy.io as scio

def complex_minmax(x, epsilon=1e-5, feature_range=(0, 1)):
    # pulse-wise min-max normalization
    # x has shape of BxCxPxR
    
    min_R = x.real.min(dim=3, keepdim=True).values.detach()
    max_R = x.real.max(dim=3, keepdim=True).values.detach()
    min_I = x.imag.min(dim=3, keepdim=True).values.detach()
    max_I = x.imag.max(dim=3, keepdim=True).values.detach()
    
    range_R = max_R - min_R
    range_I = max_I - min_I
    
    range_R = torch.where(range_R < epsilon, torch.ones_like(range_R), range_R)
    range_I = torch.where(range_I < epsilon, torch.ones_like(range_I), range_I)
    

    x_R = (x.real - min_R) / range_R
    x_I = (x.imag - min_I) / range_I
    

    min_target, max_target = feature_range
    x_R = x_R * (max_target - min_target) + min_target
    x_I = x_I * (max_target - min_target) + min_target
    
    return x_R + 1j * x_I

def complex_std(x, epsilon=1e-5):
    # pulse-wise standarization
    # x has shape of BxCxPxR
    mu_R = x.real.mean(dim=3, keepdim=True).detach()
    mu_I = x.imag.mean(dim=3, keepdim=True).detach()
    x_R = x.real - mu_R
    x_I = x.imag - mu_I
    sigma_R = torch.sqrt(torch.var(x.real, dim=3, keepdim=True, unbiased=False) + epsilon)
    sigma_I = torch.sqrt(torch.var(x.imag, dim=3, keepdim=True, unbiased=False) + epsilon)
    x_R /= sigma_R
    x_I /= sigma_I
    return x_R + 1j * x_I

def complex_double2float(x):
    x_r = x.real.float()
    x_i = x.imag.float()
    return x_r + 1j * x_i

def filter_coef_loader(path, coef_type=None):
    """
    coef_type:
        ori_mf(rect_mf): reference signal w/o window function;
        tay_mf: reference signal w/ taylor window;
        che_mf: reference signal w/ cheb window;
        ham_mf: reference signal w/ hamming window;
        han_mf: reference signal w/ hanning window;
        black_mf: reference signal w/ blackman window;
        cos_mf: reference signal w/ cosine window;
        # rect_mf: reference signal w/ rectangular window;
    """

    # coef_path = os.join(path, coef_type)
    # coef = np.load(path)
    # coef_real = torch.randn([1,32], dtype=torch.float32)
    # coef_imag = torch.randn([1,32], dtype=torch.float32)
    # coef = torch.complex(coef_real, coef_imag)
    coef = scio.loadmat(os.path.join(path, 's_tx.mat'))['s_tx']
    coef = torch.tensor(coef) # [1, T]

  
    if coef_type == 'ori_mf':
        coef = complex_double2float(torch.conj(coef)) # element-wise production
        # return complex_double2float(torch.fliplr(torch.conj(coef*win)))
        zero = torch.complex(torch.zeros([1, 1], dtype=torch.float32), torch.zeros([1, 1], dtype=torch.float32))
        coef = torch.cat((zero, coef), dim=1) 
        return coef
    elif coef_type == 'tay_mf':
        win = scio.loadmat(os.path.join(path, 'w_taylor.mat'))['w_taylor']
    elif coef_type == 'che_mf':
        win = scio.loadmat(os.path.join(path, 'w_chebyshev.mat'))['w_chebyshev']
    elif coef_type == 'ham_mf':
        win = scio.loadmat(os.path.join(path, 'w_hamming.mat'))['w_hamming']
    elif coef_type == 'han_mf':
        win = scio.loadmat(os.path.join(path, 'w_hanning.mat'))['w_hanning']
    elif coef_type == 'black_mf':
        win = scio.loadmat(os.path.join(path, 'w_blackman.mat'))['w_blackman']
    elif coef_type == 'cos_mf':
        win = scio.loadmat(os.path.join(path, 'w_cosine.mat'))['w_cosine']
    # elif coef_type == 'rect_mf':
    #     win = scio.loadmat(os.path.join(path, 'w_rectangular.mat'))['w_rectangular']

    win = torch.permute(torch.tensor(win), [1, 0]) # [1, T]
    coef = complex_double2float(torch.conj(coef * win)) # element-wise production
        # return complex_double2float(torch.fliplr(torch.conj(coef*win)))
    zero = torch.complex(torch.zeros([1, 1], dtype=torch.float32), torch.zeros([1, 1], dtype=torch.float32))
    coef = torch.cat((zero, coef), dim=1) 
    return coef

    # if coef_type == 'tay_mf':
    #     win = scio.loadmat(os.path.join(path, 'w_taylor.mat'))['w_taylor']
    #     win = torch.permute(torch.tensor(win), [1, 0]) # [1, T]
    #     coef = complex_double2float(torch.conj(coef * win)) # element-wise production
    #     # return complex_double2float(torch.fliplr(torch.conj(coef*win)))
    # zero = torch.complex(torch.zeros([1, 1], dtype=torch.float32), torch.zeros([1, 1], dtype=torch.float32))
    # coef = torch.cat((zero, coef), dim=1) # [1, 1 + T]
    # return coef

def ref_loader(path):
    ref = scio.loadmat(os.path.join(path, 's_tx.mat'))['s_tx']
    ref = torch.tensor(ref) # [1, T]

    return ref

  

if __name__ == '__main__':
    coef = filter_coef_loader('/data/PeCoLab/DataSets/NuaaSim/doc_code/echo_simulator_py', 'tay_mf')
    print(coef)

