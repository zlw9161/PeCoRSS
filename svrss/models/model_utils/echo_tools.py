# Toolkit for complex numbers
# zlw@20240810
import torch
import os
import numpy as np
import scipy.io as scio

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
        ori_mf: reference signal w/o window function;
        che_mf: reference signal w/ cheb window;
        ham_mf: reference signal w/ hamming window;
        han_mf: reference signal w/ hanning window;
        tay_mf:
    """

    # coef_path = os.join(path, coef_type)
    # coef = np.load(path)
    # coef_real = torch.randn([1,32], dtype=torch.float32)
    # coef_imag = torch.randn([1,32], dtype=torch.float32)
    # coef = torch.complex(coef_real, coef_imag)
    coef = scio.loadmat(os.path.join(path, 's_tx.mat'))['s_tx']
    coef = torch.tensor(coef) # [1, T]

    if coef_type == 'tay_mf':
        win = scio.loadmat(os.path.join(path, 'w_taylor.mat'))['w_taylor']
        win = torch.permute(torch.tensor(win), [1, 0]) # [1, T]
        coef = complex_double2float(torch.conj(coef * win)) # element-wise production
        # return complex_double2float(torch.fliplr(torch.conj(coef*win)))
    zero = torch.complex(torch.zeros([1, 1], dtype=torch.float32), torch.zeros([1, 1], dtype=torch.float32))
    coef = torch.cat((zero, coef), dim=1) # [1, 1 + T]
    return coef

if __name__ == '__main__':
    coef = filter_coef_loader('/media/liwenzhang/Data/DataSets/NuaaSim/doc_code/echo_simulator_py', 'tay_mf')
    print(coef)

