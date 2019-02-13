import math
import numbers
import numpy as np
import torch

from .backend import (fft1d_c2c, ifft1d_c2c, modulus_complex, pad, real,
    subsample_fourier, unpad)
from .filter_bank import (calibrate_scattering_filters,
    scattering_filter_factory)
from .utils import cast_phi, cast_psi, compute_border_indices, compute_padding

def joint_scattering( x, psi1, psi2_freq, psi2_time, phi, J, pad_left=0, pad_right=0,
               ind_start=None, ind_end=None, oversampling=0,
               max_order=2, average=True, size_scattering=(0, 0, 0), vectorize=False):
    """
    Main function implementing the time frequency joint time scattering transform
    psi2_freq : dictionary
        a dictionary of filters (in the Fourier domain), that is applied after psi2_time in layer 2 of
        the network. A better description is needed. 
    psi2_time : dictionary 


    """

    # S is simply a dictionary if we do not perform the averaging...
    if vectorize:
        batch_size = x.shape[0]
        kJ = max(J - oversampling, 0)
        temporal_size = ind_end[kJ] - ind_start[kJ]
        S = x.new(batch_size, sum(size_scattering), temporal_size).fill_(0.)
    else:
        S = {}

    # pad to a dyadic size and make it complex
    U0 = pad(x, pad_left=pad_left, pad_right=pad_right, to_complex=True)
    # compute the Fourier transform
    U0_hat = fft1d_c2c(U0)
    if vectorize:
        # initialize the cursor
        cc = [0] + list(size_scattering[:-1])  # current coordinate
        cc[1] = cc[0] + cc[1]
        if max_order == 2:
            cc[2] = cc[1] + cc[2]
    # Get S0
    k0 = max(J - oversampling, 0)
    if average:
        S0_J_hat = subsample_fourier(U0_hat * phi[0], 2**k0)
        S0_J = unpad(real(ifft1d_c2c(S0_J_hat)),
                     ind_start[k0], ind_end[k0])
    else:
        S0_J = x
    if vectorize:
        S[:, cc[0], :] = S0_J.squeeze(dim=1)
        cc[0] += 1
    else:
        S[()] = S0_J

    # build (j_2_freq X theta_2^freq) for innermost loop of layer 2
    psi2_freq_theta2_product = [ (n2, theta2) for n2 in range(len(psi2_freq)) for theta2 in [-1, 1] ] 

    # First order:
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)
        assert psi1[n1]['xi'] < 0.5 / (2**k1)
        U1_hat = subsample_fourier(U0_hat * psi1[n1][0], 2**k1)
        # Take the modulus
        U1 = modulus_complex(ifft1d_c2c(U1_hat))
        if average or max_order > 1:
            U1_hat = fft1d_c2c(U1)
        if average:
            # Convolve with phi_J
            k1_J = max(J - k1 - oversampling, 0)
            S1_J_hat = subsample_fourier(U1_hat * phi[k1], 2**k1_J)
            S1_J = unpad(real(ifft1d_c2c(S1_J_hat)),
                         ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            # just take the real value and unpad
            S1_J = unpad(real(U1), ind_start[k1], ind_end[k1])
        if vectorize:
            S[:, cc[1], :] = S1_J.squeeze(dim=1)
            cc[1] += 1
        else:
            S[(n1,)] = S1_J
        
        if max_order == 2:    
            # 2nd order
            # psi_j2^time and psi_j2^frequency are different, assume both are passed into this func
            for n2 in range(len(psi2_time)):
                j2_time = psi2_time[n2]['j']

                # preallocate array to collect time convolved coefficients
                y_time = np.zeros(  (temporal_size ,  len(psi1)) )

                for n1 in range(len(psi1)):
                    # same subsampling k2 as in scattering1D (?)
                    k2_time = max(j2_time - k1 - oversampling, 0)
                    
                    # y_2^time x(t, j_1) = (U_1 x[j_1] * psi_j2^time)(t)
                    # assuming subsampling factor = 2**k2_time
                    # not sure how to access correct wavelet in psi2_time[n2], using k1
                    time_fourier_product = subsample_fourier( U1 * psi2_time[n2][k1], 2**k2_time )
                    y_time[ :, n1 ] = ifft1d_c2c(time_fourier_product)
                    # GOAL 1 attempt

                    for (n2, theta2) in psi2_freq_theta2_product:
                        j2_freq = psi2_freq[n2]['j']
                        k2_freq = max(j2_freq - k1 - oversampling, 0)
                    
                        #y_2^freq x(t, j1) = ( y_2^time * psi_j2^freq )(t, j1)
                        freq_fourier_product = subsample_fourier( y_time[:, n1] * psi2_freq[n1][k1], 2**k2_freq )    
                        y_freq = ifft1d_c2c(freq_fourier_product)
                        # GOAL 2 attempt

                        # U_2^freq(t, j_1) = | y_2^freq |(t, j_1)
                        U2_freq = modulus_complex(y_freq)

                        # S2(t, j, j2^time, j2^freq, theta_2^freq) = ( U2^freq * phi )(t, j1)
                        # how to pick low pass filter?
                        joint_fourier_product = subsample_fourier( U2_freq * phi[k1 + k2_freq] )
                        S2 = ifft1d_c2c( joint_fourier_product ) 
                        # GOAL 3 attempt

                        # trying to store S2 into S
                        # if correct, go back to define S this way
                        S[n1, n2, j2_time, j2_freq, theta2] = S2

                #free time-convolved intermediate coefficients 
                del y_time
    return S
