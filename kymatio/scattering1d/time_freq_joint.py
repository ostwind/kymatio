import math
import numbers
import numpy as np
import torch

from .backend import (fft1d_c2c, ifft1d_c2c, modulus_complex, pad, real,
    subsample_fourier, unpad)
from .filter_bank import (calibrate_scattering_filters,
    scattering_filter_factory)
from .utils import cast_phi, cast_psi, compute_border_indices, compute_padding

#TODO: 
#   custom scattering_filter_factory
#   do tests   

class JointScattering(object):
    """The time frequency joint scattering transform

    """
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
                 oversampling=0, vectorize=True):
        super(JointScattering, self).__init__()
        # Store the parameters
        self.J = J
        self.shape = shape
        self.Q = Q

        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.vectorize = vectorize

        # Build internal values
        self.build()

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """

        # Set these default values for now. In the future, we'll want some
        # flexibility for these, but for now, let's keep them fixed.
        self.r_psi = math.sqrt(0.5)
        self.sigma0 = 0.1
        self.alpha = 5.
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3
        self.normalize = 'l1'

        # check the shape
        if isinstance(self.shape, numbers.Integral):
            self.T = self.shape
        elif isinstance(self.shape, tuple):
            self.T = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")

        # Compute the minimum support to pad (ideally)
        min_to_pad = compute_minimum_support_to_pad(
            self.T, self.J, self.Q, r_psi=self.r_psi, sigma0=self.sigma0,
            alpha=self.alpha, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude,
            normalize=self.normalize)
        # to avoid padding more than T - 1 on the left and on the right,
        # since otherwise torch sends nans
        J_max_support = int(np.floor(np.log2(3 * self.T - 2)))
        self.J_pad = min(int(np.ceil(np.log2(self.T + 2 * min_to_pad))),
                         J_max_support)
        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.T)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.J, self.pad_left, self.pad_left + self.T)
        # Finally, precompute the filters
        # need to modify scattering_filter_factory to output psi2_time, psi2_freq
        phi_f, psi1_f, psi2_time, psi2_freq, _ = scattering_filter_factory_joint(
            self.J_pad, self.J, self.Q, normalize=self.normalize,
            to_torch=True, criterion_amplitude=self.criterion_amplitude,
            r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha,
            P_max=self.P_max, eps=self.eps)
        self.psi1_f = psi1_f
        self.psi2_time = psi2_time
        self.psi2_freq = psi2_freq
        self.phi_f = phi_f
        self._type(torch.FloatTensor)

    def _type(self, target_type):
        """Change the datatype of the filters

        This function is used internally to convert the filters. It does not
        need to be called explicitly.

        Parameters
        ----------
        target_type : type
            The desired type of the filters, typically `torch.FloatTensor`
            or `torch.cuda.FloatTensor`.
        """
        cast_psi(self.psi1_f, target_type)
        cast_psi(self.psi2_f, target_type)
        cast_phi(self.phi_f, target_type)
        return self

    def cpu(self):
        """Move to the CPU

        This function prepares the object to accept input Tensors on the CPU.
        """
        return self._type(torch.FloatTensor)

    def cuda(self):
        """Move to the GPU

        This function prepares the object to accept input Tensors on the GPU.
        """
        return self._type(torch.cuda.FloatTensor)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return JointScattering.compute_meta_scattering(
            self.J, self.Q, max_order=self.max_order)

    def output_size(self, detail=False):
        """Get size of the scattering transform

        Calls the static method `precompute_size_scattering()` with the
        parameters of the transform object.

        Parameters
        ----------
        detail : boolean, optional
            Specifies whether to provide a detailed size (number of coefficient
            per order) or an aggregate size (total number of coefficients).

        Returns
        ------
        size : int or tuple
            See the documentation for `precompute_size_scattering()`.
        """

        return JointScattering.precompute_size_scattering(
            self.J, self.Q, max_order=self.max_order, detail=detail)

    def forward(self, x):
        """Apply the scattering transform

        Given an input Tensor of size `(B, T0)`, where `B` is the batch
        size and `T0` is the length of the individual signals, this function
        computes its scattering transform. If the `vectorize` flag is set to
        `True`, the output is in the form of a Tensor or size `(B, C, T1)`,
        where `T1` is the signal length after subsampling to the scale `2**J`
        (with the appropriate oversampling factor to reduce aliasing), and
        `C` is the number of scattering coefficients.  If `vectorize` is set
        `False`, however, the output is a dictionary containing `C` keys, each
        a tuple whose length corresponds to the scattering order and whose
        elements are the sequence of filter indices used.

        Furthermore, if the `average` flag is set to `False`, these outputs
        are not averaged, but are simply the wavelet modulus coefficients of
        the filters.

        Parameters
        ----------
        x : tensor
            An input Tensor of size `(B, T0)`.

        Returns
        -------
        S : tensor or dictionary
            If the `vectorize` flag is `True`, the output is a Tensor
            containing the scattering coefficients, while if `vectorize`
            is `False`, it is a dictionary indexed by tuples of filter indices.
        """
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            if not(self.average):
                raise ValueError(
                    'Options average=False and vectorize=True are ' +
                    'mutually incompatible. Please set vectorize to False.')
            size_scattering = self.precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0
        S = joint_scattering(x, self.psi1_f, self.psi2_f, self.phi_f,
                       self.J, max_order=self.max_order, average=self.average,
                       pad_left=self.pad_left, pad_right=self.pad_right,
                       ind_start=self.ind_start, ind_end=self.ind_end,
                       oversampling=self.oversampling,
                       vectorize=self.vectorize,
                       size_scattering=size_scattering)

        if self.vectorize:
            scattering_shape = S.shape[-2:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            for k, v in S.items():
                scattering_shape = v.shape[-2:]
                S[k] = v.reshape(batch_shape + scattering_shape)

        return S

    def __call__(self, x):
        return self.forward(x)

    @staticmethod
    def compute_meta_scattering(J, Q, max_order=2):
        """Get metadata on the transform.

        This information specifies the content of each scattering coefficient,
        which order, which frequencies, which filters were used, and so on.

        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform.
            In other words, the maximum scale is given by `2**J`.
        Q : int >= 1
            The number of first-order wavelets per octave.
            Second-order wavelets are fixed to one wavelet per octave.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`. Defaults to `2`.

        Returns
        -------
        meta : dictionary
            A dictionary with the following keys:

            - `'order`' : tensor
                A Tensor of length `C`, the total number of scattering
                coefficients, specifying the scattering order.
            - `'xi'` : tensor
                A Tensor of size `(C, max_order)`, specifying the center
                frequency of the filter used at each order (padded with NaNs).
            - `'sigma'` : tensor
                A Tensor of size `(C, max_order)`, specifying the frequency
                bandwidth of the filter used at each order (padded with NaNs).
            - `'j'` : tensor
                A Tensor of size `(C, max_order)`, specifying the dyadic scale
                of the filter used at each order (padded with NaNs).
            - `'n'` : tensor
                A Tensor of size `(C, max_order)`, specifying the indices of
                the filters used at each order (padded with NaNs).
            - `'key'` : list
                The tuples indexing the corresponding scattering coefficient
                in the non-vectorized output.
        """
        sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = \
            calibrate_scattering_filters(J, Q)

        meta = {}

        meta['order'] = [[], [], []]
        meta['xi'] = [[], [], []]
        meta['sigma'] = [[], [], []]
        meta['j'] = [[], [], []]
        meta['n'] = [[], [], []]
        meta['key'] = [[], [], []]

        meta['order'][0].append(0)
        meta['xi'][0].append(())
        meta['sigma'][0].append(())
        meta['j'][0].append(())
        meta['n'][0].append(())
        meta['key'][0].append(())

        for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
            meta['order'][1].append(1)
            meta['xi'][1].append((xi1,))
            meta['sigma'][1].append((sigma1,))
            meta['j'][1].append((j1,))
            meta['n'][1].append((n1,))
            meta['key'][1].append((n1,))

            if max_order < 2:
                continue

            for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
                if j2 > j1:
                    meta['order'][2].append(2)
                    meta['xi'][2].append((xi1, xi2))
                    meta['sigma'][2].append((sigma1, sigma2))
                    meta['j'][2].append((j1, j2))
                    meta['n'][2].append((n1, n2))
                    meta['key'][2].append((n1, n2))

        for field, value in meta.items():
            meta[field] = value[0] + value[1] + value[2]

        pad_fields = ['xi', 'sigma', 'j', 'n']
        pad_len = max_order

        for field in pad_fields:
            meta[field] = [x+(math.nan,)*(pad_len-len(x)) for x in meta[field]]

        array_fields = ['order', 'xi', 'sigma', 'j', 'n']

        for field in array_fields:
            meta[field] = torch.from_numpy(np.array(meta[field]))

        return meta

    @staticmethod
    def precompute_size_scattering(J, Q, max_order=2, detail=False):
        """Get size of the scattering transform

        The number of scattering coefficients depends on the filter
        configuration and so can be calculated using a few of the scattering
        transform parameters.

        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform.
            In other words, the maximum scale is given by `2**J`.
        Q : int >= 1
            The number of first-order wavelets per octave.
            Second-order wavelets are fixed to one wavelet per octave.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`. Defaults to `2`.
        detail : boolean, optional
            Specifies whether to provide a detailed size (number of coefficient
            per order) or an aggregate size (total number of coefficients).

        Returns
        -------
        size : int or tuple
            If `detail` is `False`, returns the number of coefficients as an
            integer. If `True`, returns a tuple of size `max_order` containing
            the number of coefficients in each order.
        """
        sigma_low, xi1, sigma1, j1, xi2, sigma2, j2 = \
            calibrate_scattering_filters(J, Q)

        size_order0 = 1
        size_order1 = len(xi1)
        size_order2 = 0
        for n1 in range(len(xi1)):
            for n2 in range(len(xi2)):
                if j2[n2] > j1[n1]:
                    size_order2 += 1
        if detail:
            if max_order == 2:
                return size_order0, size_order1, size_order2
            else:
                return size_order0, size_order1
        else:
            if max_order == 2:
                return size_order0 + size_order1 + size_order2
            else:
                return size_order0 + size_order1

def scattering_filter_factory_joint():
    '''
    build filter banks with two wavelet dictionaries for layer 2: psi2_time, psi2_freq
    '''
    pass 

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

def compute_minimum_support_to_pad(T, J, Q, criterion_amplitude=1e-3,
                                   normalize='l1', r_psi=math.sqrt(0.5),
                                   sigma0=1e-1, alpha=5., P_max=5, eps=1e-7):
    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    T : int
        temporal size of the input signal
    J : int
        scale of the scattering
    Q : int
        number of wavelets per octave
    normalize : string, optional
        normalization type for the wavelets.
        Only `'l2'` or `'l1'` normalizations are supported.
        Defaults to `'l1'`
    criterion_amplitude: float `>0` and `<1`, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger criterion_amplitude, the smaller the padding size is.
        Defaults to `1e-3`
    r_psi : float, optional
        Should be `>0` and `<1`. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent
        wavelets).
        Defaults to `sqrt(0.5)`.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering,
        it is equal to :math:`\\frac{\\sigma_0}{2^J}`.
        Defaults to `1e-1`.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger the alpha, the more conservative the value of maximal
        subsampling is.
        Defaults to `5`.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic.
        `P_max = 5` is more than enough for double precision.
        Defaults to `5`.
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to `1e-7`.

    Returns
    -------
    min_to_pad: int
        minimal value to pad the signal on one size to avoid any
        boundary error.
    """
    J_tentative = int(np.ceil(np.log2(T)))
    _, _, _, t_max_phi = scattering_filter_factory(
        J_tentative, J, Q, normalize=normalize, to_torch=False,
        max_subsampling=0, criterion_amplitude=criterion_amplitude,
        r_psi=r_psi, sigma0=sigma0, alpha=alpha, P_max=P_max, eps=eps)
    min_to_pad = 3 * t_max_phi
    return min_to_pad