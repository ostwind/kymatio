# % JOINT_TF_WAVELET_FACTORY_1D Create joint time-frequency wavelet cascade
# %
# % Usage
# %    [Wop, time_filters, freq_filters] = joint_tf_wavelet_factory_1d( ...
# %        N, time_filt_opt, freq_filt_opt, scat_opt)
# %
# % Input
# %     N (int): The size of the signals to be transformed.
# %     time_filt_opt (struct): The filter options for the time domain, same as
# %        FILTER_BANK.
# %     freq_filt_opt (struct): The filter options for the time domain, same as
# %        FILTER BANK. These should be constructed in the same way as those used
# %        when defining wavelet operators for SCAT_FREQ.
# %     scat_opt (struct): The options for the scattering transform, same as
# %        WAVELET_LAYER_1D and JOINT_TF_WAVELET_LAYER_1D.
# %
# % Output
# %    Wop (cell): A cell array of wavelet layer transforms needed for the
# %       scattering transform.
# %    time_filters (cell): A cell array of filters used to define the time
# %       domain wavelets.
# %    freq_filters (cell): A cell array of filters used to define the
# %       log-frequency domain wavelets.
# %
# % Description
# %    To calculate the joint time-frequency scattering transform, we need to
# %    define a set of filter banks, both in the time and log-frequency domains.
# %    Given these filter banks, we can define the wavelet transforms that make
# %    up the layers of the scattering transform. The filter banks are generated
# %    using the FILTER_BANK function applied to the time_filt_opt and
# %    freq_filt_opt structs. The first layer is then created using the
# %    WAVELET_LAYER_1D function with the first time filter bank given as an
# %    argument. All subsequent layers are formed using the
# %    JOINT_TF_WAVELET_LAYER_1D function, which is given a pair of time domain
# %    and log-frequency filter banks as an argument. The scat_opt struct is
# %    given as arguments to all layers. The resulting function handles are
# %    stored in the cell array Wop, where each element corresponds to a separate
# %    layer of the scattering transform.
# %
# % See also
# %    JOINT_TF_WAVELET_LAYER_1D, WAVELET_FACTORY_1D
import math 
from kymatio.scattering1d.filter_bank import *

def scattering_filter_factory_joint( J_support, J_scattering, Q, r_psi=math.sqrt(0.5),
                              criterion_amplitude=1e-3, normalize='l1',
                              to_torch=False, max_subsampling = None, max_time_subsampling=None, max_freq_subsampling=None,
                              sigma0=0.1, alpha=5., P_max=5, eps=1e-7,
                              **kwargs ):
    #N, time_filt_opt, freq_filt_opt, scat_opt):
    
    # compute the spectral parameters of the filters
    sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = calibrate_scattering_filters(
        J_scattering, Q, r_psi=r_psi, sigma0=sigma0, alpha=alpha)

    # instantiate the dictionaries which will contain the filters
    phi_f = {}
    psi1_f = []
    psi2_f = [] 
    psi2_freq, psi2_time = [], []

    # desired support size of filters, no subsampling
    T = 2**J_support
    
    # dilate by 2^-mu along time, dilate by 2^-l along frequency
    # refactored 
    def compute_maximal_subsampling(j1s, j2s, max_subsampling):
        max_sub_phi = max_subsampling
        max_sub_psi2s = [ max_subsampling for j2 in j2s ]

        if max_subsampling is None:
            # compute the current value for the max_subsampling,
            # which depends on the input it can accept.
            for n2, j2 in enumerate(j2s):
                possible_subsamplings_after_order1 = [
                    j1 for j1 in j1s if j2 > j1]
                if len(possible_subsamplings_after_order1) > 0:
                    max_sub_psi2s[n2] = max(possible_subsamplings_after_order1)
                else:
                    max_sub_psi2s[n2] = 0

            # Determine the maximal subsampling for phi, which depends on the
            # input it can accept (both 1st and 2nd order)
            max_subsampling_after_psi1 = max(j1s)
            max_subsampling_after_psi2 = max(j2s)
            max_sub_phi = max(max_subsampling_after_psi1,
                          max_subsampling_after_psi2)
        return max_sub_phi, max_sub_psi2s

    max_sub_phi, max_sub_psi2s = compute_maximal_subsampling(j1s, j2s, max_subsampling)

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    for (n2, max_sub_psi2) in enumerate(max_sub_psi2s):
        # We first compute the filter without subsampling
        psi_f = {}
        psi_f[0] = morlet_1d(
            T, xi2[n2], sigma2[n2], normalize=normalize, P_max=P_max,
            eps=eps)
        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for subsampling in range(1, max_sub_psi2 + 1):
            factor_subsampling = 2**subsampling
            psi_f[subsampling] = periodize_filter_fourier(
                psi_f[0], nperiods=factor_subsampling)
        psi2_f.append(psi_f)

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with T=2**J_support
    for (n1, j1) in enumerate(j1s):
        psi1_f.append({0: morlet_1d(
            T, xi1[n1], sigma1[n1], normalize=normalize,
            P_max=P_max, eps=eps)})

    # compute the low-pass filters phi
    # compute the filters at all possible subsamplings
    phi_f[0] = gauss_1d(T, sigma_low, P_max=P_max, eps=eps)
    for subsampling in range(1, max_sub_phi + 1):
        factor_subsampling = 2**subsampling
        # compute the low_pass filter
        phi_f[subsampling] = periodize_filter_fourier(
            phi_f[0], nperiods=factor_subsampling)

    # Embed the meta information within the filters
    for (n1, j1) in enumerate(j1s):
        psi1_f[n1]['xi'] = xi1[n1]
        psi1_f[n1]['sigma'] = sigma1[n1]
        psi1_f[n1]['j'] = j1
    for (n2, j2) in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2[n2]
        psi2_f[n2]['sigma'] = sigma2[n2]
        psi2_f[n2]['j'] = j2
    phi_f['xi'] = 0.
    phi_f['sigma'] = sigma_low
    phi_f['j'] = 0

    # compute the support size allowing to pad without boundary errors
    # at the finest resolution
    t_max_phi = compute_temporal_support(
        phi_f[0].reshape(1, -1), criterion_amplitude=criterion_amplitude)

    # prepare for pytorch if necessary
    if to_torch:
        for k in phi_f.keys():
            if type(k) != str:
                # view(-1, 1).repeat(1, 2) because real numbers!
                phi_f[k] = torch.from_numpy(
                    phi_f[k]).view(-1, 1).repeat(1, 2)
        for psi_f in psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    # view(-1, 1).repeat(1, 2) because real numbers!
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).view(-1, 1).repeat(1, 2)
        for psi_f in psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    # view(-1, 1).repeat(1, 2) because real numbers!
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).view(-1, 1).repeat(1, 2)

    return phi_f, psi1_f, psi2_f, t_max_phi