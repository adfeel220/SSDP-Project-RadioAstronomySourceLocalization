## SSDP project implementation
# Author: Chun-Tso Tsai
# Date: 2022-05-09

import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm



def bluebild(S, W, P, grid, wv, threshold, cal_sensitivity=False):
    '''
    params:
    S: visibility matrix of one time frame. shape = (n_antenna, n_antenna)
    W: beamforming weight of one time frame. shape = (n_antenna, n_antenna)
    P: antenna positions of one time frame. shape = (n_antenna, 3)
    grid: the grid of the sky region in the estimation area. shape = (n_px, 3)
    wv: wavelength of the observation
    threshold: the threshold to choose leading eigenpairs

    output:
    I: estimated intensity function of the grid
    '''
    n_antenna = W.shape[0]

    # Compute the kernal of Gram matrix
    G_phi = np.zeros((n_antenna, n_antenna))
    for dim in range(P.shape[1]):
        Pi = np.tile(P[:, dim], (n_antenna, 1))
        Pj = Pi.T
        G_phi[:,:] += np.power((Pi - Pj)/wv, 2)
    G_phi = 4 * np.pi * np.sinc( 2*np.sqrt(G_phi) )  # shape = (n_antenna, n_antenna)

    G_psi = W.conj().T @ G_phi @ W  # shape = (n_antenna, n_antenna)

    # Find eigenpairs for the generalized eigenvalue problem
    eigvals, eigvecs = eigh(S, G_psi, eigvals_only=False)
    # eigenavlues are in ascending order, flip it to descending
    eigvals = np.flipud(eigvals)
    eigvecs = np.fliplr(eigvecs)

    # Find the K leading eigenpairs w.r.t. the threshold
    K = 0
    total_eig = np.sum(eigvals)
    total_leading_K_eig = 0
    for k in range(eigvals.shape[0]):
        total_leading_K_eig += eigvals[k]
        if total_leading_K_eig/total_eig >= threshold:
            K = k+1
            break
    # print(f'Selecting {K} leading eigenpairs out of {n_antenna} for threshold = {threshold}')

    # Find the normalized eigenvectors
    v = np.zeros((n_antenna, K), dtype=np.csingle)
    for k in range(K):
        v[:, k] = eigvecs[:, k] / np.sqrt(eigvecs[:,k].conj().T @ G_psi @ eigvecs[:,k])
    D = eigvals[:K]
    
    # Assign 
    Phi = np.exp(2j*np.pi/wv * grid@P.conj().T)     # Phi.shape = (n_px, n_antenna)
    E = Phi @ W @ v                 # E.shape = (n_px, K)
    intensity = np.real(E*E.conj()) @ D    # intensity.shape = (n_px,)

    if cal_sensitivity:
        E_all = Phi @ W @ eigvecs
        sensitivity = np.sum(np.real(E_all * E_all.conj()), axis=1) # sensitivity.shape = (n_px,)
        return intensity, sensitivity
    else:
        return intensity



def bluebild_long_exposure(S, W, P, grid, wv, threshold, time_window_lng=None, time_start=0, process_id=None, equalize=True):
    '''
    params:
    S: visibility matrix. shape = (n_timeframes, n_antenna, n_antenna)
    W: beamforming weight. shape = (n_timeframes, n_antenna, n_antenna)
    P: antenna positions. shape = (n_timeframes, n_antenna, 3)
    grid: the grid of the sky region in the estimation area. shape = (n_px, 3)
    wv: wavelength of the observation
    threshold: the threshold to choose leading eigenpairs

    output:
    I: estimated intensity function of the grid
    '''
    n_time = S.shape[0]
    n_px   = grid.shape[0]

    if time_window_lng is None:
        time_window_lng = n_time

    aggre_intensity = np.zeros(n_px)
    print(f'Calculate intensity from {time_start} with length {time_window_lng}. Threshold = {threshold}')

    # Single process, print the status bar
    if process_id is None:
        for t in tqdm(range(time_start, time_start+time_window_lng)):
            intensity, sensitivity = bluebild(S[t,:,:], W[t,:,:], P[t,:,:], grid, wv, threshold, cal_sensitivity=True)
            if equalize:
                equalized_intensity = intensity / sensitivity
            else:
                equalized_intensity = intensity

            aggre_intensity += equalized_intensity
    # Concurrent process, no print rough status
    else:
        for t in range(time_start, time_start+time_window_lng):
            intensity, sensitivity = bluebild(S[t,:,:], W[t,:,:], P[t,:,:], grid, wv, threshold, cal_sensitivity=True)
            if equalize:
                equalized_intensity = intensity / sensitivity
            else:
                equalized_intensity = intensity

            aggre_intensity += equalized_intensity

    return aggre_intensity / time_window_lng