## SSDP project implementation of Bluebild algorithm
# Author: Chun-Tso Tsai
# Date: 2022-05-17

from readline import get_endidx
import numpy as np
from tqdm import tqdm


def music(S:np.ndarray, W:np.ndarray, P:np.ndarray, grid:np.ndarray, wv:float, threshold:float, get_eigen:bool=False):
    '''
    params:
    S: visibility matrix of one time frame. shape = (n_antenna, n_antenna)
    W: beamforming weight of one time frame. shape = (n_antenna, n_antenna)
    P: antenna positions of one time frame. shape = (n_antenna, 3)
    grid: the grid of the sky region in the estimation area. shape = (n_px, 3)
    wv: wavelength of the observation
    threshold: the threshold to choose leading eigenpairs

    return:
    I: estimated intensity function of the grid. shape = (n_px, )
    eigvals: the sorted eigenvalues comptued in the process.
    '''
    n_px = grid.shape[0]

    ## Calculate and sort the eigenvalues in descending order
    eigvals, eigvecs = np.linalg.eig(S)
    eig_sorted_idx = eigvals.argsort()[::-1]

    eigvals = eigvals[eig_sorted_idx]
    eigvecs = eigvecs[:,eig_sorted_idx]

    # Find the K leading eigenpairs w.r.t. the threshold
    K = 0
    total_eig = np.sum(eigvals)
    total_leading_K_eig = 0
    for k in range(eigvals.shape[0]):
        total_leading_K_eig += eigvals[k]
        if total_leading_K_eig/total_eig >= threshold:
            K = k+1
            break

    # Select the last eigenpairs (from K to end)
    last_eigvecs = eigvecs[:, K:]
    # The phase difference matrix
    phase_shift = np.exp(-2j*np.pi/wv*(P @ grid.T)) # shape = (n_antennas, n_px)
    intensity = np.zeros(n_px)
    for px in range(n_px):
        intensity[px] = 1 / np.real(phase_shift[:,px].conj().T @ W @ last_eigvecs @ last_eigvecs.conj().T @ W.conj().T @ phase_shift[:,px])

    if get_eigen:
        return intensity, eigvals

    return intensity
    

def music_long_exposure(S:np.ndarray, W:np.ndarray, P:np.ndarray, grid:np.ndarray, wv:float, threshold:float, time_length=None) -> np.ndarray:
    '''
    params:
    S: visibility matrix. shape = (n_timeframes, n_antenna, n_antenna)
    W: beamforming weight. shape = (n_timeframes, n_antenna, n_antenna)
    P: antenna positions. shape = (n_timeframes, n_antenna, 3)
    grid: the grid of the sky region in the estimation area. shape = (n_px, 3)
    wv: wavelength of the observation
    threshold: the threshold to choose leading eigenpairs. value in [0,1]
    time_length: the number of time frames to average the signal. None to use all time frames

    return:
    I: estimated intensity function of the grid. shape = (n_px, )

    Example:
    >>> intensity_music = music_long_exposure(data['S'], data['W'], data['XYZ'], data['px_grid'], data['lambda_'], 0.7, 20)
    '''
    if time_length is None:
        n_timeframes = S.shape[0]
    else:
        n_timeframes = time_length
    n_pixel = grid.shape[0]

    intensity_long = np.zeros(n_pixel)
    for i in tqdm(range(n_timeframes)):
        intensity_long += music(S[i,:,:], W[i,:,:], P[i,:,:], grid, wv, threshold) / n_timeframes

    return intensity_long

