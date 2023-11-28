import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

from scipy.fftpack import dctn, fftn, fftshift, fftfreq
from scipy.interpolate import RectBivariateSpline, RectSphereBivariateSpline

    
def plane2sphere_v2(x, y):
    phi = np.arctan2(y, x)
    for i in range(len(phi)):
        if  phi[i] < 0.:
            phi[i] += 2. * np.pi
    theta = 2. * np.arctan(np.sqrt(x ** 2 + y ** 2) / 2.)
    return phi, theta
    
def Beam_scaled(beam, theta_coord):
    aux_theta = theta_coord/2.
    intensity_rescaling_factor = np.cos(aux_theta) ** 5 / (np.cos(aux_theta) - np.sin(aux_theta))
    B_matrix = intensity_rescaling_factor[:, :, np.newaxis] * beam 
    return B_matrix

def interpolation(Ndim, phi, theta, beam, target_phi, target_theta):
    Beams_interpolated = np.zeros(shape=(Ndim, Ndim, beam.shape[-1]))
    for i in np.arange(beam.shape[-1]):
        #interp = RectSphereBivariateSpline(theta[1:], phi[:-1], beam[:-1, 1:, i].T, 
                                           #pole_values=(beam[0, 0, i],None),
                                           #pole_exact=(True,False)
                                          #)
        interp = RectSphereBivariateSpline(theta[:], phi[1:], beam[:, 1:, i], 
                                           pole_values=(beam[0, 0, i],None),
                                           pole_exact=(True,False)
                                          )
        Beams_interpolated[:, :, i] = interp.ev(target_theta, target_phi).reshape(Ndim, Ndim)
    return Beams_interpolated

def directional_window(Beam, theta_coords, theta_max=75., alpha=0.05):
    for i in range(theta_coords.shape[0]):
        for j in range(theta_coords.shape[1]):
            if theta_coords[i,j] < theta_max:
                Beam[i, j, :] *= np.exp(-alpha*(1/(theta_coords[i,j]-theta_max)**2 - 1/theta_max**2))
            else:
                Beam[i, j, :] = 0.
    return Beam

def q_matrix(x_fft_coords, y_fft_coords, baseline_length):
    xk_res = x_fft_coords[1]-x_fft_coords[0]
    half_dim = np.int64((len(x_fft_coords) -1)/2. - baseline_length/xk_res)
    ind_x = np.absolute(x_fft_coords).argmin()
    ind_x_l = ind_x-half_dim
    ind_x_r = ind_x+half_dim+1
    ind_y = np.absolute(y_fft_coords).argmin()
    ind_y_l = ind_y-half_dim
    ind_y_r = ind_y+half_dim+1
    return  half_dim, np.meshgrid(x_fft_coords[ind_x_l:ind_x_r], y_fft_coords[ind_x_l:ind_x_r])
    
def FFT_Beam_matrix(FFT_beam, x_fft_coords, y_fft_coords, half_dim, baseline_length, angle):
    q_x_shift = baseline_length * np.cos(np.deg2rad(angle))
    q_y_shift = baseline_length * np.sin(np.deg2rad(angle))
    ind_x = np.absolute(x_fft_coords - q_x_shift).argmin()
    ind_y = np.absolute(y_fft_coords - q_y_shift).argmin()
    return FFT_beam[ind_y - half_dim: ind_y + half_dim+1, ind_x - half_dim: ind_x + half_dim + 1, :]