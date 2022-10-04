from transformation import *
from scipy.interpolate import RegularGridInterpolator


class E_field():
    def __init__(self, ant_coords, e_field_data, alignment_x, alignment_z, ant_lat):
        # ant_coords: (nphi, ntheta, 2)
        # E_field: (nfreq, nphi, ntheta, 2)
        # ant_lat: In radians.
        self.xAxis = np.array(alignment_x)
        self.zAxis = np.array(alignment_z)
        self.nfreq = e_field_data.shape[0]
        self.field = e_field_data
        self.antenna_latitude = ant_lat
        self.phi_array = np.unique(ant_coords[:, :, 0])
        self.theta_array = np.unique(ant_coords[:, :, 1])

    def make_interp(self, points, freq_ind):
        aux = points.reshape(-1, 2)
        result = np.zeros(shape=aux.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[freq_ind, :, :, i],
                                             bounds_error=False,
                                             fill_value=0.)
            result[:, i] = interp(aux, method='cubic')
        return result.reshape(points.shape)





