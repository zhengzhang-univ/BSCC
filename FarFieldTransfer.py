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
        self.phi_array = np.unique(ant_coords[:,:,0])
        self.theta_array = np.unique(ant_coords[:,:,1])

    def make_interp(self, points, freq_ind):
        aux = points.reshape(-1, 2)
        result = np.zeros(shape=aux.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[freq_ind, :, :, i],
                                             bounds_error = False,
                                             fill_value = 0.)
            result[:, i] = interp(aux, method='cubic')
        return result.reshape(points.shape)


"""
    def antenna_sph2local_car(self, ant_sph_coords, E_field):
        coord = np.apply_along_axis(Coord_sph2car_arr, -1, ant_sph_coords)
        coord = np.einsum("lm, jkm -> jkl", self.pm, coord)
        JMat = np.apply_along_axis(Field_TransMatr_sph2car, -1, ant_sph_coords)
        field = np.einsum("jklm, ijkm -> ijkl", JMat, E_field)
        field = np.einsum("lm, ijkm -> ijkl", self.pm, field)
        return coord, field

    def local_car2equatorial_sph(self, local_coord, e_field, LST):
        lcar2eqcar = TransMatr_LocalCar_to_EquatorialCar(LST, self.lat)
        coord = np.einsum("lm, jkm -> jkl", lcar2eqcar, local_coord)  # Coordinates transformation
        field = np.einsum("lm, ijkm -> ijkl", lcar2eqcar, e_field)  # Field transformation
        # from equatorial Cartesian system to equatorial spherical system
        coord = np.apply_along_axis(Coord_car2sph_arr, -1, coord)
        JMat = np.apply_along_axis(Field_TransMatr_car2sph, -1, coord)
        field = np.einsum("jklm, ijkm -> ijkl", JMat, field)
        return coord, field

"""
