from transformation import *
from scipy.interpolate import RegularGridInterpolator


Pauli_I = 0.5 * np.array([[1., 0.],
                          [0., 1.]])
Pauli_Q = 0.5 * np.array([[1., 0.],
                          [0., -1.]])
Pauli_U = 0.5 * np.array([[0., 1.],
                          [1., 0.]])
Pauli_V = 0.5 * np.array([[0., -1.j],
                          [1.j, 0.]])

pauli_array = np.array([Pauli_I, Pauli_Q, Pauli_U, Pauli_V])


class E_field():
    def __init__(self, ant_coords, e_field_data, alignment_x, alignment_z, ant_lat):
        # ant_coords: antenna coordinates array of shape (N_sky, 2)
        # where N_sky = N_phi * N_theta.
        # E_field: (N_freq, N_sky, 2)
        # ant_lat: antenna latitude in radians.
        self.pointing_mat = pointing_matrix(alignment_x, alignment_z)
        self.nfreq = e_field_data.shape[0]
        self.field = e_field_data
        self.antenna_latitude = ant_lat
        self.phi_array = np.unique(ant_coords[:, 0])
        self.theta_array = np.unique(ant_coords[:, 1])
        print("The far-field object has been initialized!")

    def map_sky_coords_to_ant_coords(self, LSTs, eq_sph_coords):
        # eq_sph_coods: (N_sky, 2)
        eq_car_coords = coordinate_mapping_sph2car(eq_sph_coords)  # shape: (N_sky, 3)
        R_eq2h = rotation_matrix_eq2local(LSTs, self.antenna_latitude)  # shape: (N_LST, 3, 3)
        ant_car_coords = np.einsum("ij,tjk,sk -> tsi", self.pointing_mat.T, R_eq2h,eq_car_coords)
        ant_sph_coords = coordinate_mapping_car2sph(ant_car_coords)  # shape: (N_LST, N_sky, 2)

        R_sc = rotation_matrix_sph2car(ant_sph_coords)  # shape: (N_LST, N_sky, 3, 2)
        R_h2eq = rotation_matrix_local2eq(LSTs, self.antenna_latitude) # shape: (N_LST, 3, 3)
        R_cs_eq = rotation_matrix_car2sph(eq_sph_coords)  # shape: (N_sky, 3, 3)
        field_trans_mat = np.einsum("sij, tjk, tskl -> tsil",
                                    R_cs_eq, R_h2eq@self.pointing_mat, R_sc) # shape: (N_LST, N_sky, 3, 2)
        return ant_sph_coords, field_trans_mat

    def make_interp(self, points, freq_ind):
        # the shape pf "points": (N_LST, N_sky, 2)
        result = np.zeros(shape=points.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[freq_ind, :, :, i],
                                             bounds_error=False,
                                             fill_value=0.)
            for t in range(points.shape[0]):
                result[t, :, i] = interp(points[t])
        return result  # shape: (N_LST, N_sky, 2)

    def e_field_in_eq_coords(self, LSTs, sky_coords, freq_ind):
        ant_coords, field_transform_operator = self.map_sky_coords_to_ant_coords(LSTs, sky_coords)
        interpolated_efield_ant = self.make_interp(ant_coords, freq_ind) # shape: (N_LST, N_sky, 2)
        del ant_coords
        result = np.einsum("tsij, tsj -> tsi", field_transform_operator, interpolated_efield_ant)
        return result[:, :, 1:]

    def generate_auto_beam_at_LSTs(self, LSTs, sky_coords, freq_ind, time_averaged = False):
        # E_field_integrated = np.mean(self.e_field_in_eq_coords(LSTs, sky_coords, freq_ind),
        #                              axis=0)
        if time_averaged:
            E_field = np.mean(self.e_field_in_eq_coords(LSTs, sky_coords, freq_ind), axis=0)
            B_matrix = np.einsum("sl, plm, sm -> sp",
                                 np.conjugate(E_field),
                                 pauli_array,
                                 E_field)
        else:
            E_field = self.e_field_in_eq_coords(LSTs, sky_coords, freq_ind)
            B_matrix = np.einsum("tsl, plm, tsm -> tsp",
                                 np.conjugate(E_field),
                                 pauli_array,
                                 E_field)
        return B_matrix.real

