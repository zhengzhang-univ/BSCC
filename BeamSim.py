from transformation import *


Pauli_I = 0.5 * np.array([[1.,  0.],
                          [0.,  1.]])
Pauli_Q = 0.5 * np.array([[1.,  0.],
                          [0.,  -1.]])
Pauli_U = 0.5 * np.array([[0.,  1.],
                          [1.,  0.]])
Pauli_V = 0.5 * np.array([[0.,  -1.j],
                          [1.j,  0.]])

pauli_array = np.array([Pauli_I, Pauli_Q, Pauli_U, Pauli_V])


class Beam_pattern():
    def __init__(self, sky_coords, LST_start, LST_end, dt):
        self.coords = sky_coords
        self.equatorial_Cartesian_coords = np.apply_along_axis(Coord_sph2car, -1, self.coords)
        self.LSTs = np.arange(LST_start, LST_end, dt)
        self.nLST = self.LSTs.shape[0]

    def antenna_spherical_coords(self, E_obj):
        local_Cartesian_coords = np.einsum("ijk, nk -> inj",
                                           Equatorial_Car_to_Local_Car(self.LSTs,
                                                                       E_obj.antenna_latitude),  # shape: (N_LST, 3, 3)
                                           self.equatorial_Cartesian_coords) # shape: (N_LST, N_sky, 3)
        ant_Cartesian_coords = np.einsum("mk, ijk -> ijm",
                                         local_Car_to_ant_Car(E_obj.xAxis, E_obj.zAxis),
                                         local_Cartesian_coords) # shape: (N_LST, N_sky, 3)
        ant_spherical_coords = np.apply_along_axis(Coord_car2sph, -1,
                                                   ant_Cartesian_coords) # shape: (N_LST, N_sky, 2)
        return ant_spherical_coords


    def field_trans_operater_ant_sph2eq_sph(self, ant_spherical_coords, E_obj):
        field_trans_aux = np.apply_along_axis(Field_sph2car, -1,
                                              ant_spherical_coords) # shape: (N_LST, N_sky, 3, 2)
        field_trans_aux = np.einsum("mk, ijkl -> ijml",
                                    ant_Car_to_local_Car(E_obj.xAxis, E_obj.zAxis),
                                    field_trans_aux) # shape: (N_LST, N_sky, 3, 2)
        field_trans_aux = np.einsum("imk, ijkl -> ijml",
                                    Local_Car_to_Equatorial_Car(self.LSTs,
                                                                E_obj.antenna_latitude),
                                    field_trans_aux) # shape: (N_LST, N_sky, 3, 2)
        aux = np.apply_along_axis(Field_car2sph, -1,
                                  ant_spherical_coords) # shape: (N_LST, N_sky, 3, 3)
        field_trans_aux = np.einsum("ijmk, ijkl -> ijml",
                                    aux,
                                    field_trans_aux)  # shape: (N_LST, N_sky, 3, 2)
        return field_trans_aux

    def transform_farfield(self, E_obj):
        ant_sph_coords = self.antenna_spherical_coords(E_obj)
        field_trans_operator = self.field_trans_operater_ant_sph2eq_sph(ant_sph_coords, E_obj)
        result = [] # shape: (N_freq, N_sky, 2)
        for i in np.arange(E_obj.nfreq):
            aux = E_obj.make_interp(ant_sph_coords, i) # shape: (N_LST, N_sky, 2)
            aux = np.einsum("ijkl, ijl -> ijk",
                            field_trans_operator, # shape: (N_LST, N_sky, 3, 2)
                            aux)
            result.append(np.mean(aux, axis=0))
        return np.array(result) # shape: (N_freq, N_sky, 3)

    def generate_beam(self, E_obj_1, E_obj_2):
        E_field1 = self.transform_farfield(E_obj_1)
        E_field2 = self.transform_farfield(E_obj_2)
        B_matrix = np.einsum("ijl, plm, ijm -> ijp",
                             np.conjugate(E_field1), pauli_array, E_field2) # (N_freq, N_sky, N_pol)
        return B_matrix

    def generate_auto_beam(self, E_obj):
        E_field = self.transform_farfield(E_obj)
        B_matrix = np.einsum("ijl, plm -> ijpm",
                             np.conjugate(E_field),
                             pauli_array)
        B_matrix = np.einsum("ijpm, ijm -> ijp", B_matrix, E_field)
        # B_matrix = np.einsum("ijl, plm, ijm -> ijp",
        #                      np.conjugate(E_field), pauli_array, E_field)  # (N_freq, N_sky, N_pol)
        return B_matrix




