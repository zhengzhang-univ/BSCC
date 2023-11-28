from transformation import *
from scipy.interpolate import RegularGridInterpolator, RectSphereBivariateSpline


import healpy as hp


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
        self.R_ant2h = self.pointing_mat.T
        self.R_h2ant = self.pointing_mat
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
        ant_car_coords = np.einsum("ij,tjk,sk -> tsi", self.R_h2ant, R_eq2h, eq_car_coords)
        ant_sph_coords = coordinate_mapping_car2sph(ant_car_coords)  # shape: (N_LST, N_sky, 2)

        R_sc = rotation_matrix_sph2car(ant_sph_coords)  # shape: (N_LST, N_sky, 3, 2)
        R_h2eq = rotation_matrix_local2eq(LSTs, self.antenna_latitude) # shape: (N_LST, 3, 3)
        R_cs_eq = rotation_matrix_car2sph(eq_sph_coords)  # shape: (N_sky, 3, 3)
        field_trans_mat = np.einsum("sij, tjk, tskl -> tsil",
                                    R_cs_eq, R_h2eq@self.R_ant2h, R_sc) # shape: (N_LST, N_sky, 3, 2)
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
    
    def make_interp_2(self, points, freq_ind):
        # the shape pf "points": (N_LST, N_sky, 2)
        result = np.zeros(shape=points.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RectSphereBivariateSpline(self.phi_array[1:], self.theta_array[1:-1],
                                             self.field[freq_ind, 1:, 1:-1, i],
                                            pole_values=(beam[0, 0, i],None),
                                           pole_exact=(True,False))
            for t in range(points.shape[0]):
                result[t, :, i] = interp(points[t])
        return result  # shape: (N_LST, N_sky, 2)

    def e_field_in_eq_coords(self, LSTs, sky_coords, freq_ind):
        ant_coords, field_transform_operator = self.map_sky_coords_to_ant_coords(LSTs, sky_coords)
        interpolated_efield_ant = self.make_interp_2(ant_coords, freq_ind) # shape: (N_LST, N_sky, 2)
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


class E_field_thesis_old():
    def __init__(self, ant_coords, e_field_data, ant_lat):
        # ant_coords: antenna coordinates array of shape (N_sky, 2)
        # where N_sky = N_phi * N_theta.
        # E_field: (N_sky, 2)
        # ant_lat: antenna latitude in radians.
        self.field = e_field_data
        self.antenna_latitude = ant_lat
        self.phi_array = np.unique(ant_coords[:, 0])
        self.theta_array = np.unique(ant_coords[:, 1])
        print("The far-field object has been initialized!")
     

    def map_sky_coords_to_ant_coords(self, eq_sph_coords, beam_angles, orthogonal_feed=False):
        # eq_sph_coods: (N_sky, 2)
        R_ant2h, R_h2ant = R_ant_h(beam_angles, yy=orthogonal_feed)
        eq_car_coords = coordinate_mapping_sph2car(eq_sph_coords)  # shape: (N_sky, 3)
        R_eq2h = rotation_matrix_eq2local(0., self.antenna_latitude)[0]  
        ant_car_coords = np.einsum("dij,jk,sk -> dsi", R_h2ant, R_eq2h, eq_car_coords)
        ant_sph_coords = coordinate_mapping_car2sph(ant_car_coords)  # shape: (dim, N_sky, 2)
        R_sc = rotation_matrix_sph2car(ant_sph_coords)  # shape: (dim, N_sky, 3, 2)
        R_h2eq = rotation_matrix_local2eq(0., self.antenna_latitude)[0] 
        R_cs_eq = rotation_matrix_car2sph(eq_sph_coords)  # shape: (N_sky, 3, 3)
        field_trans_mat = np.einsum("sij, jm, dmk, dskl -> dsil",
                                    R_cs_eq, R_h2eq, R_ant2h, R_sc) # shape: (dim, N_sky, 3, 2)
        return ant_sph_coords, field_trans_mat


    def make_interp(self, points):
        # the shape pf "points": (dim, N_sky, 2)
        result = np.zeros(shape=points.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[:, :, i],
                                             bounds_error=False,
                                             fill_value=0.)
            for d in range(points.shape[0]):
                result[d, :, i] = interp(points[d])
        return result  # shape: (dim, N_sky, 2)

    def e_field_in_eq_coords(self, sky_coords, beam_angles, orthogonal_feed=False):
        ant_coords, field_transform_operator = self.map_sky_coords_to_ant_coords(sky_coords, beam_angles, orthogonal_feed)
        interpolated_efield_ant = self.make_interp(ant_coords) # shape: (dim, N_sky, 2)
        del ant_coords
        result = np.einsum("dsij, dsj -> dsi", field_transform_operator, interpolated_efield_ant)
        return result[:, :, 1:]

    def generate_auto_beam(self, sky_coords, beam_angles, orthogonal_feed=False):
        E_field = self.e_field_in_eq_coords(sky_coords, beam_angles, orthogonal_feed)
        B_matrix = np.einsum("dsl, plm, dsm -> dps",
                             np.conjugate(E_field),
                             pauli_array,
                             E_field)
        return B_matrix.real
    
    def generate_multipole_beams(self, sky_coords, beam_angles):
        B_maps_x = self.generate_auto_beam(sky_coords, beam_angles)
        B_matrix_x = []
        for d in range(B_maps_x.shape[0]):
            B_matrix_x.append(hp.sphtfunc.map2alm(B_maps_x[d][:-1]))
        #B_maps_y = self.generate_auto_beam(sky_coords, beam_angles, orthogonal_feed=True)
        #B_matrix_y = []
        #for d in range(B_maps_y.shape[0]):
        #    B_matrix_y.append(hp.sphtfunc.map2alm(B_maps_y[d][:-1]))
        #return np.array([B_maps_x, B_maps_y]), np.array([B_matrix_x, B_matrix_y])
        return B_maps_x, B_matrix_x


class E_field_thesis():
    def __init__(self, ant_coords, e_field_data, ant_lat):
        # ant_coords: antenna coordinates array of shape (N_sky, 2), in radians 
        # where N_sky = N_phi * N_theta.
        # E_field: (N_sky, 2)
        # ant_lat: antenna latitude in radians.
        self.Nside = 128
        self.phi_array = np.unique(ant_coords[:, 0])
        self.theta_array = np.unique(ant_coords[:, 1])
        
        self.e_field_healpix(e_field_data)
        
        self.antenna_latitude = ant_lat

        print("The far-field object has been initialized!")
        
    def e_field_healpix(self, complex_field):
        nside = self.Nside
        npix = hp.nside2npix(nside)
        
        theta_vals = self.theta_array
        phi_vals = self.phi_array
                
        real_e_field_map = np.full(npix, hp.UNSEEN)
        imag_e_field_map = np.full(npix, hp.UNSEEN)
        real_e_field_map_1 = np.full(npix, hp.UNSEEN)
        imag_e_field_map_1 = np.full(npix, hp.UNSEEN)
        
        for idx in range(npix):
            h_theta, h_phi = hp.pix2ang(nside, idx) 
            dphi = np.abs(phi_vals - h_phi)
            dtheta = np.abs(theta_vals - h_theta)
            closest_phi_idx = np.argmin(dphi)
            closest_theta_idx = np.argmin(dtheta)

            real_e_field_map[idx] = complex_field[closest_phi_idx, closest_theta_idx, 0].real
            imag_e_field_map[idx] = complex_field[closest_phi_idx, closest_theta_idx, 0].imag
            real_e_field_map_1[idx] = complex_field[closest_phi_idx, closest_theta_idx, 1].real
            imag_e_field_map_1[idx] = complex_field[closest_phi_idx, closest_theta_idx, 1].imag

        self.e_field_0_real = real_e_field_map
        self.e_field_0_imag = imag_e_field_map
                
        self.e_field_1_real = real_e_field_map_1
        self.e_field_1_imag = imag_e_field_map_1
        
        return
     

    def map_sky_coords_to_ant_coords(self, eq_sph_coords, beam_angles):
        # eq_sph_coods: (N_sky, 2)
        # beam_angles in radians.
        R_ant2h, R_h2ant = R_ant_h(beam_angles, yy=False)
        eq_car_coords = coordinate_mapping_sph2car(eq_sph_coords)  # shape: (N_sky, 3)
        R_eq2h = rotation_matrix_eq2local(0., self.antenna_latitude)[0]  
        ant_car_coords = np.einsum("dij,jk,sk -> dsi", R_h2ant, R_eq2h, eq_car_coords)
        ant_sph_coords = coordinate_mapping_car2sph(ant_car_coords)  # shape: (dim, N_sky, 2)
        R_sc = rotation_matrix_sph2car(ant_sph_coords)  # shape: (dim, N_sky, 3, 2)
        R_h2eq = rotation_matrix_local2eq(0., self.antenna_latitude)[0] 
        R_cs_eq = rotation_matrix_car2sph(eq_sph_coords)  # shape: (N_sky, 3, 3)
        field_trans_mat = np.einsum("sij, jm, dmk, dskl -> dsil",
                                    R_cs_eq, R_h2eq, R_ant2h, R_sc) # shape: (dim, N_sky, 3, 2)
        return ant_sph_coords, field_trans_mat


    def make_interp(self, points):
        # the shape pf "points": (dim, N_sky, 2)
        result = np.zeros(shape=points.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[:, :, i],
                                             bounds_error=False,
                                             method = "linear"
                                             #fill_value=0.
                                            )
            for d in range(points.shape[0]):
                result[d, :, i] = interp(points[d])
        return result  # shape: (dim, N_sky, 2)
    
    def make_interp_2(self, points):
        # the shape pf "points": (N_LST, N_sky, 2)
        result = np.zeros(shape=points.shape,  dtype=complex)
        
        for t in range(points.shape[0]):
            target_theta, target_phi = points[t, :, 1], points[t, :, 0]
            # Convert (interp_theta, interp_phi) to HEALPix indices
            interp_indices = hp.ang2pix(self.Nside, target_theta, target_phi)
            result[t, :, 0] = self.e_field_0_real[interp_indices] + 1j*self.e_field_0_imag[interp_indices]
            result[t, :, 1] = self.e_field_1_real[interp_indices] + 1j*self.e_field_1_imag[interp_indices]
        return result  # shape: (N_LST, N_sky, 2)

    def e_field_in_eq_coords(self, sky_coords, beam_angles):
        ant_coords, field_transform_operator = self.map_sky_coords_to_ant_coords(sky_coords, beam_angles)
        interpolated_efield_ant = self.make_interp_2(ant_coords) # shape: (dim, N_sky, 2)
        del ant_coords
        result = np.einsum("dsij, dsj -> dsi", field_transform_operator, interpolated_efield_ant)
        return result[:, :, 1:]

    def generate_auto_beam(self, sky_coords, beam_angles):
        E_field = self.e_field_in_eq_coords(sky_coords, beam_angles)
        B_matrix = np.einsum("dsl, plm, dsm -> dps",
                             np.conjugate(E_field),
                             pauli_array,
                             E_field)
        return B_matrix.real
    
    def generate_multipole_beams(self, sky_coords, beam_angles):
        B_maps_x = self.generate_auto_beam(sky_coords, beam_angles)
        B_matrix_x = []
        for d in range(B_maps_x.shape[0]):
            B_matrix_x.append(hp.sphtfunc.map2alm(B_maps_x[d][:-1]))
        #B_maps_y = self.generate_auto_beam(sky_coords, beam_angles, orthogonal_feed=True)
        #B_matrix_y = []
        #for d in range(B_maps_y.shape[0]):
        #    B_matrix_y.append(hp.sphtfunc.map2alm(B_maps_y[d][:-1]))
        #return np.array([B_maps_x, B_maps_y]), np.array([B_matrix_x, B_matrix_y])
        return B_maps_x, B_matrix_x



