import numpy as np


def coordinate_mapping_sph2car(spherical_coords_array):
    sph_coords_array = spherical_coords_array.reshape(-1, 2)
    nsky = sph_coords_array.shape[0]
    theta_array = sph_coords_array[:, 1]
    phi_array = sph_coords_array[:, 0]
    car_coords_array = np.zeros(shape=(nsky, 3))
    car_coords_array[:, 0] = np.sin(theta_array) * np.cos(phi_array)
    car_coords_array[:, 1] = np.sin(theta_array) * np.sin(phi_array)
    car_coords_array[:, 2] = np.cos(theta_array)
    return car_coords_array.reshape(spherical_coords_array.shape[:-1]+(3,))


def coordinate_mapping_car2sph(car_coords_array):
    XYZ_array = car_coords_array.reshape(-1, 3)
    nsky = XYZ_array.shape[0]
    sph_coords_array = np.zeros(shape=(nsky, 2))
    aux = np.sqrt(XYZ_array[:, 0] ** 2 + XYZ_array[:, 1] ** 2)
    sph_coords_array[:, 1] = np.arctan2(aux, XYZ_array[:, 2])
    phi_array = np.arctan2(XYZ_array[:, 1], XYZ_array[:, 0])

    def correct_phi(phi):
        if phi < 0:
            result = phi + 2 * np.pi
        else:
            result = phi
        return result

    func = np.vectorize(correct_phi)
    sph_coords_array[:, 0] = func(phi_array)
    return sph_coords_array.reshape(car_coords_array.shape[:-1]+(2,))


def rotation_matrix_local2eq(LSTs, latitude):
    # return shape: (N_LST, 3, 3)
    LSTs = np.array(LSTs).flatten()
    phi_array = (LSTs / 3600.) * (np.pi / 12.0)
    the_array = (np.pi / 2 - latitude) * np.ones_like(phi_array)

    sin_phi, cos_phi = np.sin(phi_array), np.cos(phi_array)
    sin_the, cos_the = np.sin(the_array), np.cos(the_array)
    matrix = np.zeros(shape=(LSTs.shape[0], 3, 3))
    matrix[:, 0, 0] = cos_the * cos_phi
    matrix[:, 1, 0] = cos_the * sin_phi
    matrix[:, 2, 0] = -sin_the
    matrix[:, 0, 1] = -sin_phi
    matrix[:, 1, 1] = cos_phi
    matrix[:, 0, 2] = sin_the * cos_phi
    matrix[:, 1, 2] = sin_the * sin_phi
    matrix[:, 2, 2] = cos_the
    return matrix


def rotation_matrix_eq2local(LSTs, latitude):
    matrix = rotation_matrix_local2eq(LSTs, latitude)
    return np.swapaxes(matrix, -1, -2)


def rotation_matrix_sph2car(sph_coords_array):
    def rot_mat(phi_the_array):
        sin_phi, cos_phi = np.sin(phi_the_array[0]), np.cos(phi_the_array[0])
        sin_the, cos_the = np.sin(phi_the_array[1]), np.cos(phi_the_array[1])
        matr = np.zeros(shape=(3, 3))
        matr[:, 0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
        matr[:, 1] = cos_the * cos_phi, cos_the * sin_phi, -sin_the
        matr[:, 2] = -sin_phi, cos_phi, 0.0
        return matr[:, 1:]
    result = np.apply_along_axis(rot_mat, -1, sph_coords_array)
    return result


def rotation_matrix_car2sph(sph_coords_array):
    def rot_mat(phi_the_array):
        sin_phi, cos_phi = np.sin(phi_the_array[0]), np.cos(phi_the_array[0])
        sin_the, cos_the = np.sin(phi_the_array[1]), np.cos(phi_the_array[1])
        matr = np.zeros(shape=(3, 3))
        matr[:, 0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
        matr[:, 1] = cos_the * cos_phi, cos_the * sin_phi, -sin_the
        matr[:, 2] = -sin_phi, cos_phi, 0.0
        return matr.T
    result = np.apply_along_axis(rot_mat, -1, sph_coords_array)
    return result


def pointing_matrix(x_Axis, z_Axis):
    # Mapping beam Cartesian coordinates to local/horizontal Cartesian coordinates.
    xAxis = np.array(x_Axis)
    zAxis = np.array(z_Axis)
    yAxis = np.cross(zAxis, xAxis)
    unit_xAxis = xAxis / np.linalg.norm(xAxis)
    unit_zAxis = zAxis / np.linalg.norm(zAxis)
    unit_yAxis = yAxis / np.linalg.norm(yAxis)
    point_Mat = np.zeros(shape=(3, 3))
    point_Mat[:, 0] = unit_xAxis
    point_Mat[:, 1] = unit_yAxis
    point_Mat[:, 2] = unit_zAxis
    return point_Mat




