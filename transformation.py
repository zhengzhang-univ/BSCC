import numpy as np


def coordinate_mapping_sph2car(sph_coords_array):
    # sph_coords_array = [phi, the]
    # confined on the unit sphere, whose centre is the origin.
    x = np.sin(sph_coords_array[1]) * np.cos(sph_coords_array[0])
    y = np.sin(sph_coords_array[1]) * np.sin(sph_coords_array[0])
    z = np.cos(sph_coords_array[1])
    return x, y, z


def coordinate_mapping_car2sph(XYZ_array):
    # confined on the unit sphere, whose centre is the origin.
    aux = np.sqrt(XYZ_array[0] ** 2 + XYZ_array[1] ** 2)
    the = np.arctan2(aux, XYZ_array[2])
    phi = np.arctan2(XYZ_array[1], XYZ_array[0])
    if phi < 0:
        phi += 2 * np.pi
    return the, phi  # in radians

"""
def jacobian_car_sph(phi, the):
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_the, cos_the = np.sin(the), np.cos(the)
    matr = np.zeros(shape=(3, 3))
    matr[:, 0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
    matr[:, 1] = cos_the * cos_phi, cos_the * sin_phi, -sin_the
    matr[:, 2] = -sin_the * sin_phi, sin_the * cos_phi, 0.0
    return matr


def jacobian_sph_car(phi, the):
    matr = jacobian_car_sph(phi, the)
    return np.linalg.inv(matr)
"""

def rotation_matrix_local2eq(phi, the):
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_the, cos_the = np.sin(the), np.cos(the)
    matr = np.zeros(shape=(3, 3))
    matr[0] = cos_the*cos_phi, cos_the*sin_phi, -sin_the
    matr[1] = -sin_phi, cos_phi, 0.0
    matr[2] = sin_the*cos_phi, sin_the*sin_phi, cos_the
    return matr

def rotation_matrix_sph2car(phi, the):
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_the, cos_the = np.sin(the), np.cos(the)
    matr = np.zeros(shape=(3, 3))
    matr[0] = sin_the*cos_phi, sin_the*sin_phi, cos_the
    matr[1] = cos_the*cos_phi, cos_the*sin_phi, -sin_the
    matr[2] = -sin_phi, cos_phi, 0.0
    return matr


def Field_sph2car(sph_coords_array):
    aux = rotation_matrix_sph2car(sph_coords_array[0], sph_coords_array[1])
    matr = aux[:, 1:]
    return matr


def Field_car2sph(sph_coords_array):
    # In spherical coordinates.
    matr = rotation_matrix_sph2car(sph_coords_array[0], sph_coords_array[1])
    return matr.T


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


def ant_Car_to_local_Car(xAxis, zAxis):
    return pointing_matrix(xAxis, zAxis)


def local_Car_to_ant_Car(xAxis, zAxis):
    return pointing_matrix(xAxis, zAxis).T


def Local_Car_to_Equatorial_Car_single_LST(LST, antenna_lat = np.pi / 2 ):
    # LST should be given in seconds, and lat in radians from -pi/2 to pi/2.
    phi = (LST / 3600.) * (np.pi / 12.0)
    the = np.pi / 2 - antenna_lat
    return rotation_matrix_local2eq(phi, the)

def Equatorial_Car_to_Local_Car_single_LST(LST, antenna_lat = np.pi / 2 ):
    # LST should be given in seconds, and lat in radians from -pi/2 to pi/2.
    phi = (LST / 3600.) * (np.pi / 12.0)
    the = np.pi / 2 - antenna_lat
    return rotation_matrix(phi, the).T


def Local_Car_to_Equatorial_Car(LST_array, lat):
    # LST_array is a 2d array of shape (ntime,1).
    LSTs = np.array(LST_array).reshape(-1,1)
    return np.apply_along_axis(Local_Car_to_Equatorial_Car_single_LST, -1, LSTs,  antenna_lat=lat)


def Equatorial_Car_to_Local_Car(LST_array, lat):
    # LST should be given in seconds, and lat in radians from -pi/2 to pi/2.
    return np.swapaxes(Local_Car_to_Equatorial_Car(LST_array, lat), -1, -2)





