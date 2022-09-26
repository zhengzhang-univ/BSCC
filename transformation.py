import numpy as np


def Coord_sph2car(sph_coords_array):
    # sph_coords_array = [phi, the]
    # confined on the unit sphere, whose centre is the origin.
    x = np.sin(sph_coords_array[1]) * np.cos(sph_coords_array[0])
    y = np.sin(sph_coords_array[1]) * np.sin(sph_coords_array[0])
    z = np.cos(sph_coords_array[1])
    return x, y, z


def Coord_car2sph(XYZ_array):
    # confined on the unit sphere, whose centre is the origin.
    aux = np.sqrt(XYZ_array[0] ** 2 + XYZ_array[1] ** 2)
    the = np.arctan2(aux, XYZ_array[2])
    phi = np.arctan2(XYZ_array[1], XYZ_array[0])
    if phi < 0:
        phi += 2 * np.pi
    return the, phi  # in radians


def jacobian_sph_car(phi, the):
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_the, cos_the = np.sin(the), np.cos(the)
    matr = np.zeros(shape=(3, 3))
    matr[0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
    matr[1] = cos_phi * cos_the, sin_phi * cos_the, -sin_the
    matr[2] = -sin_phi, cos_phi, 0.0
    return matr


def jacobian_car_sph(phi, the):
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_the, cos_the = np.sin(the), np.cos(the)
    matr = np.zeros(shape=(3, 3))
    matr[:, 0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
    matr[:, 1] = cos_phi * cos_the, sin_phi * cos_the, -sin_the
    matr[:, 2] = -sin_phi, cos_phi, 0.0
    return matr


def Field_sph2car(sph_coords_array):
    aux = jacobian_car_sph(sph_coords_array[0], sph_coords_array[1])
    matr = aux[:, 1:]
    return matr


def Field_car2sph(sph_coords_array):
    # In spherical coordinates.
    matr = jacobian_sph_car(sph_coords_array[0], sph_coords_array[1])
    return matr


def pointing_matrix(xAxis, zAxis):
    # Mapping beam Cartesian coordinates to local/horizontal Cartesian coordinates.
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
    return np.linalg.inv(pointing_matrix(xAxis, zAxis))


def Local_Car_to_Equatorial_Car_single_LST(LST, antenna_lat = np.pi / 2 ):
    # LST should be given in seconds, and lat in radians from -pi/2 to pi/2.
    phi = (LST / 3600.) * (np.pi / 12.0)
    the = np.pi / 2 - antenna_lat
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_the, cos_the = np.sin(the), np.cos(the)
    matr = np.zeros(shape=(3, 3))
    matr[:, 0] = cos_phi*cos_the, sin_phi*cos_the, -sin_the
    matr[:, 1] = -sin_phi, cos_phi, 0.0
    matr[:, 2] = sin_the*cos_phi, sin_the*sin_phi, cos_the
    return matr


def Local_Car_to_Equatorial_Car(LST_array, lat):
    # LST_array is a 2d array of shape (ntime,1).
    LSTs = np.array(LST_array).reshape(-1,1)
    return np.apply_along_axis(Local_Car_to_Equatorial_Car_single_LST, -1, LSTs,  antenna_lat=lat)


def Equatorial_Car_to_Local_Car(LST_array, lat):
    # LST should be given in seconds, and lat in radians from -pi/2 to pi/2.
    return np.swapaxes(Local_Car_to_Equatorial_Car(LST_array, lat), -1, -2)




def Coord_ant_sph2equatorial_sph(ant_sph_coords, pointingMatrix, LST, lat):
    aux_C = np.apply_along_axis(Coord_sph2car, -1, ant_sph_coords)  # Coordinates transformation
    aux_C = np.einsum("lm, jkm -> jkl", pointingMatrix, aux_C)  # Coordinates transformation
    lcar2eqcar = TransMatr_LocalCar_to_EquatorialCar(LST, lat)
    aux_C = np.einsum("lm, jkm -> jkl", lcar2eqcar, aux_C)  # Coordinates transformation
    aux_C = np.apply_along_axis(Coord_car2sph, -1, aux_C)
    return aux_C

def Coord_equatorial_sph_2_ant_sph(sky_coords, pointingMatrix, LST, lat):
    aux_C = np.apply_along_axis(Coord_sph2car, -1, sky_coords)
    eqcar2lcar = TransMatr_EquatorialCar_2_LocalCar(LST, lat)
    aux_C = np.einsum("lm, jkm -> jkl", eqcar2lcar, aux_C)
    # Coordinates transformation
    aux_C = np.einsum("lm, jkm -> jkl", np.linalg.inv(pointingMatrix), aux_C)  # Coordinates transformation
    aux_C = np.apply_along_axis(Coord_car2sph, -1, aux_C)
    return aux_C

def Coord_Field_ant2equatorial(ant_sph_coords, E_field, pointingMatrix, LST, lat):
    # from antenna spherical system to antenna Cartesian system
    aux_C = np.apply_along_axis(Coord_sph2car, -1, ant_sph_coords)  # Coordinates transformation
    JMat = np.apply_along_axis(Field_TransMatr_sph2car, -1, ant_sph_coords)
    aux_E = np.einsum("jklm, ijkm -> ijkl", JMat, E_field) # Field transformation
    # from antenna Cartesian system to local-sky (horizontal) system
    aux_C = np.einsum("lm, jkm -> jkl", pointingMatrix, aux_C)  # Coordinates transformation
    aux_E = np.einsum("lm, ijkm -> ijkl", pointingMatrix, aux_E)  # Field transformation
    # from local-sky (horizontal) Cartesian system to equatorial Cartesian system
    lcar2eqcar = TransMatr_LocalCar_to_EquatorialCar(LST, lat)
    aux_C = np.einsum("lm, jkm -> jkl", lcar2eqcar, aux_C)  # Coordinates transformation
    aux_E = np.einsum("lm, ijkm -> ijkl", lcar2eqcar, aux_E)  # Field transformation
    # from equatorial Cartesian system to equatorial spherical system
    aux_C = np.apply_along_axis(Coord_car2sph, -1, aux_C)
    JMat = np.apply_along_axis(Field_TransMatr_car2sph, -1, aux_C)
    aux_E = np.einsum("jklm, ijkm -> ijkl", JMat, aux_E)
    return aux_C, aux_E

