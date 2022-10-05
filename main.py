from transformation import *
from FarFieldTransfer import E_field
import healpy as hp

# from BeamSim import Beam_pattern

alignment_x = np.array([1., 0., 0.])
alignment_z = np.array([0., 0., 1.])
antenna_latitude = 1.

path = "/Users/zhengzhang/PythonProjects/TIBEC/REACH_Efield.txt"
antenna_sph_coords = np.loadtxt(path,
                                comments=('// >>', '73 37', '#'),
                                usecols=(0, 1),
                                max_rows=73 * 37, ).reshape(-1, 2)

antenna_sph_coords = np.deg2rad(antenna_sph_coords)

e_field = (np.loadtxt(path,
                      comments=('// >>', '73 37'),
                      usecols=(2, 4),
                      ) + 1j * np.loadtxt(path,
                                          comments=('// >>', '73 37'),
                                          usecols=(3, 5),
                                          )).reshape(26, 73, 37, 2)

test = E_field(antenna_sph_coords, e_field, alignment_x, alignment_z, antenna_latitude)

del e_field, antenna_sph_coords


def healpix_map_coordinates(nside):
    npixel = 12 * nside ** 2
    result = np.zeros(shape=(npixel, 2))
    result[:, 1], result[:, 0] = hp.pixelfunc.pix2ang(nside, np.arange(npixel))
    return result


sky_sph_coords = healpix_map_coordinates(128)

result = test.generate_auto_beam([1, 1000, 2000], sky_sph_coords, 0)
