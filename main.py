from transformation import *
from FarFieldTransfer import E_field
from BeamSim import Beam_pattern

path = "/Users/zhengzhang/Dropbox/CarlaShareSept2022/Data 02_09_2022/HIRAX_201_cut.txt"
alignment_x = np.array([1., 0., 0.])
alignment_z = np.array([0., 0., 1.])
antenna_latitude = 1.

antenna_sph_coords = np.loadtxt(path,
                                comments=('// >>', '361 181', '#'),
                                usecols=(0, 1),
                                max_rows=361 * 181,).reshape(361, 181, 2)

antenna_sph_coords = np.deg2rad(antenna_sph_coords)

e_field = (np.loadtxt(path,
                      comments=('// >>', '361 181'),
                      usecols=(2, 4),
                      ) + 1j * np.loadtxt(path,
                      comments=('// >>', '361 181'),
                      usecols=(3, 5),
                      )).reshape(201, 361, 181, 2)

Far_field_obj = E_field(antenna_sph_coords, e_field, alignment_x, alignment_z, antenna_latitude)

del e_field

Sky_coords = np.deg2rad(np.loadtxt(path,
                                comments=('// >>', '361 181', '#'),
                                usecols=(0, 1),
                                max_rows=361 * 181,))[:100, :]

test = Beam_pattern(Sky_coords, 1000, 1001, 1)

test.generate_auto_beam(Far_field_obj)
