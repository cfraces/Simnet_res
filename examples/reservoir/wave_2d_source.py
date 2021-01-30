from sympy import Symbol, sin, exp, tanh, Function, Number, pi
import numpy as np
import scipy.io as sio
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle, Line
from simnet.pdes import PDES
from simnet.PDES.wave_equation import WaveEquation
from simnet.controller import SimNetController
from simnet.node import Node
from simnet.architecture.fourier_net import FourierNetArch

# define geometry
geo = Rectangle((0, 0),
                (1, 1))
hr_radius = 0.10
geo_hr = Circle((0.5, 0.5),  # high resolution sample circle around source
                hr_radius)
geo = geo - geo_hr

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# define time domain
time_length = 1.0
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}


class WaveTrain(TrainDomain):
  def __init__(self, **config):
    super(WaveTrain, self).__init__()
    # boundary conditions
    edges = geo.boundary_bc(outvar_sympy={'z': 0},
                            batch_size_per_area=1024,
                            lambda_sympy={'lambda_z': time_length},
                            param_ranges=time_range)
    self.add(edges, name="Edges")

    # interior
    f0 = 15
    gaussian = exp(-1000 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    source = f0 ** 2 * gaussian * (1 - 2 * pi ** 2 * f0 ** 2 * (t_symbol - 1 / f0) ** 2) * exp(
      -pi ** 2 * f0 ** 2 * (t_symbol - 1 / f0) ** 2)
    interior = geo.interior_bc(outvar_sympy={'wave_equation': -source},
                               bounds={x: (0, 1), y: (0, 1)},
                               batch_size_per_area=2 ** 11,
                               lambda_sympy={'lambda_wave_equation': time_length},
                               param_ranges=time_range)
    self.add(interior, name="Interior")

    # hr interior
    interior_hr = geo_hr.interior_bc(outvar_sympy={'wave_equation': -source},
                                     bounds={x: (0.5 - hr_radius, 0.5 + hr_radius),
                                             y: (0.5 - hr_radius, 0.5 + hr_radius)},
                                     batch_size_per_area=2 ** 16,
                                     lambda_sympy={'lambda_wave_equation': time_length},
                                     param_ranges=time_range)
    self.add(interior_hr, name="InteriorHR")


class WaveInference(InferenceDomain):
  def __init__(self, **config):
    super(WaveInference, self).__init__()
    # inf data
    mesh_x, mesh_y = np.meshgrid(np.linspace(0, 1, 128),
                                 np.linspace(0, 1, 128),
                                 indexing='ij')
    mesh_x = np.expand_dims(mesh_x.flatten(), axis=-1)
    mesh_y = np.expand_dims(mesh_y.flatten(), axis=-1)
    for i, specific_t in enumerate(np.linspace(0, time_length, 20)):
      interior = {'x': mesh_x,
                  'y': mesh_y,
                  't': np.full_like(mesh_x, float(specific_t))}
      inf = Inference(interior, ['z'])
      self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class WaveSolver(Solver):
  train_domain = WaveTrain
  inference_domain = WaveInference

  def __init__(self, **config):
    super(WaveSolver, self).__init__(**config)

    self.equations = (WaveEquation(u='z', c=1.0, dim=2, time=True).make_node()
                      + [Node.from_sympy(Symbol('t') ** 2 * Symbol('z_star'),
                                         'z')])  # enforce "hard" boundary conditions

    wave_net = self.arch.make_node(name='wave_net',
                                   inputs=['x', 'y', 't'],
                                   outputs=['z_star'])
    self.nets = [wave_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_2d/wave_{}'.format(int(time.time())),
      'rec_results_cpu': True,
      'max_steps': 400000,
      'decay_steps': 4000,
      'start_lr': 3e-4,
      'layer_size': 256,
    })


if __name__ == '__main__':
  ctr = SimNetController(WaveSolver)
  ctr.run()
