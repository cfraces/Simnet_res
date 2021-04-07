from sympy import Symbol, sin, exp, tanh, Function, Number, pi
import numpy as np
import scipy.io as sio
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.pdes import PDES
from simnet.controller import SimNetController
from simnet.node import Node

# from simnet.architecture.fourier_net import FourierNetArch

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
time_length = 2.0
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}


# Define wave equation, we will not use SimNet version
# because we want the equation z_tt = c^2 (z_xx + z_yy)
# instead of z_tt = grad(c^2 (z_x + z_y))
class WaveEquation(PDES):
  """
  Wave equation

  Parameters
  ==========
  u : str
      The dependent variable.
  c : float, Sympy Symbol/Expr, str
      Wave speed coefficient. If `c` is a str then it is
      converted to Sympy Function of form 'c(x,y,z,t)'.
      If 'c' is a Sympy Symbol or Expression then this
      is substituted into the equation.
  dim : int
      Dimension of the wave equation (1, 2, or 3). Default is 2.
  time : bool
      If time-dependent equations or not. Default is True.
  """
  name = 'WaveEquation'

  def __init__(self, u='u', c='c', dim=3, time=True):
    # set params
    self.u = u
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 'y': y, 'z': z, 't': t}
    if self.dim == 1:
      input_variables.pop('y')
      input_variables.pop('z')
    elif self.dim == 2:
      input_variables.pop('z')
    if not self.time:
      input_variables.pop('t')

    # Scalar function
    assert type(u) == str, "u needs to be string"
    u = Function(u)(*input_variables)

    # wave speed coefficient
    if type(c) is str:
      c = Function(c)(*input_variables)
    elif type(c) in [float, int]:
      c = Number(c)

    nw = 2
    no = 2
    # Residual oil saturation
    Sor = 0.0
    # Residual water saturation
    Swc = 0.0
    # End points
    krwmax = 1
    kromax = 1
    # Gravity
    g = 32.2  # ft/s2   9.81 / (0.304)
    phi = 0.25
    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no
    # Densities (lbm / scf)
    rhoo = 40  # Oil
    rhow = 62.238  # Water
    # Viscosities
    muo = 2e-4  # lb/ft-s
    muw = 6e-6
    conv = 9.1688e-8  # md to ft2
    fw = lambda S: krw(S) * kro(S) / (kro(S) + krw(S) * muo / muw)
    f = fw(u)
    vw = lambda S: g * (rhoo - rhow) / (phi * muw) * c * conv * fw(S)
    v = vw(u)

    # set equations
    self.equations = {}
    self.equations['wave_equation'] = (u.diff(t, 1)
                                       + v.diff(x, 1)
                                       + f.diff(y, 1)
                                       - c * u.diff(z, 1))


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
      'network_dir': './checkpoint_2d/drop_{}'.format(int(time.time())),
      'rec_results_cpu': True,
      'max_steps': 400000,
      'decay_steps': 4000,
      'start_lr': 3e-4,
      'layer_size': 256,
    })


if __name__ == '__main__':
  ctr = SimNetController(WaveSolver)
  ctr.run()
