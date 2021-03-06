from sympy import Symbol, sin, exp, tanh, Function, Number
import numpy as np
import scipy.io as sio
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.pdes import PDES
from simnet.controller import SimNetController
from rnn import GRUArch

# define geometry
geo = Rectangle((0, 0),
                (1, 1))

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# define time domain
time_length = 2.0
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

# define wave speed numpy array
mesh_x, mesh_y = np.meshgrid(np.linspace(0, 1, 512),
                             np.linspace(0, 1, 512),
                             indexing='ij')


# Define gravity equation
class TwoPhaseFlow(PDES):
  """
  Darcy with gravity

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
  name = 'TwoPhaseFlow'

  def __init__(self, sw='sw', perm='perm', dim=3, time=True):
    # set params
    self.sw = sw
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
    assert type(sw) == str, "u needs to be string"
    sw = Function(sw)(*input_variables)

    # permeability coefficient
    if type(perm) is str:
      perm = Function(perm)(*input_variables)
    elif type(perm) in [float, int]:
      perm = Number(perm)

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
    f = fw(sw)
    vw = lambda S: g * (rhoo - rhow) / (phi * muw) * perm * conv * fw(S)
    v = vw(sw)

    # set equations
    self.equations = {}
    self.equations['darcy_equation'] = (sw.diff(t, 1)
                                        + v.diff(x, 1)
                                        + f.diff(y, 1))


class ClosedBoundary(PDES):
  """
  Closed boundary condition for Flow problems

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

  name = 'ClosedBoundary'

  def __init__(self, sw='sw', perm='perm', dim=3, time=True):
    # set params
    self.sw = sw
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
    assert type(sw) == str, "sw needs to be string"
    sw = Function(sw)(*input_variables)

    # permeability coefficient
    if type(perm) is str:
      perm = Function(perm)(*input_variables)
    elif type(perm) in [float, int]:
      perm = Number(perm)

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

    vw = lambda S: g * (rhoo - rhow) / (phi * muw) * perm * conv * fw(S)
    v = vw(sw)
    # set equations
    # self.equations = {'open_boundary': v ** 2}
    # u.diff(t)
    # + normal_x * c * u.diff(x)
    # + normal_y * c * u.diff(y)
    # + normal_z * c * u.diff(z)
    self.equations = {}
    self.equations['closed_boundary'] = v


class ReservoirTrain(TrainDomain):
  def __init__(self, **config):
    super(ReservoirTrain, self).__init__()
    # train network to emulate wave speed
    # permeability_speed = BC.from_numpy(permeability_invar, permeability_outvar, batch_size=2048)
    # self.add(permeability_speed, "Permeability")

    # initial conditions exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2))
    # (1+tanh(80*(x-0.5)))/2
    initial_conditions = geo.interior_bc(outvar_sympy={'z': exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2)),
                                                       'z__t': 0},
                                         batch_size_per_area=2048 // 2,
                                         lambda_sympy={'lambda_z': 100.0,
                                                       'lambda_z__t': 1.0},
                                         bounds={x: (0, 1), y: (0, 1)},
                                         param_ranges={t_symbol: 0})
    self.add(initial_conditions, name="IC")

    # boundary conditions
    edges = geo.boundary_bc(outvar_sympy={'closed_boundary': 0},
                            batch_size_per_area=2048 // 2,
                            lambda_sympy={'lambda_closed_boundary': 1.0 * time_length},
                            param_ranges=time_range)
    self.add(edges, name="Edges")

    # interior
    interior = geo.interior_bc(outvar_sympy={'darcy_equation': 0},
                               bounds={x: (0, 1), y: (0, 1)},
                               batch_size_per_area=2048 // 8,
                               lambda_sympy={'lambda_darcy_equation': geo.sdf},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


class Reservoirnference(InferenceDomain):
  def __init__(self, **config):
    super(Reservoirnference, self).__init__()
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(5000,
                                     bounds={x: (0, 1), y: (0, 1)},
                                     param_ranges={t_symbol: float(specific_t)})
      inf = Inference(interior, ['z'])
      self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class ReservoirSolver(Solver):
  train_domain = ReservoirTrain
  inference_domain = Reservoirnference
  arch = GRUArch

  def __init__(self, **config):
    super(ReservoirSolver, self).__init__(**config)

    self.arch.set_time_steps(0, time_length, int(time_length/10))

    self.equations = (
      TwoPhaseFlow(sw='z', perm=1, dim=2, time=True).make_node()
      + ClosedBoundary(sw='z', perm=1, dim=2, time=True).make_node()
    )
    # + OpenBoundary(sw='z', perm='perm', dim=2, time=True).make_node(stop_gradients=['perm', 'perm__x', 'perm__y'])

    reservoir_net = self.arch.make_node(name='reservoir_net',
                                        inputs=['x', 'y', 't'],
                                        outputs=['z'])
    # perm_net = self.arch.make_node(name='perm_net',
    #                                inputs=['x', 'y'],
    #                                outputs=['perm'])
    self.nets = [reservoir_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_2d/drop_gru_{}'.format(int(time.time())),
      'max_steps': 300000,
      'decay_steps': 4000,
      'start_lr': 3e-4,
      'layer_size': 256,
    })


if __name__ == '__main__':
  ctr = SimNetController(ReservoirSolver)
  ctr.run()
