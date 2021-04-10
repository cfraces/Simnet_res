from sympy import Symbol, sin, exp, tanh, Function, Number
import numpy as np
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.pdes import PDES
from simnet.controller import SimNetController

# params for domain
height = 1.0
width = 1.0
permeability = 1.0

# define geometry
geo = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

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


class GravityEquation(PDES):
  """
    Darcy with Gravity equation

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
  name = 'GravityEquation'

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
    self.equations['gravity_equation'] = (u.diff(t, 1)
                                          + v.diff(x, 1)
                                          + f.diff(y, 1)
                                          - c * u.diff(z, 1))


# define closed boundary conditions
class ClosedBoundary(PDES):
  """
    Closed boundary condition for wave problems

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

    self.equations['closed_boundary'] = v


class GravityTrain(TrainDomain):
  def __init__(self, **config):
    super(GravityTrain, self).__init__()
    initial_conditions = geo.interior_bc(outvar_sympy={'z': 0.5,
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
    interior = geo.interior_bc(outvar_sympy={'gravity_equation': 0},
                               bounds={x: (-width / 2, width / 2),
                                       y: (-height / 2, height / 2)},
                               batch_size_per_area=2048 // 2,
                               lambda_sympy={'lambda_gravity_equation': time_length},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


class GravityInference(InferenceDomain):
  def __init__(self, **config):
    super(GravityInference, self).__init__()
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(5000,
                                     bounds={x: (0, 1), y: (0, 1)},
                                     param_ranges={t_symbol: float(specific_t)})
      inf = Inference(interior, ['z'])
      self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class GravitySolver(Solver):
  train_domain = GravityTrain
  inference_domain = GravityInference

  # arch = FourierNetArch

  def __init__(self, **config):
    super(GravitySolver, self).__init__(**config)

    self.equations = (GravityEquation(u='z', c=1, dim=2, time=True).make_node())

    gravity_net = self.arch.make_node(name='gravity_net',
                                      inputs=['x', 'y', 't'],
                                      outputs=['z'])

    self.nets = [gravity_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity/simple_{}'.format(int(time.time())),
      'max_steps': 400000,
      'decay_steps': 4000,
      'start_lr': 3e-4,
      'layer_size': 256,
    })


if __name__ == '__main__':
  ctr = SimNetController(GravitySolver)
  ctr.run()
