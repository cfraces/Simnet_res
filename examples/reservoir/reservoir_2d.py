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


class TwoPhaseFlow(PDES):
  """
     Darcy without Gravity equation

    Parameters
    ==========
    sw : float, Sympy Symbol/Expr, str
        The water saturation. deependent variable
    permeability : float, Sympy Symbol/Expr, str
        Permeability. If `perm` is a str then it is
        converted to Sympy Function of form 'perm(x,y,z,t)'.
        If 'perm' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.
  """
  name = 'TwoPhaseFlow'

  def __init__(self, sw, permeability=1):
    self.sw = sw

    # coordinates
    x, y = Symbol('x'), Symbol('y')

    t = Symbol('t')

    # make input
    input_variables = {'x': x, 'y': y, 't': t}

    # saturation component
    sw = Function('sw')(*input_variables)

    # pressure component
    p = Function('p')(*input_variables)

    # permeability
    if type(permeability) is str:
      permeability = Function(permeability)(*input_variables)
    elif type(permeability) in [float, int]:
      permeability = Number(permeability)

    # Relative permeability
    krw = sw ** 1.5
    kro = (1 - sw) ** 1.5

    # Viscosity
    muo = 1  # TODO: Change
    muw = 1

    # FVF
    # TODO: change with true formula
    Bw = 1
    Bo = 1

    # Unit conversion
    alpha = 0.001127

    # Pore compressibility
    phi = 0.25

    # transmissibility
    Twx = krw * permeability * alpha / (muw * Bw)
    Twy = krw * permeability * alpha / (muw * Bw)  # TODO: Add gravity
    Tox = kro * permeability * alpha / (muo * Bo)
    Toy = kro * permeability * alpha / (muo * Bo)  # TODO: Add gravity

    # set equations:
    self.equations = {}
    self.equations['water_residual'] = (Twx * p.diff(x)).diff(x) + (Twy * p.diff(y)).diff(y) - (phi * sw / Bw).diff(t)
    self.equations['oil_residual'] = (Tox * p.diff(x)).diff(x) + (Toy * p.diff(y)).diff(y) + (phi * sw / Bo).diff(t)


class ReservoirTrain(TrainDomain):
  def __init__(self, **config):
    super(ReservoirTrain, self).__init__()

    # initial condition
    initial_condition = geo.boundary_bc(outvar_sympy={'s_w': exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2)),
                                                      's_w__t': 0,
                                                      'p': 15},
                                        bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
                                        batch_size_per_area=10000,
                                        param_ranges={t_symbol: 0})
    self.add(initial_condition, name="IC")

    # no slip
    edges = geo.boundary_bc(outvar_sympy={'s_w': 1, 'p': 15},
                            batch_size_per_area=10000,
                            param_ranges=time_range)
    self.add(edges, name="Edges")

    # Interior
    interior = geo.interior_bc(outvar_sympy={'water_residual': 0, 'oil_residual': 0},
                               bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
                               batch_size_per_area=10000,
                               param_ranges=time_range
                               )
    self.add(interior, name="Interior")


class ReservoirInference(InferenceDomain):
  def __init__(self, **config):
    super(ReservoirInference, self).__init__()
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(5000,
                                     bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
                                     param_ranges={t_symbol: float(specific_t)})
      inf = Inference(interior, ['s_w', 'p'])
      self.add(inf, "Inference_" + str(i).zfill(4))


class ReservoirSolver(Solver):
  def __init__(self, **config):
    super(ReservoirSolver, self).__init__(**config)

    self.equations = (TwoPhaseFlow(sw='s_w').make_node())

    reservoir_net = self.arch.make_node(name='reservoir_net',
                                        inputs=['x', 'y', 't'],
                                        outputs=['s_w', 'p'])
    self.nets = [reservoir_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_2d/closed_{}'.format(int(time.time())),
      'max_steps': 400000,
      'decay_steps': 4000,
      'start_lr': 3e-4
    })


if __name__ == '__main__':
  ctr = SimNetController(ReservoirSolver)
  ctr.run()
