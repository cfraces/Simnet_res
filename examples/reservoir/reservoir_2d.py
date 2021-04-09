from sympy import Symbol, sin, exp, tanh, Function, Number
import numpy as np
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle, Channel2D, Line
from simnet.pdes import PDES
from simnet.controller import SimNetController
from simnet.sympy_utils.functions import parabola

# params for domain
channel_length = (-2.5, 2.5)
channel_width = (-0.5, 0.5)
permeability = 1.0
inlet_vel = 1.5

x, y, t_symbol = Symbol('x'), Symbol('y'), Symbol('t')

# define geometry
channel = Channel2D((channel_length[0], channel_width[0]),
                    (channel_length[1], channel_width[1]))
geo = channel

inlet = Line((channel_length[0], channel_width[0]),
             (channel_length[0], channel_width[1]), -1)
outlet = Line((channel_length[1], channel_width[0]),
              (channel_length[1], channel_width[1]), 1)
plane1 = Line((channel_length[0] + 0.5, channel_width[0]),
              (channel_length[0] + 0.5, channel_width[1]), 1)
plane2 = Line((channel_length[0] + 1.0, channel_width[0]),
              (channel_length[0] + 1.0, channel_width[1]), 1)
plane3 = Line((channel_length[0] + 3.0, channel_width[0]),
              (channel_length[0] + 3.0, channel_width[1]), 1)
plane4 = Line((channel_length[0] + 3.5, channel_width[0]),
              (channel_length[0] + 3.5, channel_width[1]), 1)


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
        Dimension of the equation (1, 2, or 3). Default is 2.
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
    self.equations['flow_x'] = Twx * p.diff(x) + Tox * p.diff(x)
    self.equations['flow_y'] = Twy * p.diff(y) + Toy * p.diff(y)


class ReservoirTrain(TrainDomain):
  def __init__(self, **config):
    super(ReservoirTrain, self).__init__()


    # initial
    initial_conditions = geo.interior_bc(outvar_sympy={'sw': 0,
                                                       'sw__t': 0},
                                         batch_size_per_area=2048 // 2,
                                         lambda_sympy={'lambda_sw': 100.0,
                                                       'lambda_sw__t': 1.0},
                                         bounds={x: (channel_length), y: (channel_width)},
                                         param_ranges={t_symbol: 0}
                                         )
    self.add(initial_conditions, name='IC')

    # inlet
    parabola_sympy = parabola(y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel)
    inletBC = inlet.boundary_bc(outvar_sympy={'sw': 1, 'p': 1},
                                batch_size_per_area=64)
    self.add(inletBC, name="Inlet")

    # outlet
    outletBC = outlet.boundary_bc(outvar_sympy={'p': 0},
                                  batch_size_per_area=64)
    self.add(outletBC, name="Outlet")

    # so slip channel
    channelWall = channel.boundary_bc(outvar_sympy={'flow_x': 0, 'flow_y': 0},
                                      batch_size_per_area=256)
    self.add(channelWall, name="ChannelWall")

    # interior
    interior = geo.interior_bc(
      outvar_sympy={'residual_water': 0, 'residual_oil': 0},
      bounds={x: (channel_length), y: (channel_width)},
      lambda_sympy={'lambda_residual_water': geo.sdf,
                    'lambda_residual_oil': geo.sdf},
      batch_size_per_area=1000)
    self.add(interior, name="Interior")


class ReservoirInference(InferenceDomain):
  def __init__(self, **config):
    super(ReservoirInference, self).__init__()
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(10000,
                                     bounds={x: (channel_length), y: (channel_width)},
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
