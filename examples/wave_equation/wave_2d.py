from sympy import Symbol, sin, exp, tanh, Function, Number
import numpy as np
import scipy.io as sio

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.pdes import PDES
from simnet.controller import SimNetController
from simnet.node import Node
from simnet.architecture.fourier_net import FourierNetArch

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
wave_speed_invar = {}
wave_speed_invar['x'] = np.expand_dims(mesh_x.flatten(), axis=-1)
wave_speed_invar['y'] = np.expand_dims(mesh_y.flatten(), axis=-1)
wave_speed_outvar = {}
wave_speed_outvar['c'] = np.tanh(
    80 * (wave_speed_invar['x'] - 0.25)) / 4 + 0.75  # wave speed changes from 2 to 1 on 0.25 line.


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

        # set equations
        self.equations = {}
        self.equations['wave_equation'] = (u.diff(t, 2)
                                           - c ** 2 * u.diff(x, 2)
                                           - c ** 2 * u.diff(y, 2)
                                           - c ** 2 * u.diff(z, 2))


# define open boundary conditions
class OpenBoundary(PDES):
    """
  Open boundary condition for wave problems
  Ref: http://hplgit.github.io/wavebc/doc/pub/._wavebc_cyborg002.html

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

    name = 'OpenBoundary'

    def __init__(self, u='u', c='c', dim=3, time=True):
        # set params
        self.u = u
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

        # normal
        normal_x, normal_y, normal_z = Symbol('normal_x'), Symbol('normal_y'), Symbol('normal_z')

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

        # set equations
        self.equations = {}
        self.equations['open_boundary'] = (u.diff(t)
                                           + normal_x * c * u.diff(x)
                                           + normal_y * c * u.diff(y)
                                           + normal_z * c * u.diff(z))


class WaveTrain(TrainDomain):
    def __init__(self, **config):
        super(WaveTrain, self).__init__()
        # train network to emulate wave speed
        wave_speed = BC.from_numpy(wave_speed_invar, wave_speed_outvar, batch_size=2048 // 2)
        self.add(wave_speed, "WaveSpeed")

        # initial conditions
        initial_conditions = geo.interior_bc(outvar_sympy={'z': exp(-200 * ((x - 0.75) ** 2 + (y - 0.5) ** 2)),
                                                           'z__t': 0},
                                             batch_size_per_area=2048 // 2,
                                             lambda_sympy={'lambda_z': 300.0,
                                                           'lambda_z__t': 1.0},
                                             bounds={x: (0, 1), y: (0, 1)},
                                             param_ranges={t_symbol: 0})
        self.add(initial_conditions, name="IC")

        # boundary conditions
        edges = geo.boundary_bc(outvar_sympy={'open_boundary': 0},
                                batch_size_per_area=2048 // 2,
                                lambda_sympy={'lambda_open_boundary': 1.0 * time_length},
                                param_ranges=time_range)
        self.add(edges, name="Edges")

        # interior
        interior = geo.interior_bc(outvar_sympy={'wave_equation': 0},
                                   bounds={x: (0, 1), y: (0, 1)},
                                   batch_size_per_area=2048 // 2,
                                   lambda_sympy={'lambda_wave_equation': time_length},
                                   param_ranges=time_range)
        self.add(interior, name="Interior")


class WaveInference(InferenceDomain):
    def __init__(self, **config):
        super(WaveInference, self).__init__()
        # inf data
        for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
            interior = geo.sample_interior(20000,
                                           bounds={x: (0, 1), y: (0, 1)},
                                           param_ranges={t_symbol: float(specific_t)})
            inf = Inference(interior, ['z', 'c'])
            self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class WaveSolver(Solver):
    train_domain = WaveTrain
    inference_domain = WaveInference
    arch = FourierNetArch

    def __init__(self, **config):
        super(WaveSolver, self).__init__(**config)

        self.arch.set_frequencies(('full', [i / 2 for i in range(0, 10)]))

        self.equations = (WaveEquation(u='z', c='c', dim=2, time=True).make_node(stop_gradients=['c'])
                          + OpenBoundary(u='z', c='c', dim=2, time=True).make_node())

        wave_net = self.arch.make_node(name='wave_net',
                                       inputs=['x', 'y', 't'],
                                       outputs=['z'])
        speed_net = self.arch.make_node(name='speed_net',
                                        inputs=['x', 'y'],
                                        outputs=['c'])
        self.nets = [wave_net, speed_net]

    @classmethod  # Explain This
    def update_defaults(cls, defaults):
        defaults.update({
            'network_dir': './network_checkpoint_wave_2d_29',
            'max_steps': 200000,
            'decay_steps': 2000,
            'start_lr': 3e-4,
            'layer_size': 256,
        })


if __name__ == '__main__':
    ctr = SimNetController(WaveSolver)
    ctr.run()
