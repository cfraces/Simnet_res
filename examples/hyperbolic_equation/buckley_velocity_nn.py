from sympy import Symbol, exp, Function, Number
import numpy as np
import scipy.io as sio
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_1d import Line1D
from simnet.pdes import PDES
from simnet.controller import SimNetController

# params for domain
L = float(1.0)
tf = 1.0

# define geometry
geo = Line1D(0, L)

# define sympy variables to parametrize domain curves
x = Symbol('x')

# define time domain
time_length = 2.0
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

# define pore velocity numpy array
pore_vel_invar = {}
pore_vel_invar['x'] = np.linspace(0, L, 512).reshape(-1, 1)

pore_vel_outvar = {}
pore_vel_outvar['c'] = np.ones_like(pore_vel_invar['x'])

# Increments
pore_vel_outvar['c'] = pore_vel_invar['x'] + 1.5 + np.cos(100 * pore_vel_invar['x'])


# 2 - (
# np.tanh(80 * (pore_vel_invar['x'] - 0.25)) / 4 + np.tanh(80 * (pore_vel_invar['x'] - 0.5)) / 4 +
# np.tanh(80 * (pore_vel_invar['x'] - 0.75)) / 4 + 0.75)


class BuckleyVelocity(PDES):
  name = 'BuckleyVelocity'

  def __init__(self, u='u', c='c', dim=3, time=True, weighting='grad_magnitude_u'):
    # set params
    self.u = u
    self.dim = dim
    self.time = time
    self.weighting = weighting

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

    swc = 0.0
    sor = 0.
    sinit = 0.
    M = 2
    f = (u - swc) ** 2 / ((u - swc) ** 2 + ((1 - u - sor) ** 2) / M)
    eps = 2.5e-3

    self.equations = {}
    self.equations['buckley_heterogeneous'] = u.diff(t) + c * f.diff(x)
    # self.equations['buckley_heterogeneous'] = ((u.diff(t) + c * f.diff(x) - eps*(u.diff(x)).diff(x))
    #                                            / (Function(self.weighting)(*input_variables) + 1))


class GradMag(PDES):
  name = 'GradMag'

  def __init__(self, u='u'):
    # set params
    self.u = u

    # coordinates
    x = Symbol('x')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 't': t}

    # Scalar function
    assert type(u) == str, "u needs to be string"
    u = Function(u)(*input_variables)

    # set equations
    self.equations = {}
    # self.equations['grad_magnitude_' + self.u] = 0.05*u.diff(t) ** 2 + 0.01*u.diff(x) ** 2
    self.equations['grad_magnitude_' + self.u] = u.diff(t) ** 2 + u.diff(x) ** 2


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # train network to emulate wave speed
    pore_vel = BC.from_numpy(pore_vel_invar, pore_vel_outvar, batch_size=2048)
    self.add(pore_vel, "PoreVel")

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'z': 0, 'z__t': 0},
                         bounds={x: (0.0, L)},
                         batch_size_per_area=5000,
                         lambda_sympy={'lambda_z': 1.0,
                                       'lambda_z__t': 1.0},
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    # boundary conditions
    edge = geo.boundary_bc(outvar_sympy={'z': 1},
                           batch_size_per_area=5000,
                           lambda_sympy={'lambda_z': 1.0},
                           param_ranges={t_symbol: (0, tf)},
                           criteria=x <= 0)
    self.add(edge, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_heterogeneous': 0},
                               bounds={x: (0.0, L)},
                               batch_size_per_area=5000,
                               lambda_sympy={'lambda_buckley_heterogeneous': 1.0},
                               param_ranges={t_symbol: (0, tf)})
    self.add(interior, name="Interior")


class BuckleyVal(ValidationDomain):
  def __init__(self, **config):
    super(BuckleyVal, self).__init__()
    # make validation data
    deltaT = 0.01
    deltaX = 0.01 / 2.56
    x = np.arange(0, L, deltaX)
    t = np.arange(0, L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    w = sio.loadmat('./buckley/Buckley_het_cos.mat')
    u = np.expand_dims(w['usol'].flatten(), axis=-1)
    invar_numpy = {'x': X, 't': T}
    outvar_numpy = {'z': u}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


class BuckleyInference(InferenceDomain):
  def __init__(self, **config):
    super(BuckleyInference, self).__init__()
    mesh_x, mesh_t = np.meshgrid(np.linspace(0, 1, 256),
                                 np.linspace(0, 1, 256),
                                 indexing='ij')
    mesh_x = np.expand_dims(mesh_x.flatten(), axis=-1)
    mesh_t = np.expand_dims(mesh_t.flatten(), axis=-1)
    sampled_interior = {'x': mesh_x,
                        't': mesh_t}
    # geo.sample_interior(1024 * 5,
    #                     bounds={x: (0, L)},
    #                     param_ranges={t_symbol: (0, tf)})
    interior = Inference(sampled_interior, ['z', 'c'])
    self.add(interior, name="Inference_1")


class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal
  inference_domain = BuckleyInference

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.equations = (
      BuckleyVelocity(u='z', c='c', dim=1, time=True).make_node(
        stop_gradients=['c', 'c__x', 'c__t']))
      # + GradMag('z').make_node())

    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't'],
                                      outputs=['z'])
    speed_net = self.arch.make_node(name='speed_net',
                                    inputs=['x'],
                                    outputs=['c'])

    self.nets = [buckley_net, speed_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './network_checkpoint/buckley_vel_cos_diff_weight{}'.format(int(time.time())),
      'max_steps': 70000,
      'decay_steps': 500,
      'start_lr': 5e-4,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
