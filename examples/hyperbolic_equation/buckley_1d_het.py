# import SimNet library
from sympy import Symbol, sin, Heaviside, DiracDelta, sqrt, cos, ln, pi, Max, Number, Function
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, Inference
from simnet.sympy_utils.geometry_1d import Line1D
# from hyperbolic_equation import BuckleyHeterogeneous, GradMag
from simnet.controller import SimNetController
from simnet.architecture.fourier_net import FourierNetArch
import scipy.io as sio
from scipy.stats import truncnorm
import time

from simnet.pdes import PDES

# params for domain
L = float(1.0)
tf = 1.0

# define geometry
geo = Line1D(0, L)

# define sympy variables to parametrize time
# x = Symbol('x')
t_symbol = Symbol('t')
time_range = {t_symbol: (0, L)}

x = Symbol('x')


# make validation data
# deltaT = 0.01 * tf
# deltaX = L * 0.01 / 2.56
# xg = np.arange(0, L, deltaX)
# tg = np.arange(0, tf, deltaT)
# X, T = np.meshgrid(xg, tg)
# X = np.expand_dims(X.flatten(), axis=-1)
# T = np.expand_dims(T.flatten(), axis=-1)
# # v1
# ref_buckley_v_1 = sio.loadmat('./buckley/Buckley_het_bm_vel_wide_1.mat')
# u1 = np.expand_dims(ref_buckley_v_1['usol'].flatten(), axis=-1)
# outvar_v_1_numpy = {'u': u1}
# ud_1 = np.ones_like(X) * 1.0 + 1
# invar_numpy_v1 = {'x': X, 't': T, 'ud': ud_1}
#
# # v50
# ref_buckley_v_50 = sio.loadmat('./buckley/Buckley_het_bm_vel_wide_0.5.mat')
# u50 = np.expand_dims(ref_buckley_v_50['usol'].flatten(), axis=-1)
# outvar_v_50_numpy = {'u': u50}
# ud_50 = np.ones_like(X) * 0.5 + 1
# invar_numpy_v50 = {'x': X, 't': T, 'ud': ud_50}
#
# # v2
# ref_buckley_v_2 = sio.loadmat('./buckley/Buckley_het_bm_wide_vel_2.mat')
# u2 = np.expand_dims(ref_buckley_v_2['usol'].flatten(), axis=-1)
# outvar_v_2_numpy = {'u': u2}
# ud_2 = np.ones_like(X) * 2.0 + 1
# invar_numpy_v2 = {'x': X, 't': T, 'ud': ud_2}


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
  return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class BuckleyHeterogeneous(PDES):
  """
  Buckley Leverett equation with heterogeneities

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

  Examples
  ========
  >>> we = BuckleyHeterogeneousn(c=0.8, dim=3)
  >>> we.pprint(preview=False)
    wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    buckley_equation: u__t - 0.8*u__x - 0.8*u__y - 0.8*u__z
  >>> we = BuckleyHeterogeneousn(c='c', dim=2, time=False)
  >>> we.pprint(preview=False)
    buckley_equation: -c*u__x__x - c*u__y - c*c__x*u__x - c*c__y*u__y
  """

  name = 'BuckleyLeverettHeterogeneous'

  def __init__(self, u='u', dim=3, time=True):
    # set params
    self.u = u
    self.dim = dim
    self.time = time
    # self.weighting = weighting

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
    # if type(rand_v_1) is str:
    #   c = Function(c)(*input_variables)
    # elif type(c) in [float, int]:
    #   c = Number(c)
    rand_v_1 = Symbol("rand_v_1")
    rand_v_2 = Symbol("rand_v_2")
    # narrow
    # v_d = ((-2 * ln(rand_v_1)) ** 0.5) * cos(2 * np.pi * rand_v_2) / 5 + 1  # + x
    # wide
    v_d = ((-2 * ln(rand_v_1)) ** 0.5) * cos(2 * np.pi * rand_v_2) + 4  # + x

    # set equations
    self.equations = {}

    f = Max(-(1.366025403514163 * u) * (Heaviside(u - 0.577357735773577) - 1)
            + 2 * (u ** 2) * Heaviside(u - 0.577357735773577) / (2 * (u) ** 2 + (u - 1) ** 2), 0)

    # Piecewise f
    # swc = 0.0
    # sor = 0.
    # sinit = 0.
    # M = 2
    #
    # tangent = [0.568821882188219, 0.751580500446855]
    # f = v_d * Max(-(tangent[1] / (tangent[0] - sinit) * (u - sinit)) * (Heaviside(u - tangent[0]) - 1) + Heaviside(
    #   u - tangent[0]) * (u - swc) ** 2 / ((u - swc) ** 2 + ((1 - u - sor) ** 2) / M), 0)

    # f = v_d * (u - swc) ** 2 / ((u - swc) ** 2 + ((1 - u - sor) ** 2) / M)

    # Heterogenous
    # s_tangent = 0.577357735773577
    # f_tangent = 0.788685333982125 * v_d
    # f = Max(-(f_tangent * u / s_tangent) * (Heaviside(u - s_tangent) - 1)
    #         + 2 * v_d * (u ** 2) * Heaviside(u - s_tangent) / (2 * (u) ** 2 + (u - 1) ** 2), 0)

    self.equations['buckley_heterogeneous'] = u.diff(t) + v_d * f.diff(x).replace(DiracDelta, lambda x: 0)

    # Diffusion
    # eps = 1e-2
    # self.equations['buckley_heterogeneous'] = u.diff(t) + f.diff(x) - eps*(u.diff(x)).diff(x)

    # self.equations['buckley_heterogeneous'] = ((u.diff(t) + v_d * f.diff(x))
    #                                            / (Function(self.weighting)(*input_variables) + 1))


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': 0, 'u__t': 0},
                         bounds={x: (0.0, L)},
                         batch_size_per_area=5000,
                         lambda_sympy={'lambda_u': 1.0,
                                       'lambda_u__t': 1.0},
                         param_ranges={t_symbol: 0.0,
                                       Symbol('rand_v_1'): (1e-5, 1.0),
                                       Symbol('rand_v_2'): (1e-5, 1.0)})
    self.add(IC, name="IC")

    # boundary conditions
    BC = geo.boundary_bc(outvar_sympy={'u': 1},
                         batch_size_per_area=5000,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges={t_symbol: (0, tf),
                                       Symbol('rand_v_1'): (1e-5, 1.0),
                                       Symbol('rand_v_2'): (1e-5, 1.0)},
                         criteria=x <= 0)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_heterogeneous': 0},
                               bounds={x: (0.0, L)},
                               batch_size_per_area=5000,
                               lambda_sympy={'lambda_buckley_heterogeneous': 1.0},
                               param_ranges={t_symbol: (0, tf),
                                             Symbol('rand_v_1'): (1e-5, 1.0),
                                             Symbol('rand_v_2'): (1e-5, 1.0)})
    self.add(interior, name="Interior")


class BuckleyVal(ValidationDomain):
  def __init__(self, **config):
    super(BuckleyVal, self).__init__()
    # make validation data
    deltaT = 0.01
    deltaX = 0.01 / 2.56
    x = np.arange(0, L, deltaX)
    t = np.arange(0, L, deltaT)
    R1 = np.tile(np.random.rand(x.shape[0]), (t.shape[0], 1))
    R2 = np.tile(np.random.rand(x.shape[0]), (t.shape[0], 1))
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    R1 = np.expand_dims(R1.flatten(), axis=-1)
    R2 = np.expand_dims(R2.flatten(), axis=-1)

    w = sio.loadmat('./buckley/Buckley_Swc0_Sor_0_M_2.mat')
    u = np.expand_dims(w['usol'].flatten(), axis=-1)
    invar_numpy = {'x': X, 't': T, 'rand_v_1': R1, 'rand_v_2': R2}
    outvar_numpy = {'u': u}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


class BuckleyInference(InferenceDomain):
  def __init__(self, **config):
    super(BuckleyInference, self).__init__()
    # save entire domain
    # Normal distribution
    sample_vel = get_truncated_normal(mean=1, sd=0.3, low=0.5, upp=2).rvs(300)
    # for i, velocity in enumerate(np.linspace(vel_ranges[0], vel_ranges[1], 10)):
    for i, velocity in enumerate(sample_vel):
      velocity = float(velocity)
      sampled_interior = geo.sample_interior(1024 * 5,
                                             bounds={x: (0, L)},
                                             param_ranges={t_symbol: (0, tf),
                                                           Symbol('rand_v_1'): (1e-5, 1.0),
                                                           Symbol('rand_v_2'): (1e-5, 1.0)})
      interior = Inference(sampled_interior, ['u'])
      self.add(interior, name="Inference_" + str(i).zfill(5))


# Define neural network
class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal
  inference_domain = BuckleyInference
  arch = FourierNetArch

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.equations = BuckleyHeterogeneous(u='u', dim=1, time=True).make_node()
    # self.equations = (
    #   BuckleyHeterogeneous(u='u', dim=1, time=True).make_node(stop_gradients=['grad_magnitude_u'])
    #   + GradMag('u').make_node())
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't', 'rand_v_1', 'rand_v_2'],
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './network_checkpoint/buckley_het_f_{}'.format(int(time.time())),
      'max_steps': 30000,
      'decay_steps': 300,
      'start_lr': 5e-4,
      'rec_results_cpu': True,
      'amp': False,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
