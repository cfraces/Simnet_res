# import SimNet library
from sympy import Symbol, sin, Heaviside, DiracDelta
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, Inference
from simnet.sympy_utils.geometry_1d import Line1D
from hyperbolic_equation import BuckleyEquationParam
from simnet.controller import SimNetController
import scipy.io as sio
import time

# params for domain
L = float(1)

# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
tf = 1.0

x = Symbol('x')
# time_range = {t_symbol: (0, tf)}

# make validation data
deltaT = 0.01 * tf
deltaX = L * 0.01 / 2.56
xg = np.arange(0, L, deltaX)
tg = np.arange(0, tf, deltaT)
X, T = np.meshgrid(xg, tg)
X = np.expand_dims(X.flatten(), axis=-1)
T = np.expand_dims(T.flatten(), axis=-1)

# v1
ref_buckley_v_1 = sio.loadmat('./buckley/Buckley_resid_0_vel_1.mat')
u1 = np.expand_dims(ref_buckley_v_1['usol'].flatten(), axis=-1)
outvar_v_1_numpy = {'u': u1}
ud_1 = np.zeros_like(X) + 1.0
invar_numpy_v1 = {'x': X, 't': T, 'ud': ud_1}

# v875
ref_buckley_v_875 = sio.loadmat('./buckley/Buckley_resid_0_vel_0.875.mat')
u875 = np.expand_dims(ref_buckley_v_875['usol'].flatten(), axis=-1)
outvar_v_875_numpy = {'u': u875}
ud_875 = np.zeros_like(X) + 0.875
invar_numpy_v875 = {'x': X, 't': T, 'ud': ud_875}

# v75
ref_buckley_v_75 = sio.loadmat('./buckley/Buckley_resid_0_vel_0.75.mat')
u75 = np.expand_dims(ref_buckley_v_75['usol'].flatten(), axis=-1)
outvar_v_75_numpy = {'u': u75}
ud_75 = np.zeros_like(X) + 0.75
invar_numpy_v75 = {'x': X, 't': T, 'ud': ud_75}

vel = Symbol('ud')
vel_ranges = (0.75, 1.0)
param_ranges = {vel: vel_ranges, t_symbol: (0, tf)}


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': 0, 'u__t': 0},
                         bounds={x: (0, L)},
                         batch_size_per_area=5000,
                         lambda_sympy={'lambda_u': 1.0,
                                       'lambda_u__t': 1.0},
                         param_ranges={t_symbol: 0.0, vel: vel_ranges})
    self.add(IC, name="IC")

    # boundary conditions
    BC = geo.boundary_bc(outvar_sympy={'u': 1},
                         batch_size_per_area=5000,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges=param_ranges,
                         criteria=x <= 0)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_equation_param': 0},
                               bounds={x: (0, L)},
                               batch_size_per_area=5000,
                               lambda_sympy={'lambda_buckley_equation_param': 1.0},
                               param_ranges=param_ranges)
    self.add(interior, name="Interior")


class BuckleyVal(ValidationDomain):
  def __init__(self, **config):
    super(BuckleyVal, self).__init__()

    # u 1
    val_v1 = Validation.from_numpy(invar_numpy_v1, outvar_v_1_numpy)
    self.add(val_v1, name='Val_v1')

    # u 875
    val_v875 = Validation.from_numpy(invar_numpy_v875, outvar_v_875_numpy)
    self.add(val_v875, name='Val_v875')

    # u 75
    val_v75 = Validation.from_numpy(invar_numpy_v75, outvar_v_75_numpy)
    self.add(val_v75, name='Val_v75')


class BuckleyInference(InferenceDomain):
  def __init__(self, **config):
    super(BuckleyInference, self).__init__()
    # save entire domain
    for i, velocity in enumerate(np.linspace(vel_ranges[0], vel_ranges[1], 10)):
      velocity = float(velocity)
      sampled_interior = geo.sample_interior(1024,
                                             bounds={x: (0, L)},
                                             param_ranges={vel: velocity})
      interior = Inference(sampled_interior, ['u'])
      self.add(interior, name="Inference_" + str(i).zfill(5))


# Define neural network
class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal
  inference_domain = BuckleyInference

  # arch = SirenArch

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.equations = BuckleyEquationParam(u='u', c='ud', dim=1, time=True).make_node()
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't', 'ud'],
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './network_checkpoint/buckley_{}'.format(int(time.time())),
      'max_steps': 30000,
      'decay_steps': 300,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
