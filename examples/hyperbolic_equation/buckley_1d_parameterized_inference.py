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
from scipy.stats import truncnorm
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

# v10p
# ref_buckley_v_10p = sio.loadmat('./buckley/Buckley_M_2_vel_0.1.mat')
# u10p = np.expand_dims(ref_buckley_v_10p['usol'].flatten(), axis=-1)
# outvar_v_10p_numpy = {'u': u10p}
# ud_10p = np.zeros_like(X) + 0.1
# invar_numpy_v10p = {'x': X, 't': T, 'ud': ud_10p}

# v1
ref_buckley_v_1 = sio.loadmat('./buckley/Buckley_swc_100_vel_1.mat')
u1 = np.expand_dims(ref_buckley_v_1['usol'].flatten(), axis=-1)
outvar_v_1_numpy = {'u': u1}
ud_1 = np.zeros_like(X) + 1.0
invar_numpy_v1 = {'x': X, 't': T, 'ud': ud_1}

# v50
ref_buckley_v_50 = sio.loadmat('./buckley/Buckley_swc_100_vel_0.5.mat')
u50 = np.expand_dims(ref_buckley_v_50['usol'].flatten(), axis=-1)
outvar_v_50_numpy = {'u': u50}
ud_50 = np.zeros_like(X) + 0.5
invar_numpy_v50 = {'x': X, 't': T, 'ud': ud_50}

# v2
ref_buckley_v_2 = sio.loadmat('./buckley/Buckley_swc_100_vel_2.mat')
u2 = np.expand_dims(ref_buckley_v_2['usol'].flatten(), axis=-1)
outvar_v_2_numpy = {'u': u2}
ud_2 = np.zeros_like(X) + 2.0
invar_numpy_v2 = {'x': X, 't': T, 'ud': ud_2}

# v10
# ref_buckley_v_10 = sio.loadmat('./buckley/Buckley_M_2_vel_10.mat')
# u10 = np.expand_dims(ref_buckley_v_10['usol'].flatten(), axis=-1)
# outvar_v_10_numpy = {'u': u10}
# ud_10 = np.zeros_like(X) + 10.0
# invar_numpy_v10 = {'x': X, 't': T, 'ud': ud_10}

vel = Symbol('ud')
vel_ranges = (0.5, 2.0)
param_ranges = {vel: vel_ranges, t_symbol: (0, tf)}


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
  return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': 0.15, 'u__t': 0},
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

    # u 10p
    # val_v10p = Validation.from_numpy(invar_numpy_v10p, outvar_v_10p_numpy)
    # self.add(val_v10p, name='Val_v10p')

    # u 1
    val_v1 = Validation.from_numpy(invar_numpy_v1, outvar_v_1_numpy)
    self.add(val_v1, name='Val_v1')

    # u .50
    val_v50 = Validation.from_numpy(invar_numpy_v50, outvar_v_50_numpy)
    self.add(val_v50, name='Val_v50')

    # u 2
    val_v2 = Validation.from_numpy(invar_numpy_v2, outvar_v_2_numpy)
    self.add(val_v2, name='Val_v2')

    # u 10
    # val_v10 = Validation.from_numpy(invar_numpy_v10, outvar_v_10_numpy)
    # self.add(val_v10, name='Val_v10')


class BuckleyInference(InferenceDomain):
  def __init__(self, **config):
    super(BuckleyInference, self).__init__()
    # save entire domain
    # Normal distribution
    sample_vel = get_truncated_normal(mean=1, sd=0.3, low=0.5, upp=2).rvs(1000)
    # for i, velocity in enumerate(np.linspace(vel_ranges[0], vel_ranges[1], 10)):
    for i, velocity in enumerate(sample_vel):
      velocity = float(velocity)
      sampled_interior = geo.sample_interior(1024 * 5,
                                             bounds={x: (0, L)},
                                             param_ranges={t_symbol: (0, tf),
                                                           vel: velocity})
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
    # N(1, 0.3, 0.5,2)-> buckley_param_1619474515
    defaults.update({
      'network_dir': './network_checkpoint/buckley_param_{}'.format(int(time.time())),
      'initialize_network_dir': './network_checkpoint/buckley_param_1624420123',
      'rec_results_cpu': True,
      'rec_results_freq': 1,
      'max_steps': 1
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
