# import SimNet library
from sympy import Symbol, sin, Heaviside, DiracDelta, sqrt, cos, ln, pi
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation
from simnet.sympy_utils.geometry_1d import Line1D
from hyperbolic_equation import BuckleyHeterogeneous
from simnet.controller import SimNetController
import scipy.io as sio
import time

# params for domain
L = float(1.0)

# define geometry
geo = Line1D(0, L)

# define sympy variables to parametrize time
# x = Symbol('x')
t_symbol = Symbol('t')
time_range = {t_symbol: (0, L)}

# Synpy variable to parametrize random velocities
# vel_ranges = (0.5, 2.0)

class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

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
                         param_ranges={t_symbol: 0.0,
                                       Symbol('rand_v_1'): (1e-5, 1),
                                       Symbol('rand_v_2'): (1e-5, 1)},
                         criteria=x <= 0)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_heterogeneous': 0},
                               bounds={x: (0.0, L)},
                               batch_size_per_area=5000,
                               lambda_sympy={'lambda_buckley_heterogeneous': 1.0},
                               param_ranges={t_symbol: 0.0,
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
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    w = sio.loadmat('./buckley/Buckley_Swc0_Sor_0_M_2.mat')
    u = np.expand_dims(w['usol'].flatten(), axis=-1)
    invar_numpy = {'x': X, 't': T}
    outvar_numpy = {'u': u}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


# Define neural network
class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal

  # arch = SirenArch

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.equations = BuckleyHeterogeneous(u='u', dim=1, time=True).make_node()
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't', 'rand_v_1', 'rand_v_2'],
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './network_checkpoint/buckley_het{}'.format(int(time.time())),
      'max_steps': 30000,
      'decay_steps': 300,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
