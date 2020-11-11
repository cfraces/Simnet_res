# import SimNet library
from sympy import Symbol, Eq, sin, Heaviside, DiracDelta, Abs
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation
from simnet.sympy_utils.geometry_1d import Line1D
from hyperbolic_equation import BuckleyGravity, GradMag, BuckleyGravityWeighted
from simnet.controller import SimNetController
import scipy.io as sio
import time

# params for domain
L1 = float(0)
L2 = float(2.45)

# define geometry
geo = Line1D(L1, L2)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_range = {t_symbol: (0, 1)}


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': 0},
                         bounds={x: (L1, L2)},
                         batch_size_per_area=5000,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    # boundary conditions
    BC = geo.boundary_bc(outvar_sympy={'u': 0.5898},
                         batch_size_per_area=1000,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges=time_range,
                         criteria=x <= 0)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_gravity': 0},
                               bounds={x: (L1, L2)},
                               batch_size_per_area=5000,
                               lambda_sympy={'lambda_buckley_gravity': 1.0},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


class BuckleyVal(ValidationDomain):
  def __init__(self, **config):
    super(BuckleyVal, self).__init__()
    # make validation data
    deltaT = 0.01
    deltaX = 0.01 * (L2 - L1) / 2.56
    x = np.arange(L1, L2, deltaX)
    t = np.arange(0, 1, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    w = sio.loadmat('./buckley/Buckley_grav_Swc0_Sor_0_M_2_vert.mat')
    u = np.expand_dims(w['usol'].flatten(), axis=-1)
    invar_numpy = {'x': X, 't': T}
    outvar_numpy = {'u': u}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


# Define neural network
class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.equations = BuckleyGravity(u='u', c=2, dim=1, time=True).make_node()

    # self.equations = (BuckleyGravityWeighted(u='u', c=2, dim=1, time=True).make_node(stop_gradients=['grad_magnitude_u'])
    #                   + GradMag('u').make_node())
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't'],
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity/buckley{}'.format(int(time.time())),
      'max_steps': 30000,
      'decay_steps': 300,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
