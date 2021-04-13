# import SimNet library
from sympy import Symbol, sin, Heaviside, DiracDelta
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation
from simnet.sympy_utils.geometry_1d import Line1D
from hyperbolic_equation import BuckleyEquation
from simnet.controller import SimNetController
import scipy.io as sio
import time
from rnn import GRUArch

# params for domain
L = float(15)

# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
# TODO: Try seq_length=1
seq_length = 3
time_length = seq_length * L
time_range = {t_symbol: (0, time_length)}


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': 0, 'u__t': 0},
                         bounds={x: (0, L)},
                         batch_size_per_area=1024,
                         lambda_sympy={'lambda_u': 1.0,
                                       'lambda_u__t': 1.0},
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    # boundary conditions
    BC = geo.boundary_bc(outvar_sympy={'u': 1},
                         batch_size_per_area=1024,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges=time_range,
                         criteria=x <= 0)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_equation': 0},
                               bounds={x: (0, L)},
                               batch_size_per_area=4096,
                               lambda_sympy={'lambda_buckley_equation': 1.0},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


class BuckleyVal(ValidationDomain):
  def __init__(self, **config):
    super(BuckleyVal, self).__init__()
    # make validation data
    deltaT = 0.01 * time_length
    deltaX = L * 0.01 / 2.56
    x = np.arange(0, L, deltaX)
    t = np.arange(0, time_length, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    w = sio.loadmat('./buckley/Buckley_Swc0_Sor_0_M_2_T_10.mat')
    u = np.expand_dims(w['usol'].flatten(), axis=-1)
    invar_numpy = {'x': X, 't': T}
    outvar_numpy = {'u': u}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')


# Define neural network
class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal
  arch = GRUArch

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.arch.set_time_steps(0, time_length, int(time_length / L))

    self.equations = BuckleyEquation(u='u', c=0, dim=1, time=True).make_node()
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't'],
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './network_checkpoint/buckley_long_gru_{}'.format(int(time.time())),
      'max_steps': 30000,
      'decay_steps': 300,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
