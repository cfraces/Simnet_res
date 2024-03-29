# import SimNet library
from sympy import Symbol, sin, Heaviside, DiracDelta, exp
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation
from simnet.sympy_utils.geometry_1d import Line1D
from hyperbolic_equation import BuckleyEquation
from simnet.controller import SimNetController
from simnet.architecture.siren import SirenArch
import scipy.io as sio
import time

# params for domain
L = float(np.pi)

# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_range = {t_symbol: (0, 1)}

# define eps that weights the diffusion equation
eps = Symbol('eps')
eps_range = {eps: (0, 1)}  # exponentional range (e^(-2*eps-4.6)) This goes from (0.0100, 0.0013)


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': sin(x)},
                         bounds={x: (0, L)},
                         batch_size_per_area=2000,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges={**{t_symbol: 0.0}, **eps_range})
    self.add(IC, name="IC")

    # boundary conditions
    BC = geo.boundary_bc(outvar_sympy={'u': 0},
                         batch_size_per_area=2000,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges={**time_range, **eps_range},
                         criteria=x <= 0)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'buckley_equation': 0},
                               bounds={x: (0, L)},
                               batch_size_per_area=10000,
                               lambda_sympy={'lambda_buckley_equation': 1.0},
                               param_ranges={**time_range, **eps_range})
    self.add(interior, name="Interior")


class BuckleyVal(ValidationDomain):
  def __init__(self, **config):
    super(BuckleyVal, self).__init__()

    for i, specific_eps in enumerate(np.linspace(*eps_range[eps], 5)):
      # make validation data
      deltaT = 0.01
      deltaX = 0.01 * np.pi / 2.56
      x = np.arange(0, L, deltaX)
      t = np.arange(0, L, deltaT)
      X, T = np.meshgrid(x, t, indexing='ij')
      X = np.expand_dims(X.flatten(), axis=-1)
      T = np.expand_dims(T.flatten(), axis=-1)
      w = sio.loadmat('./data/Buckley_sin_Swc0_Sor_0_M_2.mat')
      u = np.expand_dims(w['usol'].flatten(), axis=-1)
      invar_numpy = {'x': X, 't': T, 'eps': np.full_like(X, specific_eps)}
      # outvar_numpy = {'u': u, 'buckley_equation': np.zeros_like(u)}
      outvar_numpy = {'u': u}
      val = Validation.from_numpy(invar_numpy, outvar_numpy)
      self.add(val, name='Val_' + str(i).zfill(3))


# Define neural network
class BuckleySolver(Solver):
  train_domain = BuckleyTrain
  val_domain = BuckleyVal

  # arch = SirenArch

  def __init__(self, **config):
    super(BuckleySolver, self).__init__(**config)

    self.equations = BuckleyEquation(u='u', c=0, dim=1, time=True, eps=exp(-2 * eps - 4.6)).make_node()
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't', 'eps'],  # network gives different solutions for different eps
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_linear/buckley_pd_{}'.format(int(time.time())),
      'max_steps': 50000,
      'decay_steps': 500,
      'start_lr': 1e-3,
      'rec_results_cpu': True,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
