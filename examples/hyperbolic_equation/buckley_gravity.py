# import SimNet library
from sympy import Symbol, Eq, sin, Heaviside, DiracDelta, Abs, Piecewise
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation, BC
from simnet.sympy_utils.geometry_1d import Line1D
from hyperbolic_equation import BuckleyGravity, GradMag, BuckleyGravityWeighted
from simnet.controller import SimNetController
import scipy.io as sio
import time

# params for domain
L1 = float(-0.55)
L2 = float(2.45)

# define geometry
geo = Line1D(L1, L2)
#point = Point1D(0.0) # with New version of SimNet
point = Line1D(0.0, 1.0) 

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_range = {t_symbol: (0, 1)}


class BuckleyTrain(TrainDomain):
  def __init__(self, **config):
    super(BuckleyTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # boundary left
    IC = geo.boundary_bc(outvar_sympy={'u': 1.0},
                         batch_size_per_area=200,
                         lambda_sympy={'lambda_u': 10.0},
                         param_ranges=time_range,
                         criteria=Eq(x, -0.55))
    self.add(IC, name="IC_left")

    # boundary right
    IC = geo.boundary_bc(outvar_sympy={'u': 0.0},
                         batch_size_per_area=200,
                         lambda_sympy={'lambda_u': 10.0},
                         param_ranges=time_range,
                         criteria=Eq(x, 2.45))

    """
    # Tried initial conditions as a linear function like in the first time step of the data 
    IC = geo.interior_bc(outvar_sympy={'u': 1.0 - (x - L1)/(L2-L1)},
                         bounds={x: (L1, L2)},
                         batch_size_per_area=200,
                         lambda_sympy={'lambda_u': 10.0},
                         param_ranges=time_range)
    self.add(IC, name="IC")
    """

    # Tried Piecwise linear initial conditions as a linear function that interh
    IC = geo.interior_bc(outvar_sympy={'u': Piecewise(((1-0.5895)/L1 * x + 0.5898, x < 0), (-0.5895/L2 * x + 0.5898, True))},
                         bounds={x: (L1, L2)},
                         batch_size_per_area=200,
                         lambda_sympy={'lambda_u': 10.0},
                         param_ranges=time_range)
    self.add(IC, name="IC")


    """
    # Tried initial conditions from a few time steps of the data 
    deltaT = 0.01
    deltaX = 0.01 * (L2 - L1) / 2.56
    X = np.arange(L1, L2, deltaX)
    T = np.arange(0, 1, deltaT)
    X, T = np.meshgrid(X, T, indexing='ij')
    X = np.expand_dims(X[:,10:20].flatten(), axis=-1)
    T = np.expand_dims(T[:,10:20].flatten(), axis=-1)
    w = sio.loadmat('./buckley/Buckley_grav_Swc0_Sor_0_M_2_vert.mat')
    u = np.expand_dims(w['usol'][:,10:20].flatten(), axis=-1)
    invar_numpy = {'x': X, 't': T} # train on second initial conditions from data
    outvar_numpy = {'u': u}
    IC = BC.from_numpy(invar_numpy, outvar_numpy, batch_size=512)
    self.add(IC, name="IC")

    # Tried with no IC and only setting the source u=0.5898 at x=0 and t=0
    source = point.boundary_bc(outvar_sympy={'u': 0.5898},
                               batch_size_per_area=200,
                               lambda_sympy={'lambda_u': 10.0},
                               param_ranges={t_symbol: 0.0},
                               criteria=Eq(x, 0))
    self.add(source, name="Source")
    """

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
    X, T = np.meshgrid(x, t, indexing='ij')
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

    #self.equations = BuckleyGravity(u='u', c=2, dim=1, time=True).make_node()
    self.equations = (BuckleyGravityWeighted(u='u', c=2, dim=1, time=True).make_node(stop_gradients=['grad_magnitude_u'])
                      + GradMag('u').make_node())
    buckley_net = self.arch.make_node(name='buckley_net',
                                      inputs=['x', 't'],
                                      outputs=['u'])
    self.nets = [buckley_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity/buckley_w{}'.format(int(time.time())),
      'max_steps': 30000,
      'decay_steps': 300,
      'start_lr': 3e-4,
      'rec_results_cpu': True,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(BuckleySolver)
  ctr.run()
