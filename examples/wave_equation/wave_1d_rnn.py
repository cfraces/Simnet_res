from sympy import Symbol, sin
import numpy as np

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation
from simnet.sympy_utils.geometry_1d import Line1D
from simnet.controller import SimNetController

from wave_equation import WaveEquation1D
from rnn import RNNArch


# params for domain
L = float(np.pi)
seq_length = 12
time_length = seq_length*L

# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

class WaveTrain(TrainDomain):
  def __init__(self, **config):
    super(WaveTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    #initial conditions
    IC = geo.interior_bc(outvar_sympy={'u': sin(x), 'u__t': sin(x)},
                         bounds={x: (0, L)},
                         batch_size_per_area=100,
                         lambda_sympy={'lambda_u': 1.0,
                                       'lambda_u__t': 1.0},
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    #boundary conditions
    BC = geo.boundary_bc(outvar_sympy={'u': 0},
                         batch_size_per_area=100,
                         lambda_sympy={'lambda_u': 1.0},
                         param_ranges=time_range)
    self.add(BC, name="BC")

    # interior
    interior = geo.interior_bc(outvar_sympy={'wave_equation': 0},
                               bounds={x: (0, L)},
                               batch_size_per_area=400,
                               lambda_sympy={'lambda_wave_equation': 1.0},
                               param_ranges=time_range)
    self.add(interior, name="Interior")

class WaveVal(ValidationDomain):
  def __init__(self, **config):
    super(WaveVal, self).__init__()
    # make validation data
    deltaT = 0.10
    deltaX = 0.05
    x = np.arange(0, L, deltaX)
    t = np.arange(0, time_length, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = np.sin(X) * (np.cos(T) + np.sin(T))
    invar_numpy = {'x': X, 't': T}
    outvar_numpy = {'u': u}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val')

# Define neural network
class WaveSolver(Solver):
  train_domain = WaveTrain
  val_domain = WaveVal
  arch = RNNArch

  def __init__(self, **config):
    super(WaveSolver, self).__init__(**config)

    self.arch.set_time_steps(0, time_length, int(time_length/L))

    self.equations = WaveEquation1D(c=1.0).make_node()
    wave_net = self.arch.make_node(name='wave_net',
                                   inputs=['x', 't'],
                                   outputs=['u'])
    self.nets = [wave_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_wave_1d_rnn',
        'max_steps': 30000,
        'decay_steps': 300,
        })

if __name__ == '__main__':
  ctr = SimNetController(WaveSolver)
  ctr.run()
