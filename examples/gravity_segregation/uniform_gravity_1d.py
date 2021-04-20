# import SimNet library
from sympy import Symbol, Function, Number, sin, Heaviside, DiracDelta, exp
import numpy as np
import sys

# sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_1d import Line1D
from gravity_equation import GravitySegregationWeighted, GradMagSW
from simnet.controller import SimNetController
from simnet.architecture.siren import SirenArch
import scipy.io as sio
import time

# params for domain
L = float(1)

# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_length = 5.0
time_range = {t_symbol: (0, time_length)}


class GravitySegregationTrain(TrainDomain):
  def __init__(self, **config):
    super(GravitySegregationTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # initial conditions
    IC = geo.interior_bc(outvar_sympy={'sw': 0.5, 'sw__t': 0},
                         bounds={x: (0, L)},
                         batch_size_per_area=2000,
                         lambda_sympy={'lambda_sw': 1.0,
                                       'lambda_sw__t': 1.0},
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    # boundary conditions
    """BC = geo.boundary_bc(outvar_sympy={'closed_boundary_w': 0, 'closed_boundary_o': 0},
                         batch_size_per_area=2000,
                         lambda_sympy={'lambda_closed_boundary_w': geo.sdf,
                                       'lambda_closed_boundary_o': geo.sdf},
                         param_ranges=time_range,
                         criteria=any([x <= 0, x >= L])
                         )
    self.add(BC, name="BC")"""
    # Top wall
    topWall = geo.boundary_bc(outvar_sympy={'closed_boundary_o': 0},
                              batch_size_per_area=2000,
                              lambda_sympy={'lambda_closed_boundary_o': 1.0},
                              param_ranges=time_range,
                              criteria=x >= L)
    self.add(topWall, name="TopWall")

    # bottom wall
    bottomWall = geo.boundary_bc(outvar_sympy={'closed_boundary_w': 0},
                                 batch_size_per_area=2000,
                                 lambda_sympy={'lambda_closed_boundary_w': 1.0},
                                 param_ranges=time_range,
                                 criteria=x <= 0)
    self.add(bottomWall, name="BottomWall")

    # interior
    interior = geo.interior_bc(outvar_sympy={'gravity_segregation_o': 0, 'gravity_segregation': 0},
                               bounds={x: (0, L)},
                               batch_size_per_area=10000,
                               lambda_sympy={'lambda_gravity_segregation_o': geo.sdf,
                                             'lambda_gravity_segregation': geo.sdf},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


# class GravitySegregationVal(ValidationDomain):
#   def __init__(self, **config):
#     super(GravitySegregationVal, self).__init__()
#
#     # make validation data
#     deltaT = 0.01
#     deltaX = 0.01 / 2.56
#     x = np.arange(0, L, deltaX)
#     t = np.arange(0, L, deltaT)
#     X, T = np.meshgrid(x, t, indexing='ij')
#     X = np.expand_dims(X.flatten(), axis=-1)
#     T = np.expand_dims(T.flatten(), axis=-1)
#     w = sio.loadmat('./buckley/Buckley_x_Swc0_Sor_0_M_2.mat')
#     u = np.expand_dims(w['usol'].flatten(), axis=-1)
#     invar_numpy = {'x': X, 't': T}
#     # outvar_numpy = {'u': u, 'buckley_equation': np.zeros_like(u)}
#     outvar_numpy = {'u': u}
#     val = Validation.from_numpy(invar_numpy, outvar_numpy)
#     self.add(val, name='Val')


class GravitySegregationInference(InferenceDomain):
  def __init__(self, **config):
    super(GravitySegregationInference, self).__init__()
    x = Symbol('x')
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(5000,
                                     bounds={x: (0, L)},
                                     param_ranges={t_symbol: float(specific_t)})
      inf = Inference(interior, ['sw'])
      self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class GravitySegregationSolver(Solver):
  train_domain = GravitySegregationTrain
  inference_domain = GravitySegregationInference

  def __init__(self, **config):
    super(GravitySegregationSolver, self).__init__(**config)

    self.equations = (
      GravitySegregationWeighted(sw='sw', perm=1, dim=1, time=True).make_node(stop_gradients=['grad_magnitude_sw'])
      + GradMagSW('sw').make_node())
    gravity_segregation_net = self.arch.make_node(name='gravity_segregation_net',
                                                  inputs=['x', 't'],
                                                  outputs=['sw'])
    self.nets = [gravity_segregation_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity_1d/uniform_{}'.format(int(time.time())),
      'max_steps': 70000,
      'decay_steps': 500,
      'start_lr': 3e-4,
      'rec_results_cpu': True,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(GravitySegregationSolver)
  ctr.run()
