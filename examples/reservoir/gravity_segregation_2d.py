from sympy import Symbol, sin, exp, log, tanh, Function, Number
import numpy as np
import scipy.io as sio
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.sympy_utils.geometry_1d import Line1D
from simnet.pdes import PDES
from reservoir_equation import GravityEquation, ClosedBoundary, GravityEquationWeighted, GradMag, OpenBoundary
from simnet.controller import SimNetController

# params for domain
L1 = float(0)
L2 = float(1)

# define geometry
geo = Line1D(L1, L2)

# define sympy variables to parametrize domain curves
x = Symbol('x')

# define sympy varaibles to parametize domain curves
time_length = 1000
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

# define permeability numpy array
# mesh_x, mesh_y = np.meshgrid(np.linspace(0, 1, 512),
#                              np.linspace(0, 1, 512),
#                              indexing='ij')
permeability_invar = {}
mesh_x = np.linspace(L1, L2, 512)
permeability_invar['x'] = np.expand_dims(mesh_x.flatten(), axis=-1)
# permeability_invar['y'] = np.expand_dims(mesh_y.flatten(), axis=-1)
permeability_outvar = {}
permeability_outvar['c'] = 1 * (np.tanh(80 * permeability_invar['x']) -
                                np.tanh(
                                  80 * (permeability_invar['x'] - 1)) - 1)  # perm changes from 100 to 0 on 0.0 line.
permeability_outvar['c'][permeability_invar['x'] <= 0.05] = 0
permeability_outvar['c'][permeability_invar['x'] >= 0.95] = 0


class GravityTrain(TrainDomain):
  def __init__(self, **config):
    super(GravityTrain, self).__init__()
    # sympy variables
    x = Symbol('x')

    # train network to emulate wave speed
    permeability = BC.from_numpy(permeability_invar, permeability_outvar, batch_size=2048 // 2)
    self.add(permeability, "Permeability")

    # initial conditions -tanh(80*(x-0.5))/2+0.5
    IC = geo.interior_bc(outvar_sympy={'z': 0.5},
                         bounds={x: (L1, L2)},
                         batch_size_per_area=1000,
                         lambda_sympy={'lambda_z': 1.0},
                         param_ranges={t_symbol: 0.0})
    self.add(IC, name="IC")

    # boundary conditions
    edges = geo.boundary_bc(outvar_sympy={'open_boundary': 0},
                            batch_size_per_area=300,
                            lambda_sympy={'lambda_open_boundary': 1.0 * time_length},
                            param_ranges=time_range,
                            criteria=(x <= 0) | (x >= 1))
    self.add(edges, name="Edges")

    # interior
    interior = geo.interior_bc(outvar_sympy={'gravity_equation': 0},
                               bounds={x: (L1, L2)},
                               batch_size_per_area=5000,
                               lambda_sympy={'lambda_gravity_equation': 1.0},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


# class BuckleyVal(ValidationDomain):
#   def __init__(self, **config):
#     super(BuckleyVal, self).__init__()
#     # make validation data
#     deltaT = 0.01
#     deltaX = 0.01 * (L2 - L1) / 2.56
#     x = np.arange(L1, L2, deltaX)
#     t = np.arange(0, 1, deltaT)
#     X, T = np.meshgrid(x, t)
#     X = np.expand_dims(X.flatten(), axis=-1)
#     T = np.expand_dims(T.flatten(), axis=-1)
#     w = sio.loadmat('./buckley/Buckley_grav_Swc0_Sor_0_M_2_vert.mat')
#     u = np.expand_dims(w['usol'].flatten(), axis=-1)
#     invar_numpy = {'x': X, 't': T}
#     outvar_numpy = {'u': u}
#     val = Validation.from_numpy(invar_numpy, outvar_numpy)
#     self.add(val, name='Val')

class GravityInference(InferenceDomain):
  def __init__(self, **config):
    super(GravityInference, self).__init__()
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(1000,
                                     bounds={x: (0, 1)},
                                     param_ranges={t_symbol: float(specific_t)})
      inf = Inference(interior, ['z', 'c'])
      self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class GravitySolver(Solver):
  train_domain = GravityTrain
  # val_domain = GravityVal
  inference_domain = GravityInference

  def __init__(self, **config):
    super(GravitySolver, self).__init__(**config)

    # self.equations = (GravityEquation(u='z', c='c', dim=1, time=True).make_node(stop_gradients=['c'])
    #                   + OpenBoundary(u='z', c='c', dim=1, time=True).make_node())
    self.equations = (
      GravityEquationWeighted(u='z', c=1, dim=1, time=True).make_node(stop_gradients=['grad_magnitude_z'])
      + GradMag('z').make_node()
      + OpenBoundary(u='z', c='c', dim=1, time=True).make_node(stop_gradients=['c']))

    # self.equations = (GravityEquation(u='z', c='c', dim=1, time=True).make_node(stop_gradients=['c']))

    gravity_net = self.arch.make_node(name='gravity_net',
                                      inputs=['x', 't'],
                                      outputs=['z'])
    perm_net = self.arch.make_node(name='perm_net',
                                   inputs=['x'],
                                   outputs=['c'])

    self.nets = [gravity_net, perm_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity/simple_{}'.format(int(time.time())),
      'max_steps': 70000,
      'decay_steps': 500,
      'start_lr': 1e-3,
      'rec_results_cpu': True,
      'amp': True,
      'xla': True
    })


if __name__ == '__main__':
  ctr = SimNetController(GravitySolver)
  ctr.run()
