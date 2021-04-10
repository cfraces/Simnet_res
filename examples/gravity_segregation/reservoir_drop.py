from sympy import Symbol, sin, exp, tanh, Function, Number
import numpy as np
import scipy.io as sio
import time

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
# from simnet.pdes import PDES
from gravity_equation import TwoPhaseFlow
from simnet.controller import SimNetController
from simnet.node import Node

# from simnet.architecture.fourier_net import FourierNetArch

# define geometry
geo = Rectangle((0, 0),
                (1, 1))

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# define time domain
time_length = 2.0
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

# define wave speed numpy array
mesh_x, mesh_y = np.meshgrid(np.linspace(0, 1, 512),
                             np.linspace(0, 1, 512),
                             indexing='ij')


class ReservoirTrain(TrainDomain):
  def __init__(self, **config):
    super(ReservoirTrain, self).__init__()
    # train network to emulate wave speed
    # permeability_speed = BC.from_numpy(permeability_invar, permeability_outvar, batch_size=2048)
    # self.add(permeability_speed, "Permeability")

    # initial conditions exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2))
    # (1+tanh(80*(x-0.5)))/2
    initial_conditions = geo.interior_bc(outvar_sympy={'z': exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2)),
                                                       'z__t': 0},
                                         batch_size_per_area=2048 // 2,
                                         lambda_sympy={'lambda_z': 100.0,
                                                       'lambda_z__t': 1.0},
                                         bounds={x: (0, 1), y: (0, 1)},
                                         param_ranges={t_symbol: 0})
    self.add(initial_conditions, name="IC")

    # boundary conditions
    edges = geo.boundary_bc(outvar_sympy={'closed_boundary_w': 0, 'closed_boundary_o': 0},
                            batch_size_per_area=2048 // 2,
                            lambda_sympy={'lambda_closed_boundary_w': 1.0 * time_length,
                                          'lambda_closed_boundary_o': 1.0 * time_length},
                            param_ranges=time_range)
    self.add(edges, name="Edges")

    # interior
    # TODO: Try removing countercurrent
    interior = geo.interior_bc(outvar_sympy={'darcy_equation': 0},
                               bounds={x: (0, 1), y: (0, 1)},
                               batch_size_per_area=2048 // 2,
                               lambda_sympy={'lambda_darcy_equation': geo.sdf},
                               param_ranges=time_range)
    self.add(interior, name="Interior")


class Reservoirnference(InferenceDomain):
  def __init__(self, **config):
    super(Reservoirnference, self).__init__()
    # inf data
    for i, specific_t in enumerate(np.linspace(0, time_length, 40)):
      interior = geo.sample_interior(5000,
                                     bounds={x: (0, 1), y: (0, 1)},
                                     param_ranges={t_symbol: float(specific_t)})
      inf = Inference(interior, ['z'])
      self.add(inf, "Inference_" + str(i).zfill(4))


# Define neural network
class ReservoirSolver(Solver):
  train_domain = ReservoirTrain
  inference_domain = Reservoirnference

  # arch = FourierNetArch

  def __init__(self, **config):
    super(ReservoirSolver, self).__init__(**config)

    # self.arch.set_frequencies(('full', [i/2 for i in range(0, 10)]))

    self.equations = (
      TwoPhaseFlow(sw='z', perm=1, dim=2, time=True).make_node()
    )
    # + OpenBoundary(sw='z', perm='perm', dim=2, time=True).make_node(stop_gradients=['perm', 'perm__x', 'perm__y'])

    reservoir_net = self.arch.make_node(name='reservoir_net',
                                        inputs=['x', 'y', 't'],
                                        outputs=['z'])
    # perm_net = self.arch.make_node(name='perm_net',
    #                                inputs=['x', 'y'],
    #                                outputs=['perm'])
    self.nets = [reservoir_net]

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity_2d/drop_{}'.format(int(time.time())),
      'max_steps': 200000,
      'decay_steps': 4000,
      'start_lr': 3e-4,
      'layer_size': 256,
    })


if __name__ == '__main__':
  ctr = SimNetController(ReservoirSolver)
  ctr.run()
