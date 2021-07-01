from sympy import Symbol, sin, exp, tanh, Function, Number, Or, Eq
import numpy as np
import scipy.io as sio
import time
import tensorflow as tf

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, BC, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
# from simnet.pdes import PDES
from simnet.PDES.navier_stokes import GradNormal
from gravity_equation import TwoPhaseFlow
from simnet.controller import SimNetController
from simnet.node import Node
from simnet.variables import Variables

from simnet.architecture.fourier_net import FourierNetArch

# define geometry
geo = Rectangle((0, 0),
                (1, 1))

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# define time domain
time_length = 0.01
t_symbol = Symbol('t')
time_range = {t_symbol: (0, time_length)}

# define wave speed numpy array
mesh_x, mesh_y = np.meshgrid(np.linspace(0, 1, 256),
                             np.linspace(0, 1, 256),
                             indexing='ij')
class ReservoirTrain(TrainDomain):
  def __init__(self, **config):
    super(ReservoirTrain, self).__init__()
    # boundary conditions
    # edges = geo.boundary_bc(outvar_sympy={'closed_boundary_w': 0, 'closed_boundary_o': 0},
    #                         batch_size_per_area=2048 // 2,
    #                         lambda_sympy={'lambda_closed_boundary_w': 1.0 * time_length,
    #                                       'lambda_closed_boundary_o': 1.0 * time_length},
    #                         param_ranges=time_range)
    # self.add(edges, name="Edges")
    #topWall = geo.boundary_bc(outvar_sympy={'z': 0},
    #topWall = geo.boundary_bc(outvar_sympy={'closed_boundary_w': 0,
    #                                        'closed_boundary_o': 0},
    #                          batch_size_per_area=4*2048,
    #                          lambda_sympy={'lambda_closed_boundary_w': 10000.0*time_length,
    #                                        'lambda_closed_boundary_o': 10000.0*time_length},
    #                          #lambda_sympy={'lambda_z': 10000.0*time_length},
    #                          param_ranges=time_range,
    #                          criteria=Or(Eq(x, 1), Eq(x, 0)))
    #self.add(topWall, name="TopWall_"+self.name)
    #topWall = geo.boundary_bc(outvar_sympy={'normal_gradient_z': 0},
    #                          batch_size_per_area=4*2048,
    #                          lambda_sympy={'lambda_normal_gradient_z': 1.0*time_length},
    #                          param_ranges=time_range,
    #                          criteria=Or(Eq(x, 1), Eq(x, 0)))
    #self.add(topWall, name="TopWall_"+self.name)

    # interior
    # TODO: Try removing countercurrent
    interior = geo.interior_bc(outvar_sympy={'darcy_equation': 0},
                               bounds={x: (0, 1), y: (0, 1)},
                               batch_size_per_area=4*2048,
                               lambda_sympy={'lambda_darcy_equation': 1.0*time_length},
                               param_ranges=time_range)
    self.add(interior, name="Interior_"+self.name)


class ICReservoirTrain(ReservoirTrain):
  name = 'initial_conditions'
  nr_iterations = 1

  def __init__(self, **config):
    super(ICReservoirTrain, self).__init__()

    # initial conditions exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2))
    # (1+tanh(80*(x-0.5)))/2
    initial_conditions = geo.interior_bc(outvar_sympy={'z': 1.0*exp(-200 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))},
                                                       #'z__t': 0},
                                         batch_size_per_area=4*2048,
                                         lambda_sympy={'lambda_z': 10000.0,
                                                       'lambda_z__t': 1.0},
                                         bounds={x: (0, 1), y: (0, 1)},
                                         param_ranges={t_symbol: 0})
    self.add(initial_conditions, name="IC_"+self.name)


class IterationReservoirTrain(ReservoirTrain):
  name = 'iterations'
  nr_iterations = 100

  def __init__(self, **config):
    super(IterationReservoirTrain, self).__init__()

    # initial conditions exp(-200 * ((x - 0.8) ** 2 + (y - 0.5) ** 2))
    # (1+tanh(80*(x-0.5)))/2
    initial_conditions = geo.interior_bc(outvar_sympy={'z_ic': 0},
                                         batch_size_per_area=4*2048,
                                         lambda_sympy={'lambda_z_ic': 10000.0},
                                         bounds={x: (0, 1), y: (0, 1)},
                                         param_ranges={t_symbol: 0})
    self.add(initial_conditions, name="IC_"+self.name)


class Reservoirnference(InferenceDomain):
  def __init__(self, **config):
    super(Reservoirnference, self).__init__()
    # inf data time
    mesh_x, mesh_y = np.meshgrid(np.linspace(0, 1, 256),
                                 np.linspace(0, 1, 256),
                                 indexing='ij')
    mesh_x = np.expand_dims(mesh_x.flatten(), axis=-1)
    mesh_y = np.expand_dims(mesh_y.flatten(), axis=-1)
    for i, specific_t in enumerate(np.linspace(0, time_length, 20)):
      interior = {'x': mesh_x,
                  'y': mesh_y,
                  't': np.full_like(mesh_x, specific_t)}
      inf = Inference(interior, ['z'])
      self.add(inf, "Inference_"+str(i).zfill(4))

# Define neural network
class ReservoirSolver(Solver):
  seq_train_domain = [ICReservoirTrain, IterationReservoirTrain]
  inference_domain = Reservoirnference
  arch = FourierNetArch

  def __init__(self, **config):
    super(ReservoirSolver, self).__init__(**config)

    self.arch.set_frequencies(('axis', [i/2.0 for i in range(0, 10)]))

    # make time window that moves
    self.time_window = tf.get_variable("time_window", [],
                                       initializer=tf.constant_initializer(0),
                                       trainable=False,
                                       dtype=tf.float32)
    def slide_time_window(invar):
      outvar = Variables() 
      outvar['shifted_t'] = 10.0*(invar['t'] + self.time_window)
      return outvar

    # make nodes for difference between velocity and the previous time block of velocity
    def make_ic_loss(invar):
      outvar = Variables() 
      outvar['z_ic'] = invar['z'] - tf.stop_gradient(invar['z_prev_step'])
      return outvar


    self.equations = (
        #TwoPhaseFlow(sw='z', perm=1, dim=2, time=True, added_diffusivity=1e-1).make_node()
        TwoPhaseFlow(sw='z', perm=tanh(10.0*x), dim=2, time=True, added_diffusivity=0).make_node()
      + GradNormal('z', dim=2, time=False).make_node()
      + [Node(make_ic_loss), Node(slide_time_window)]
    )

    reservoir_net = self.arch.make_node(name='reservoir_net',
                                        inputs=['x', 'y', 'shifted_t'],
                                        outputs=['z'])
    reservoir_net_prev_step = self.arch.make_node(name='reservoir_net_prev_step',
                                                  inputs=['x', 'y', 'shifted_t'],
                                                  outputs=['z_prev_step'])
    self.nets = [reservoir_net, reservoir_net_prev_step]

  def custom_update_op(self):
    # list of ops
    op_list = []

    # zero train step op
    global_step = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES) if 'global_step' in v.name][0]
    zero_step_op = tf.assign(global_step, tf.zeros_like(global_step))
    op_list.append(zero_step_op)

    # make update op that shifts time window
    update_time = tf.assign_add(self.time_window, time_length)
    op_list.append(update_time)

    # make update op that sets weights from_flow_net to flow_net_prev_step
    # f_e
    reservoir_net_variables = [v for v in tf.trainable_variables() if 'reservoir_net/' in v.name]
    reservoir_net_prev_step_variables = [v for v in tf.trainable_variables() if 'reservoir_net_prev_step' in v.name]
    for v, v_prev_step in zip(reservoir_net_variables, reservoir_net_prev_step_variables):
      op_list.append(tf.assign(v_prev_step, v))

    return tf.group(*op_list)

  @classmethod  # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
      'network_dir': './checkpoint_gravity_2d_seq',
      'max_steps': 40000,
      'decay_steps': 400,
      'start_lr': 3e-4,
      'layer_size': 256,
      'xla': True,
    })


if __name__ == '__main__':
  ctr = SimNetController(ReservoirSolver)
  ctr.run()
