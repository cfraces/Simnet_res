from sympy import Symbol, Eq
import tensorflow as tf

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, MonitorDomain
from simnet.data import Validation, Monitor
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.csv_utils.csv_rw import csv_to_dict
from simnet.PDES.navier_stokes import NavierStokes
from simnet.controller import SimNetController

# Parameters for the domain
channel_length = (-10.0, 30.0)
channel_width = (-10.0, 10.0)
cylinder_center = (0.0, 0.0)
cylinder_radius = 0.5
inlet_vel = 1.0

# define geometry
rec = Rectangle((channel_length[0], channel_width[0]),
                (channel_length[1], channel_width[1]))
cylinder = Circle(cylinder_center, cylinder_radius)
geo = rec - cylinder

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# validation data
mapping = {'Points:0': 'x', 'Points:1': 'y', 'U:0': 'u', 'U:1': 'v', 'p': 'p'}
openfoam_var = csv_to_dict('openfoam/cylinder_nu_0.020.csv', mapping)
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'p']}

class Cylinder2DTrain(TrainDomain):
  def __init__(self, **config):
    super(Cylinder2DTrain, self).__init__()

    # inlet
    inlet = rec.boundary_bc(outvar_sympy={'u': 1, 'v': 0},
                            batch_size_per_area=64,
                            criteria=x < channel_length[1])  # place on all edges except outlet
    self.add(inlet, name="Inlet")

    # outlet
    outlet = rec.boundary_bc(outvar_sympy={'p': 0},
                             batch_size_per_area=64,
                             criteria=Eq(x, channel_length[1]))
    self.add(outlet, name="Outlet")

    # noslip
    noslip = cylinder.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                  batch_size_per_area=128)
    self.add(noslip, name="CylinderNS")

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0,
                                             'momentum_x': 0,
                                             'momentum_y': 0},
                               bounds={x: channel_length,
                                       y: channel_width},
                               batch_size_per_area=16)
    self.add(interior, name="Interior")

class Cylinder2DVal(ValidationDomain):
  def __init__(self, **config):
    super(Cylinder2DVal, self).__init__()
    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    self.add(val, name='Val')

class Cylinder2DMonitor(MonitorDomain):
  def __init__(self, **config):
    super(Cylinder2DMonitor, self).__init__()

    # residual monitor
    global_monitor = Monitor(geo.sample_interior(100, bounds={x: channel_length, y: channel_width}),
                             {'pressure_drop': lambda var: tf.reduce_max(var['p']),
                              'mass_imbalance': lambda var: tf.reduce_sum(var['area'] * tf.abs(var['continuity'])),
                              'momentum_imbalance': lambda var: tf.reduce_sum(var['area'] * tf.abs(var['momentum_x']) + tf.abs(var['momentum_y']))})
    self.add(global_monitor, 'GlobalMonitor')

    #metric for force on inner circle
    force = Monitor(cylinder.sample_boundary(1000),
                    {'force_x': lambda var: tf.reduce_sum(var['normal_x'] * var['area'] * var['p']),
                     'force_y': lambda var: tf.reduce_sum(var['normal_y'] * var['area'] * var['p'])})
    self.add(force, 'Force')

class ChipSolver(Solver):
  train_domain = Cylinder2DTrain
  val_domain = Cylinder2DVal
  monitor_domain = Cylinder2DMonitor

  def __init__(self, **config):
    super(ChipSolver, self).__init__(**config)

    self.equations = NavierStokes(nu=0.02, rho=1, dim=2, time=False).make_node()
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y'],
                                   outputs=['u', 'v', 'p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_cylinder2D',
        'max_steps': 200000,
        'decay_steps': 2000,
    })

if __name__ == '__main__':
  ctr = SimNetController(ChipSolver)
  ctr.run()
