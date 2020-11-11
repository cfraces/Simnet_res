from sympy import Symbol, Eq
import tensorflow as tf
import numpy as np

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain, MonitorDomain
from simnet.data import Validation, Monitor, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.sympy_utils.functions import parabola
from simnet.csv_utils.csv_rw import csv_to_dict
from simnet.PDES.navier_stokes import IntegralContinuity, NavierStokes
from simnet.controller import SimNetController

# params for domain
channel_length               = (-6.732, 6.732)
channel_width                = (-1.0, 1.0)
cylinder_center              = ( 0.0, 0.0)
outer_cylinder_radius        = 2.0
inner_cylinder_radius        = Symbol('r')
inner_cylinder_radius_ranges = (0.75, 1.0)
inlet_vel                    = 1.5
param_ranges                 = {inner_cylinder_radius: inner_cylinder_radius_ranges}

# define geometry
rec = Rectangle((channel_length[0], channel_width[0]),
                (channel_length[1], channel_width[1]))
outer_circle = Circle(cylinder_center, outer_cylinder_radius)
inner_circle = Circle((0, 0), inner_cylinder_radius)
geo = (rec + outer_circle) - inner_circle

# define sympy varaibles to parametize domain curves
x, y = Symbol('x'), Symbol('y')

# validation data
mapping = {'Points:0': 'x', 'Points:1': 'y',
           'U:0': 'u', 'U:1': 'v', 'p': 'p'}
# r1
openfoam_var_r1 = csv_to_dict('openfoam/bend_finerInternal0.csv', mapping)
openfoam_var_r1['x'] += channel_length[0] # center OpenFoam data
openfoam_var_r1['r'] = np.zeros_like(openfoam_var_r1['x']) + 1.0
openfoam_invar_r1_numpy = {key: value for key, value in openfoam_var_r1.items() if key in ['x','y','r']}
openfoam_outvar_r1_numpy = {key: value for key, value in openfoam_var_r1.items() if key in ['u','v','p']}
# r875
openfoam_var_r875 = csv_to_dict('openfoam/annularRing_r_0.8750.csv', mapping)
openfoam_var_r875['x'] += channel_length[0] # center OpenFoam data
openfoam_var_r875['r'] = np.zeros_like(openfoam_var_r875['x']) + 0.875
openfoam_invar_r875_numpy = {key: value for key, value in openfoam_var_r875.items() if key in ['x','y','r']}
openfoam_outvar_r875_numpy = {key: value for key, value in openfoam_var_r875.items() if key in ['u','v','p']}
# r75
openfoam_var_r75 = csv_to_dict('openfoam/annularRing_r_0.750.csv', mapping)
openfoam_var_r75['x'] += channel_length[0] # center OpenFoam data
openfoam_var_r75['r'] = np.zeros_like(openfoam_var_r75['x']) + 0.75
openfoam_invar_r75_numpy = {key: value for key, value in openfoam_var_r75.items() if key in ['x','y','r']}
openfoam_outvar_r75_numpy = {key: value for key, value in openfoam_var_r75.items() if key in ['u','v','p']}

class AnnularRingTrain(TrainDomain):
  def __init__(self, **config):
    super(AnnularRingTrain, self).__init__()
    # inlet
    inlet_sympy = parabola(y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel)
    inlet = geo.boundary_bc(outvar_sympy={'u': inlet_sympy, 'v': 0},
                            batch_size_per_area=32,
                            criteria=Eq(x, channel_length[0]),
                            param_ranges=param_ranges,
                            batch_per_epoch=4000)
    self.add(inlet, name="Inlet")

    # outlet
    outlet = geo.boundary_bc(outvar_sympy={'p': 0},
                             batch_size_per_area=32,
                             criteria=Eq(x, channel_length[1]),
                             param_ranges=param_ranges,
                             batch_per_epoch=4000)
    self.add(outlet, name="Outlet")

    # noslip
    noslip = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                             batch_size_per_area=32,
                             criteria=(x>channel_length[0])&(x<channel_length[1]),
                             param_ranges=param_ranges,
                             batch_per_epoch=4000)
    self.add(noslip, name="NoSlip")

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                               bounds={x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)},
                               lambda_sympy={'lambda_continuity': geo.sdf,
                                             'lambda_momentum_x': geo.sdf,
                                             'lambda_momentum_y': geo.sdf},
                               batch_size_per_area=128,
                               param_ranges=param_ranges,
                               batch_per_epoch=4000)
    self.add(interior, name="Interior")

    # make integral continuity
    for i, radius in enumerate(np.linspace(inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 10)):
      radius = float(radius)
      outlet = geo.boundary_bc(outvar_sympy={'integral_continuity': 2},
                               batch_size_per_area=128,
                               lambda_sympy={'lambda_integral_continuity': 0.1},
                               criteria=Eq(x, channel_length[1]),
                               param_ranges={inner_cylinder_radius: radius})
      self.add(outlet, name="OutletContinuity_"+str(i))

class AnnularRingInference(InferenceDomain):
  def __init__(self, **config):
    super(AnnularRingInference, self).__init__()
    # save entire domain
    for i, radius in enumerate(np.linspace(inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 10)):
      radius = float(radius)
      sampled_interior = geo.sample_interior(1024,
                                             bounds={x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)},
                                             param_ranges={inner_cylinder_radius: radius})
      interior = Inference(sampled_interior, ['u','v','p'])
      self.add(interior, name="Inference_"+str(i).zfill(5))

class AnnularRingVal(ValidationDomain):
  def __init__(self, **config):
    super(AnnularRingVal, self).__init__()
    # r 1
    val_r1 = Validation.from_numpy(openfoam_invar_r1_numpy, openfoam_outvar_r1_numpy)
    self.add(val_r1, name='Val_r1')

    # r 875
    val_r875 = Validation.from_numpy(openfoam_invar_r875_numpy, openfoam_outvar_r875_numpy)
    self.add(val_r875, name='Val_r875')

    # r 75
    val_r75 = Validation.from_numpy(openfoam_invar_r75_numpy, openfoam_outvar_r75_numpy)
    self.add(val_r75, name='Val_r75')

class AnnularRingMonitor(MonitorDomain):
  def __init__(self, **config):
    super(AnnularRingMonitor, self).__init__()
    # metric for pressure drop and mass imbalance
    global_monitor = Monitor(geo.sample_interior(512, bounds={x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}, param_ranges=param_ranges),
                            {'mass_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['continuity'])),
                             'momentum_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['momentum_x'])+tf.abs(var['momentum_x']))})
    self.add(global_monitor, 'GlobalMonitor')

    # metric for force on inner sphere
    for i, radius in enumerate(np.linspace(inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 3)):
      radius = float(radius)
      force = Monitor(inner_circle.sample_boundary(512, param_ranges={inner_cylinder_radius: radius}),
                     {'force_x': lambda var: tf.reduce_sum(var['normal_x']*var['area']*var['p']),
                      'force_y': lambda var: tf.reduce_sum(var['normal_y']*var['area']*var['p'])})
      self.add(force, 'Force_'+str(i))

class ChipSolver(Solver):
  train_domain = AnnularRingTrain
  val_domain = AnnularRingVal
  inference_domain = AnnularRingInference
  monitor_domain = AnnularRingMonitor

  def __init__(self, **config):
    super(ChipSolver, self).__init__(**config)
    self.equations = (NavierStokes(nu=0.01, rho=1, dim=2, time=False).make_node()
                      + IntegralContinuity(dim=2).make_node())
    flow_net  = self.arch.make_node(name='flow_net',
                                    inputs=['x','y','r'],
                                    outputs=['u','v','p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_annular_ring_parameterized',
        'decay_steps': 5000,
        'max_steps': 500000
        })

if __name__ == '__main__':
  ctr = SimNetController(ChipSolver)
  ctr.run()
