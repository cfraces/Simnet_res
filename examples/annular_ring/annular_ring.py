from sympy import Symbol, Eq
import tensorflow as tf

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain, MonitorDomain
from simnet.data import Validation, Monitor, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Circle
from simnet.sympy_utils.functions import parabola
from simnet.csv_utils.csv_rw import csv_to_dict
from simnet.PDES.navier_stokes import IntegralContinuity, NavierStokes
from simnet.controller import SimNetController

# params for domain
channel_length        = (-6.732, 6.732)
channel_width         = (-1.0, 1.0)
cylinder_center       = ( 0.0, 0.0)
outer_cylinder_radius = 2.0
inner_cylinder_radius = 1.0
inlet_vel             = 1.5

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
openfoam_var = csv_to_dict('openfoam/bend_finerInternal0.csv', mapping)
openfoam_var['x'] += channel_length[0] # center OpenFoam data
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'p']}

class AnnularRingTrain(TrainDomain):
  def __init__(self, **config):
    super(AnnularRingTrain, self).__init__()
    # inlet
    inlet_sympy = parabola(y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel)
    inlet = geo.boundary_bc(outvar_sympy={'u': inlet_sympy, 'v': 0},
                            batch_size_per_area=32,
                            criteria=Eq(x, channel_length[0]))
    self.add(inlet, name="Inlet")

    # outlet
    outlet = geo.boundary_bc(outvar_sympy={'p': 0},
                             batch_size_per_area=32,
                             criteria=Eq(x, channel_length[1]))
    self.add(outlet, name="Outlet")

    # noslip
    noslip = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                             batch_size_per_area=32,
                             criteria=(x>channel_length[0])&(x<channel_length[1]))
    self.add(noslip, name="NoSlip")

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                               bounds={x: channel_length,
                                       y: (-outer_cylinder_radius, outer_cylinder_radius)},
                               lambda_sympy={'lambda_continuity': geo.sdf,
                                             'lambda_momentum_x': geo.sdf,
                                             'lambda_momentum_y': geo.sdf},
                               batch_size_per_area=128)
    self.add(interior, name="Interior")

    # make integral continuity
    outlet = geo.boundary_bc(outvar_sympy={'integral_continuity': 2},
                             batch_size_per_area=128,
                             lambda_sympy={'lambda_integral_continuity': 0.1},
                             criteria=Eq(x, channel_length[1]))
    self.add(outlet, name="OutletContinuity")

class AnnularRingInference(InferenceDomain):
  def __init__(self, **config):
    super(AnnularRingInference, self).__init__()
    # save entire domain
    interior = Inference(openfoam_invar_numpy, ['u', 'v', 'p'])
    self.add(interior, name="Inference")

class AnnularRingVal(ValidationDomain):
  def __init__(self, **config):
    super(AnnularRingVal, self).__init__()
    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    self.add(val, name='Val')

class AnnularRingMonitor(MonitorDomain):
  def __init__(self, **config):
    super(AnnularRingMonitor, self).__init__()
    # metric for pressure drop and mass imbalance
    global_monitor = Monitor(geo.sample_interior(512, bounds={x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}),
                            {'mass_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['continuity'])),
                             'momentum_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['momentum_x'])+tf.abs(var['momentum_x']))})
    self.add(global_monitor, 'GlobalMonitor')

    # metric for force on inner sphere
    force = Monitor(inner_circle.sample_boundary(512),
                   {'force_x': lambda var: tf.reduce_sum(var['normal_x']*var['area']*var['p']),
                    'force_y': lambda var: tf.reduce_sum(var['normal_y']*var['area']*var['p'])})
    self.add(force, 'Force')

class ChipSolver(Solver):
  train_domain = AnnularRingTrain
  val_domain = AnnularRingVal
  inference_domain = AnnularRingInference
  monitor_domain = AnnularRingMonitor

  def __init__(self, **config):
    super(ChipSolver, self).__init__(**config)
    self.equations = (NavierStokes(nu=0.01, rho=1, dim=2, time=False).make_node()
                      + IntegralContinuity(dim=2).make_node())
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y'],
                                   outputs=['u', 'v', 'p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_annular_ring',
        'decay_steps': 2000,
        'max_steps': 200000
        })

if __name__ == '__main__':
  ctr = SimNetController(ChipSolver)
  ctr.run()

