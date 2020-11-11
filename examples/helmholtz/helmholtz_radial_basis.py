from sympy import Symbol, pi, sin

from simnet.solver import Solver
from simnet.csv_utils.csv_rw import csv_to_dict
from simnet.dataset import TrainDomain, ValidationDomain
from simnet.data import Validation
from simnet.sympy_utils.geometry_2d import Rectangle
from simnet.PDES.wave_equation import HelmholtzEquation
from simnet.controller import SimNetController
from simnet.architecture.radial_basis import RadialBasisArch

# params for domain
height = 2
width = 2

# define geometry
rec = Rectangle((-width/2, -height/2),
                (width/2, height/2))

# define sympy varaibles to parametize domain curves
x, y = Symbol('x'), Symbol('y')

class HemholtzTrain(TrainDomain):
  def __init__(self, **config):
    super(HemholtzTrain, self).__init__()

    #walls
    Wall = rec.boundary_bc(outvar_sympy={'u': 0},
                           batch_size_per_area=100)
    self.add(Wall, name="Wall")

    # interior
    interior = rec.interior_bc(outvar_sympy={'helmholtz': -(-((pi)**2)*sin(pi*x)*sin(4*pi*y)-((4*pi)**2)*sin(pi*x)*sin(4*pi*y)+1*sin(pi*x)*sin(4*pi*y))},
                               bounds={x: (-width/2, width/2), y: (-height/2, height/2)},
                               lambda_sympy={'lambda_helmholtz': rec.sdf},
                               batch_size_per_area=1000)
    self.add(interior, name="Interior")

# validation data
mapping = {'x': 'x', 'y': 'y', 'z': 'u'}
openfoam_var = csv_to_dict('validation/hemholtz.csv', mapping)
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u']}

class HemholtzVal(ValidationDomain):
  def __init__(self, **config):
    super(HemholtzVal, self).__init__()
    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    self.add(val, name='Val')

class HemholtzSolver(Solver):
  train_domain = HemholtzTrain
  val_domain = HemholtzVal
  arch = RadialBasisArch

  def __init__(self, **config):
    super(HemholtzSolver, self).__init__(**config)

    self.equations = HelmholtzEquation(u='u', k=1.0, dim=2).make_node()

    self.arch.set_bounds({'x': (-width/2, width/2), 'y': (-height/2, height/2)})
    wave_net = self.arch.make_node(name='wave_net',
                                   inputs=['x', 'y'],
                                   outputs=['u'])
    self.nets = [wave_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_hemholtz_radial_basis',
        'max_steps': 100000,
        'decay_steps': 1000,
        'nr_centers': 8192,
        'sigma': 0.1,
        })

if __name__ == '__main__':
  ctr = SimNetController(HemholtzSolver)
  ctr.run()
