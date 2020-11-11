from sympy import Symbol
import numpy as np
import tensorflow as tf

from simnet.solver import Solver
from simnet.dataset import TrainDomain, InferenceDomain, MonitorDomain
from simnet.data import BC, Inference, Monitor
from simnet.sympy_utils.geometry_1d import Line1D
from simnet.controller import SimNetController

from wave_equation import WaveEquation1D

# params for domain
L = float(np.pi)

# define geometry
geo = Line1D(0, L)

# define sympy varaibles to parametize domain curves
t_symbol = Symbol('t')
time_range = {t_symbol: (0, 2*L)}

batch_size = 8192
deltaT = 0.01
deltaX = 0.01
x = np.arange(0, L, deltaX)
t = np.arange(0, 2*L, deltaT)
X, T = np.meshgrid(x, t)
X = np.expand_dims(X.flatten(), axis=-1)
T = np.expand_dims(T.flatten(), axis=-1)
u = np.sin(X) * (np.cos(T) + np.sin(T))
invar_numpy = {'x': X, 't': T}
outvar_numpy = {'u': u}
outvar_numpy['wave_equation'] = np.zeros_like(outvar_numpy['u'])

class WaveTrain(TrainDomain):
  def __init__(self, **config):
    super(WaveTrain, self).__init__()

    interior = BC.from_numpy(invar_numpy, outvar_numpy, batch_size=batch_size)
    self.add(interior, name="Interior")

class WaveInference(InferenceDomain):
  def __init__(self, **config):
    super(WaveInference, self).__init__()
    infer = Inference(invar_numpy, ['u', 'c'])
    self.add(infer, name='Inference')

class WaveMonitor(MonitorDomain):
  def __init__(self, **config):
    super(WaveMonitor, self).__init__()
    # metric for pressure drop and mass imbalance
    global_monitor = Monitor(invar_numpy, {'average_c': lambda var: tf.reduce_mean(tf.abs(var['c']))})
    self.add(global_monitor, 'GlobalMonitor')

class WaveSolver(Solver):
  train_domain = WaveTrain
  inference_domain = WaveInference
  monitor_domain = WaveMonitor

  def __init__(self, **config):
    super(WaveSolver, self).__init__(**config)

    self.equations = WaveEquation1D(c='c').make_node(stop_gradients=['u__x', 'u__x__x', 'u__t__t'])
    wave_net = self.arch.make_node(name='wave_net',
                                   inputs=['x', 't'],
                                   outputs=['u'])
    c_net = self.arch.make_node(name='c_net',
                                inputs=['x'],
                                outputs=['c'])
    self.nets = [wave_net, c_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_wave_inverse',
        'max_steps': 100000,
        'decay_steps': 1000,
        })

if __name__ == '__main__':
  ctr = SimNetController(WaveSolver)
  ctr.run()
