""" RNN helper functions and networks
"""

from simnet.config import str2bool
from simnet.arch import Arch
from simnet.variables import Variables
from simnet.tf_utils.activation_functions import set_nonlinearity
from simnet.tf_utils.layers import fc_layer, _variable

import tensorflow as tf
import numpy as np

def rnn_layer(inputs, states, hiddens, name, activation_fn=None):
  """Helper to create a RNN layer.
  Parameters
  ----------
  inputs : tf tensor
    input tensor into layer of shape `[N, ..., k]`.
  states: tf tensor
    recurrent state tensor of shape `[N, ..., hiddens]`.
  hiddens : int
    number of hidden units.
  name : str
    name space for weights.
  activation_fn : str
    function that is activation function. None
    becomes identitiy.

  Returns
  -------
  outputs : tf tensor
    output tensor of layer with shape `[N, ..., hiddens]`.
  """
  with tf.variable_scope(name):

    channel = inputs.get_shape()[-1]

    weights_x = _variable('weights_x',
                        shape=[channel, hiddens],
                        initializer=tf.contrib.layers.xavier_initializer())

    weights_h = _variable('weights_h',
                        shape=[hiddens, hiddens],
                        initializer=tf.contrib.layers.xavier_initializer())

    biases = _variable('biases',
                       shape=[hiddens],
                       initializer=tf.constant_initializer(0.0))

    recur = tf.add(tf.matmul(inputs, weights_x), tf.matmul(states, weights_h))

    outputs = tf.add(recur, biases, name=name)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def gru_layer(inputs, states, hiddens, name='gru'):
  """Helper to create a GRU layer.
  Parameters
  ----------
  inputs : tf tensor
    input tensor into layer of shape `[N, ..., k]`.
  states: tf tensor
    recurrent state tensor of shape `[N, ..., hiddens]`.
  hiddens : int
    number of hidden units.
  name : str
    name space for weights.

  Returns
  -------
  outputs : tf tensor
    output tensor of layer with shape `[N, ..., hiddens]`.
  """
  with tf.variable_scope(name):
    channel = inputs.get_shape()[-1]
    hiddens = hiddens

    weights_x = _variable('weights_x',
                      shape=[channel, hiddens],
                      initializer=tf.contrib.layers.xavier_initializer())

    ## reset gates
    weights_r = _variable('weights_r',
                      shape=[channel, hiddens],
                      initializer=tf.contrib.layers.xavier_initializer())
    biases_r = _variable('biases_r',
                       shape=[hiddens],
                       initializer=tf.constant_initializer(0.0))

    ## update gates
    weights_z = _variable('weights_z',
                      shape=[channel, hiddens],
                      initializer=tf.contrib.layers.xavier_initializer())
    biases_z = _variable('biases_z',
                       shape=[hiddens],
                       initializer=tf.constant_initializer(0.0))

    weights_h = _variable('weights_h',
                      shape=[hiddens, hiddens],
                      initializer=tf.contrib.layers.xavier_initializer())

    z = tf.sigmoid(tf.add(tf.matmul(inputs, weights_z), biases_z, name='update_gate'))
    r = tf.sigmoid(tf.add(tf.matmul(inputs, weights_r), biases_r, name='reset_gate'))
    h = tf.tanh(tf.matmul(inputs, weights_x) + tf.matmul(tf.multiply(states, r), weights_h))
    outputs = tf.multiply((1 - z), h) + tf.multiply(z, states)

    return outputs


class RNNArch(Arch):
  """
  Recurrent neural network architecture
  """

  def __init__(self, **config):
    super(RNNArch, self).__init__(**config)
    needed_config = RNNArch.process_config(config)
    self.__dict__.update(needed_config)
    self.print_configs()

  def set_time_steps(self, t_start, t_finish, nr_steps):
    self.t_start = t_start
    self.t_finish = t_finish
    self.nr_steps = nr_steps

  @classmethod
  def add_options(cls, group):
    group.add_argument('--activation_fn',
                       help='nonlinearity for network',
                       type=str,
                       default='swish')
    group.add_argument('--layer_size',
                       help='hidden layer size',
                       type=int,
                       default=512)
    group.add_argument('--nr_layers',
                       help='nr layers in net',
                       type=int,
                       default=6)
    group.add_argument('--skip_connections',
                       help='residual skip connections',
                       type=str2bool,
                       default=False)
    group.add_argument('--weight_norm',
                       help='residual skip connections',
                       type=str2bool,
                       default=True)
    group.add_argument('--nr_rnn_layers',
                       help='nr rnn layers in net',
                       type=int,
                       default=6)

  # very simple RNN made with fully connected layers
  def _rnn_template(self, inputs):
    ic_state = _variable('ic_state',
                     shape=[1, self.layer_size],
                     initializer=tf.constant_initializer(0.0))
    s = ic_state
    for i in range(self.nr_rnn_layers):
      seq_states = []
      for t in range(self.nr_steps):
        x = inputs[:, t]
        s = rnn_layer(x, s, 
                     self.layer_size,
                     activation_fn=tf.nn.tanh,
                     name='rnn_t'+str(t)+'l'+str(i))
        seq_states.append(s)

      seq_states = tf.stack(seq_states, axis=1)
      # inputs = tf.concat([seq_states, inputs], axis=-1)
      inputs = seq_states + inputs

    return inputs

  def _network_template(self, invars, out_names):
    # get input variables into tensor of shape [None, n]
    input_variable_tensor = Variables.to_tensor(Variables.subset(invars, ['x','y','z']))
    activation_fn = set_nonlinearity(self.activation_fn)

    
    # make RNN output continuous time
    time_var = tf.expand_dims(invars['t'], axis=1)
    time_grid = tf.reshape(tf.constant(np.linspace(self.t_start, self.t_finish, self.nr_steps), dtype=tf.float32), (1, self.nr_steps, 1))
    dt = (self.t_finish - self.t_start) / (self.nr_steps-1)
    time_distance = time_var - time_grid
    sharpen = 2.0
    time_bump = ((tf.tanh(sharpen*( time_distance + dt/2.0)/dt)+1.0)
               * (tf.tanh(sharpen*(-time_distance + dt/2.0)/dt)+1.0))/4.0
    time_bump = time_bump / tf.reduce_sum(time_bump, axis=1, keepdims=True)

    # RNN
    rnn_template = tf.make_template('rnn_net', self._rnn_template)
    seq_states = rnn_template(time_bump)
    seq_states =tf.reduce_sum(seq_states, axis=1)

    # network
    x = input_variable_tensor
    x = fc_layer(x,
                 self.layer_size,
                 activation_fn=activation_fn,
                 name='first_fc',
                 weight_norm=self.weight_norm)
    x_skip = 0.0
    x = tf.concat([seq_states, x], axis=-1)
    for i in range(self.nr_layers):
      # fc layer
      x = fc_layer(x,
                   self.layer_size,
                   activation_fn=activation_fn,
                   name='fc'+str(i),
                   weight_norm=self.weight_norm)
      # skip connection
      if (i % 2 == 0) and self.skip_connections:
        x_new = x + x_skip
        x_skip = x
        x = x_new
    # fc layer to output
    x = fc_layer(x, len(out_names), activation_fn=None, name='fc_final') # no weight norm on last layer

    return Variables.from_tensor(x, out_names)
  
    
class GRUArch(Arch):
  """
  Gated Recurrent Unit architecture
  """

  def __init__(self, **config):
    super(GRUArch, self).__init__(**config)
    needed_config = GRUArch.process_config(config)
    self.__dict__.update(needed_config)
    self.print_configs()

  def set_time_steps(self, t_start, t_finish, nr_steps):
    self.t_start = t_start
    self.t_finish = t_finish
    self.nr_steps = nr_steps

  @classmethod
  def add_options(cls, group):
    group.add_argument('--activation_fn',
                       help='nonlinearity for network',
                       type=str,
                       default='swish')
    group.add_argument('--layer_size',
                       help='hidden layer size',
                       type=int,
                       default=128)
    group.add_argument('--nr_layers',
                       help='nr layers in net',
                       type=int,
                       default=3)
    group.add_argument('--skip_connections',
                       help='residual skip connections',
                       type=str2bool,
                       default=False)
    group.add_argument('--weight_norm',
                       help='residual skip connections',
                       type=str2bool,
                       default=True)
    group.add_argument('--nr_rnn_layers',
                       help='nr rnn layers in net',
                       type=int,
                       default=3)

  # very simple RNN made with fully connected layers
  def _rnn_template(self, inputs):
    ic_state = _variable('ic_state',
                     shape=[1, self.layer_size],
                     initializer=tf.constant_initializer(0.0))
    s = ic_state
    for i in range(self.nr_rnn_layers):
      seq_states = []
      for t in range(self.nr_steps):
        x = inputs[:, t]
        s = gru_layer(x, s, 
                     self.layer_size,
                     name='gru_t'+str(t)+'l'+str(i))
        seq_states.append(s)
     
      seq_states = tf.stack(seq_states, axis=1)
      inputs = seq_states + inputs

    return inputs

  def _network_template(self, invars, out_names):
    # get input variables into tensor of shape [None, n]
    input_variable_tensor = Variables.to_tensor(Variables.subset(invars, ['x','y','z']))
    activation_fn = set_nonlinearity(self.activation_fn)

    
    # make RNN output continuous time
    time_var = tf.expand_dims(invars['t'], axis=1)
    time_grid = tf.reshape(tf.constant(np.linspace(self.t_start, self.t_finish, self.nr_steps), dtype=tf.float32), (1, self.nr_steps, 1))
    dt = (self.t_finish - self.t_start) / (self.nr_steps-1)
    time_distance = time_var - time_grid
    sharpen = 2.0
    time_bump = ((tf.tanh(sharpen*( time_distance + dt/2.0)/dt)+1.0)
               * (tf.tanh(sharpen*(-time_distance + dt/2.0)/dt)+1.0))/4.0
    time_bump = time_bump / tf.reduce_sum(time_bump, axis=1, keepdims=True)

    # RNN
    rnn_template = tf.make_template('rnn_net', self._rnn_template)
    seq_states = rnn_template(time_bump)
    seq_states =tf.reduce_sum(seq_states, axis=1)

    # network
    x = input_variable_tensor
    x = fc_layer(x,
                 self.layer_size,
                 activation_fn=activation_fn,
                 name='first_fc',
                 weight_norm=self.weight_norm)
    x_skip = 0.0
    x = tf.concat([seq_states, x], axis=-1)
    for i in range(self.nr_layers):
      # fc layer
      x = fc_layer(x,
                   self.layer_size,
                   activation_fn=activation_fn,
                   name='fc'+str(i),
                   weight_norm=self.weight_norm)
      # skip connection
      if (i % 2 == 0) and self.skip_connections:
        x_new = x + x_skip
        x_skip = x
        x = x_new
    # fc layer to output
    x = fc_layer(x, len(out_names), activation_fn=None, name='fc_final') # no weight norm on last layer

    return Variables.from_tensor(x, out_names)
    
