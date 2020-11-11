"""Wave equation
Reference: https://en.wikipedia.org/wiki/Wave_equation
"""

from sympy import Symbol, Function, Number

from simnet.pdes import PDES

class WaveEquation1D(PDES):
  """
  Wave equation 1D
  The equation is given as an example for implementing
  your own PDE. A more universal implementation of the
  wave equation can be found by 
  `from simnet.PDES.wave_equation import WaveEquation`.

  Parameters
  ==========
  c : float, string
      Wave speed coefficient. If a string then the
      wave speed is input into the equation.
  """

  name = 'WaveEquation1D'

  def __init__(self, c=1.0):
    # coordinates
    x = Symbol('x')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x':x,'t':t}

    # make u function
    u = Function('u')(*input_variables)

    # wave speed coefficient
    if type(c) is str:
      c = Function(c)(*input_variables)
    elif type(c) in [float, int]:
      c = Number(c)

    # set equations
    self.equations = {}
    self.equations['wave_equation'] = u.diff(t, 2) - (c**2*u.diff(x)).diff(x)
