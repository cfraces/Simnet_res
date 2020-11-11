"""Wave equation
Reference: https://en.wikipedia.org/wiki/Buckley-Leverett
"""

from sympy import Symbol, Function, Number, Max, Heaviside, DiracDelta, sqrt

from simnet.pdes import PDES
import numpy as np


class BuckleyEquation(PDES):
  """
  Buckley Leverett equation

  Parameters
  ==========
  u : str
      The dependent variable.
  c : float, Sympy Symbol/Expr, str
      Wave speed coefficient. If `c` is a str then it is
      converted to Sympy Function of form 'c(x,y,z,t)'.
      If 'c' is a Sympy Symbol or Expression then this
      is substituted into the equation.
  dim : int
      Dimension of the wave equation (1, 2, or 3). Default is 2.
  time : bool
      If time-dependent equations or not. Default is True.

  Examples
  ========
  >>> we = BuckleyEquation(c=0.8, dim=3)
  >>> we.pprint(preview=False)
    wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    buckley_equation: u__t - 0.8*u__x - 0.8*u__y - 0.8*u__z
  >>> we = BuckleyEquation(c='c', dim=2, time=False)
  >>> we.pprint(preview=False)
    buckley_equation: -c*u__x__x - c*u__y - c*c__x*u__x - c*c__y*u__y
  """

  name = 'BuckleyLeverettEquation'

  def __init__(self, u='u', c='c', dim=3, time=True):
    # set params
    self.u = u
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 'y': y, 'z': z, 't': t}
    if self.dim == 1:
      input_variables.pop('y')
      input_variables.pop('z')
    elif self.dim == 2:
      input_variables.pop('z')
    if not self.time:
      input_variables.pop('t')

    # Scalar function
    assert type(u) == str, "u needs to be string"
    u = Function(u)(*input_variables)

    # wave speed coefficient
    if type(c) is str:
      c = Function(c)(*input_variables)
    elif type(c) in [float, int]:
      c = Number(c)

    # set equations
    self.equations = {}
    f = Max(-(1.366025403514163 * u) * (Heaviside(u - 0.577357735773577) - 1)
            + 2 * (u ** 2) * Heaviside(u - 0.577357735773577) / (2 * (u) ** 2 + (u - 1) ** 2), 0)
    # f = (u - c) * (u - c) / ((u - c) ** 2 + (1 - u) * (1 - u) / 2)
    self.equations['buckley_equation'] = (u.diff(t)  # - 0.01 * (u.diff(x)).diff(x)
                                          + f.diff(x).replace(DiracDelta, lambda x: 0)
                                          + (c * u).diff(y)
                                          + (c * u).diff(z))


class BurgerEquation(PDES):
  """
  Burger equation

  Parameters
  ==========
  u : str
      The dependent variable.
  c : float, Sympy Symbol/Expr, str
      Wave speed coefficient. If `c` is a str then it is
      converted to Sympy Function of form 'c(x,y,z,t)'.
      If 'c' is a Sympy Symbol or Expression then this
      is substituted into the equation.
  dim : int
      Dimension of the wave equation (1, 2, or 3). Default is 2.
  time : bool
      If time-dependent equations or not. Default is True.

  Examples
  ========
  >>> we = BurgerEquation(c=0.8, dim=3)
  >>> we.pprint(preview=False)
    burger_equation: u__t - 0.8*x*u__x - 0.8*y*u__y - 0.8*z*u__z
  >>> we = BurgerEquation(c='c', dim=2, time=False)
  >>> we.pprint(preview=False)
    burger_equation: -c*u__x - c*u__y - c__x*u - c__y*u
  """

  name = 'BurgerEquation'

  def __init__(self, u='u', c='c', dim=3, time=True):
    # set params
    self.u = u
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 'y': y, 'z': z, 't': t}
    if self.dim == 1:
      input_variables.pop('y')
      input_variables.pop('z')
    elif self.dim == 2:
      input_variables.pop('z')
    if not self.time:
      input_variables.pop('t')

    # Scalar function
    assert type(u) == str, "u needs to be string"
    u = Function(u)(*input_variables)

    # wave speed coefficient
    if type(c) is str:
      c = Function(c)(*input_variables)
    elif type(c) in [float, int]:
      c = Number(c)

    # set equations
    self.equations = {}
    self.equations['burger_equation'] = (u.diff(t)
                                         - x * (c * u).diff(x)
                                         - y * (c * u).diff(y)
                                         - z * (c * u).diff(z))


class BuckleyGravity(PDES):
  """
  Buckley Leverett equation with gravity term

  Parameters
  ==========
  u : str
      The dependent variable.
  c : float, Sympy Symbol/Expr, str
      Wave speed coefficient. If `c` is a str then it is
      converted to Sympy Function of form 'c(x,y,z,t)'.
      If 'c' is a Sympy Symbol or Expression then this
      is substituted into the equation.
  dim : int
      Dimension of the wave equation (1, 2, or 3). Default is 2.
  time : bool
      If time-dependent equations or not. Default is True.

  Examples
  ========
  >>> we = BuckleyGravity(c=0.8, dim=3)
  >>> we.pprint(preview=False)
    wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    buckley_equation: u__t - 0.8*u__x - 0.8*u__y - 0.8*u__z
  >>> we = BuckleyGravity(c='c', dim=2, time=False)
  >>> we.pprint(preview=False)
    buckley_equation: -c*u__x__x - c*u__y - c*c__x*u__x - c*c__y*u__y
  """

  name = 'BuckleyLeverettGravity'

  def __init__(self, u='u', c='c', dim=3, time=True):
    # set params
    self.u = u
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 'y': y, 'z': z, 't': t}
    if self.dim == 1:
      input_variables.pop('y')
      input_variables.pop('z')
    elif self.dim == 2:
      input_variables.pop('z')
    if not self.time:
      input_variables.pop('t')

    # Scalar function
    assert type(u) == str, "u needs to be string"
    u = Function(u)(*input_variables)

    # wave speed coefficient
    if type(c) is str:
      c = Function(c)(*input_variables)
    elif type(c) in [float, int]:
      c = Number(c)

    # set equations
    self.equations = {}
    nw = 2
    no = 2
    # Residual oil saturation
    Sor = 0.
    # Residual water saturation
    Swc = 0.
    # End points
    krwmax = 1
    kromax = 1
    # Mobility ratio
    M = 10.

    # Angle dip (degrees)
    theta = 90
    # Gravity number
    Ng = -9

    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no

    # Fractional flow curve
    fw = lambda S: (1 - Ng * kro(S) * np.sin(theta * np.pi / 180)) / (1 + 1 / c * kro(S) / krw(S))

    f = fw(u)

    # fractional flow curve hull
    # f = Max(
    #   (1 - Heaviside(u - 0.214)) * (u * 2.793 / 0.214) +
    #   (Heaviside(u - 0.214) - Heaviside(u - 0.4142)) * fw(u) +
    #   Heaviside(u - 0.4142) * ((u - 1) * (1 - 3.4068) / (1 - 0.4142) + 1),
    #   0)

    self.equations['buckley_gravity'] = (u.diff(t)  # - 0.01 * (u.diff(x)).diff(x)
                                         + f.diff(x).replace(DiracDelta, lambda x: 0)
                                         + (c * u).diff(y)
                                         + (c * u).diff(z))

  class BuckleyEquationWeighted(PDES):
    name = 'WeightedBuckleyLeverettEquation'

    def __init__(self, u='u', c='c', dim=3, time=True, weighting='grad_magnitude_u'):
      # set params
      self.u = u
      self.dim = dim
      self.time = time
      self.weighting = weighting

      # coordinates
      x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

      # time
      t = Symbol('t')

      # make input variables
      input_variables = {'x': x, 'y': y, 'z': z, 't': t}
      if self.dim == 1:
        input_variables.pop('y')
        input_variables.pop('z')
      elif self.dim == 2:
        input_variables.pop('z')
      if not self.time:
        input_variables.pop('t')

      # Scalar function
      assert type(u) == str, "u needs to be string"
      u = Function(u)(*input_variables)

      # wave speed coefficient
      if type(c) is str:
        c = Function(c)(*input_variables)
      elif type(c) in [float, int]:
        c = Number(c)

      # set equations
      self.equations = {}

      # True f
      f = (u - c) * (u - c) / ((u - c) ** 2 + (1 - u) * (1 - u) / 2)

      self.equations['buckley_equation'] = ((u.diff(t) + f.diff(x))
                                            / (Function(self.weighting)(*input_variables) + 1))

  class GradMag(PDES):
    name = 'GradMag'

    def __init__(self, u='u'):
      # set params
      self.u = u

      # coordinates
      x = Symbol('x')

      # time
      t = Symbol('t')

      # make input variables
      input_variables = {'x': x, 't': t}

      # Scalar function
      assert type(u) == str, "u needs to be string"
      u = Function(u)(*input_variables)

      # set equations
      self.equations = {}
      self.equations['grad_magnitude_' + self.u] = u.diff(t) ** 2 + u.diff(x) ** 2

# f = Piecewise(((u - 0.2) * 1.7075, u <= 0.6597),
#               ((u - 0.2) ** 2 / ((u - 0.2) ** 2 + (1 - u) ** 2 / 2), True))
# f = (u - c) * (u - c) / ((u - c) ** 2 + (1 - u) * (1 - u) / 2)
# f = Max(-(1.7061530374765745 * u - 0.3412306074953149) * (Heaviside(u - 0.649965377440008) - 1) + 2 * (
#     u - 0.2) ** 2 * Heaviside(u - 0.649965377440008) / (
#           2 * (u - 0.2) ** 2 + (u - 1) ** 2), 0)

# f = -0.82744967 * tanh(-0.95476178 * tanh(2.74842793 * u - 1.54729133) - 0.38609312) + 0.33048552
