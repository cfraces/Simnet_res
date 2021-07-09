"""Wave equation
Reference: https://en.wikipedia.org/wiki/Buckley-Leverett
"""

from sympy import Symbol, Function, Number, tanh, Piecewise, Max, Heaviside, DiracDelta, sqrt, ln, pi, cos
import numpy as np

from simnet.pdes import PDES


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

  def __init__(self, u='u', c='c', dim=3, time=True, eps=1e-2):
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

    # Piecewise f
    # swc = 0.0
    # sor = 0.
    # sinit = 0.
    # M = 2
    # tangent = [0.568821882188219, 0.751580500446855]
    #
    # f = Max(-(tangent[1] / (tangent[0] - sinit) * (u - sinit)) * (Heaviside(u - tangent[0]) - 1) + Heaviside(
    #   u - tangent[0]) * (u - swc) ** 2 / ((u - swc) ** 2 + ((1 - u - sor) ** 2) / M), 0)

    f = Max(-(1.366025403514163 * u) * (Heaviside(u - 0.577357735773577) - 1)
            + 2 * (u ** 2) * Heaviside(u - 0.577357735773577) / (2 * (u) ** 2 + (u - 1) ** 2), 0)

    # True f
    # f = (u - c) * (u - c) / ((u - c) ** 2 + (1 - u) * (1 - u) / 2)

    self.equations['buckley_equation'] = (u.diff(t)
                                          + f.diff(x).replace(DiracDelta, lambda x: 0)
                                          + (c * u).diff(y)
                                          + (c * u).diff(z))


class BuckleyHeterogeneous(PDES):
  """
  Buckley Leverett equation with heterogeneities

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
  >>> we = BuckleyHeterogeneousn(c=0.8, dim=3)
  >>> we.pprint(preview=False)
    wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    buckley_equation: u__t - 0.8*u__x - 0.8*u__y - 0.8*u__z
  >>> we = BuckleyHeterogeneousn(c='c', dim=2, time=False)
  >>> we.pprint(preview=False)
    buckley_equation: -c*u__x__x - c*u__y - c*c__x*u__x - c*c__y*u__y
  """

  name = 'BuckleyLeverettHeterogeneous'

  def __init__(self, u='u', dim=3, time=True, weighting='grad_magnitude_u'):
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
    # if type(rand_v_1) is str:
    #   c = Function(c)(*input_variables)
    # elif type(c) in [float, int]:
    #   c = Number(c)
    rand_v_1 = Symbol("rand_v_1")
    rand_v_2 = Symbol("rand_v_2")
    v_d = ((-2 * ln(rand_v_1)) ** 0.5) * cos(2 * np.pi * rand_v_2) / 5 + 1  # + x

    # set equations
    self.equations = {}

    # Piecewise f
    # swc = 0.0
    # sor = 0.
    # sinit = 0.
    # M = 2
    # tangent = [0.568821882188219, 0.751580500446855]
    #
    # f = Max(-(tangent[1] / (tangent[0] - sinit) * (u - sinit)) * (Heaviside(u - tangent[0]) - 1) + Heaviside(
    #   u - tangent[0]) * (u - swc) ** 2 / ((u - swc) ** 2 + ((1 - u - sor) ** 2) / M), 0)

    # f = Max(-(1.366025403514163 * u) * (Heaviside(u - 0.577357735773577) - 1)
    #         + 2 * (u ** 2) * Heaviside(u - 0.577357735773577) / (2 * (u) ** 2 + (u - 1) ** 2), 0)

    f = u * u / (u ** 2 + (1 - u) * (1 - u) / 2)

    # Heterogenous
    # s_tangent = 0.577357735773577
    # f_tangent = 0.788685333982125 * v_d
    # f = Max(-(f_tangent * u / s_tangent) * (Heaviside(u - s_tangent) - 1)
    #         + 2 * v_d * (u ** 2) * Heaviside(u - s_tangent) / (2 * (u) ** 2 + (u - 1) ** 2), 0)

    # self.equations['buckley_heterogeneous'] = u.diff(t) + f.diff(x).replace(DiracDelta, lambda x: 0)

    self.equations['buckley_heterogeneous'] = ((u.diff(t) + v_d * f.diff(x))
                                               / (Function(self.weighting)(*input_variables) + 1))


class BuckleyEquationParam(PDES):
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

  name = 'BuckleyLeverettEquationParam'

  def __init__(self, u='u', c='c', dim=3, time=True, eps=1e-2):
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

    # Piecewise f
    # swc = 0.1
    # sor = 0.05
    # sinit = 0.15
    # M = 2
    # tangent = [0.568821882188219, 0.751580500446855]
    #
    # f = Max(-(tangent[1] / (tangent[0] - sinit) * (u - sinit)) * (Heaviside(u - tangent[0]) - 1) + Heaviside(
    #   u - tangent[0]) * (u - swc) ** 2 / ((u - swc) ** 2 + ((1 - u - sor) ** 2) / M), 0)

    # f = Max(-(1.7075*u - 0.3415)*(Heaviside(u - 0.6597) - 1) + 2*(u - 0.2)**2*Heaviside(u - 0.6597)/(2*(u - 0.2)**2
    # + (u - 1)**2), 0)

    f = Max(-(1.366025403514163 * u) * (Heaviside(u - 0.577357735773577) - 1)
            + 2 * (u ** 2) * Heaviside(u - 0.577357735773577) / (2 * (u) ** 2 + (u - 1) ** 2), 0)

    # True f
    # f = (u - c) * (u - c) / ((u - c) ** 2 + (1 - u) * (1 - u) / 2)
    # f1 = (s - swc) ** 2 / ((s - swc) ** 2 + (1 - s - sor) ** 2 / M)

    self.equations['buckley_equation_param'] = (u.diff(t)
                                                + c * f.diff(x).replace(DiracDelta, lambda x: 0)
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


class BuckleyEquationWeightedParam(PDES):
  name = 'WeightedBuckleyLeverettEquationParam'

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

    self.equations['buckley_equation_param'] = ((u.diff(t) + c * f.diff(x))
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
    Ng = -3

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


class BuckleyGravityWeighted(PDES):
  name = 'WeightedGravityBuckley'

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

    # Angle dip (degrees)
    theta = 90
    # Gravity number
    Ng = -3

    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no

    # Fractional flow curve gravirt
    fw = lambda S: (1 - Ng * kro(S) * np.sin(theta * np.pi / 180)) / (1 + 1 / c * kro(S) / krw(S))

    f = fw(u)

    self.equations['buckley_gravity'] = ((u.diff(t) + f.diff(x))
                                         / (Function(self.weighting)(*input_variables) + 1))


class GravitySegregationWeighted(PDES):
  name = 'GravitySegregation'

  def __init__(self, sw='sw', perm='perm', dim=3, time=True, weighting='grad_magnitude_sw'):
    # set params
    self.sw = sw
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
    assert type(sw) == str, "sw needs to be string"
    sw = Function(sw)(*input_variables)

    # wave speed coefficient
    if type(perm) is str:
      perm = Function(perm)(*input_variables)
    elif type(perm) in [float, int]:
      perm = Number(perm)

    # set equations
    self.equations = {}

    # True f
    nw = 2
    no = 2
    # Residual oil saturation
    Sor = 0.05
    # Residual water saturation
    Swc = 0.1
    # End points
    krwmax = 1
    kromax = 1
    # Gravity
    g = 32.2  # ft/s2   9.81 / (0.304)
    phi = 0.25
    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no
    # Densities (lbm / scf)
    rhoo = 40  # Oil
    rhow = 62.238  # Water
    # Viscosities
    muo = 1  # 2e-4  # lb/ft-s
    muw = 5  # 6e-6
    conv = 9.1688e-8  # md to ft2
    fw = lambda S: krw(S) * kro(S) / (kro(S) + krw(S) * muo / muw)
    f = fw(sw)
    vw = g * (rhoo - rhow) / (phi * muw) * perm * conv * fw(sw)
    # Oil phase
    fo = lambda S: kro(S) / (1 + kro(S) * muw / (krw(S) * muo))
    vo = g * (rhow - rhoo) * perm * conv * fo(sw) / (muo * phi)
    # set equations
    self.equations = {}
    self.equations['gravity_segregation'] = ((sw.diff(t, 1)
                                              + vw.diff(x, 1))
                                             / (Function(self.weighting)(*input_variables) + 1))
    self.equations['closed_boundary_w'] = vw
    self.equations['closed_boundary_o'] = vo


class GradMagSW(PDES):
  name = 'GradMagSw'

  def __init__(self, sw='sw'):
    # set params
    self.sw = sw

    # coordinates
    x = Symbol('x')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 't': t}

    # Scalar function
    assert type(sw) == str, "sw needs to be string"
    sw = Function(sw)(*input_variables)

    # set equations
    self.equations = {}
    self.equations['grad_magnitude_' + self.sw] = sw.diff(t) ** 2 + sw.diff(x) ** 2
