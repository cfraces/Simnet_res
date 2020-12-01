"""Equations related to Reservoir Simulation
"""

from sympy import Symbol, sin, exp, log, tanh, Function, Number
import numpy as np

from simnet.pdes import PDES
from simnet.node import Node
from simnet.variables import Variables


class GravityEquation(PDES):
  """
Gravity Segregation equation

Parameters
==========
u : str
    The dependent variable.
c : float, Sympy Symbol/Expr, str
    permeability coefficient. If `c` is a str then it is
    converted to Sympy Function of form 'c(x,y,z,t)'.
    If 'c' is a Sympy Symbol or Expression then this
    is substituted into the equation.
dim : int
    Dimension of the wave equation (1, 2, or 3). Default is 2.
time : bool
    If time-dependent equations or not. Default is True.
"""
  name = 'GravityEquation'

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
    nw = 2
    no = 2
    # Residual oil saturation
    Sor = 0.
    # Residual water saturation
    Swc = 0.
    # End points
    krwmax = 1
    kromax = 1

    # Porosity
    phi = 0.25

    # Gravity number
    beta = 1 / 144  # Conversion unit factor
    alpha = 0.001127
    g = 32.2  # ft/s2   9.81 / (0.304)

    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no

    # Densities (lbm / scf)
    rhoo = 40  # Oil
    rhow = 62.238  # Water
    # Viscosities
    muo = 2e-4  # lb/ft-s
    muw = 6e-6

    conv = 9.1688e-8  # md to ft2
    const = -4e-4  # -1.1504e-07

    # Fractional flow curve
    fw = lambda S: krw(S) / (1 + krw(S) * muo / (muw * kro(S)))
    f = fw(u)
    vw = lambda S: g * (rhoo - rhow) / (phi * muw) * c * conv * fw(S)
    v = vw(u)

    # set equations
    self.equations = {}
    self.equations['gravity_equation'] = (u.diff(t, 1)
                                       + v.diff(x, 1)
                                       + f.diff(y, 1)
                                       + f.diff(z, 1))


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


# define closed boundary conditions
class ClosedBoundary(PDES):
  """
  Closed boundary condition for wave problems

  Parameters
  ==========
  u : str
      The dependent variable.
  c : float, Sympy Symbol/Expr, str
      permeability. If `c` is a str then it is
      converted to Sympy Function of form 'c(x,y,z,t)'.
      If 'c' is a Sympy Symbol or Expression then this
      is substituted into the equation.
  dim : int
      Dimension of the equation (1, 2, or 3). Default is 2.
  time : bool
      If time-dependent equations or not. Default is True.
  """

  name = 'ClosedBoundary'

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

      # Porosity
      phi = 0.25

      # Gravity number
      beta = 1 / 144  # Conversion unit factor
      alpha = 0.001127
      g = 32.2  # ft/s2   9.81 / (0.304)

      # Relperms from Corey-Brooks
      Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
      krw = lambda S: krwmax * Sstar(S) ** nw
      kro = lambda S: kromax * (1 - Sstar(S)) ** no

      # Densities (lbm / scf)
      rhoo = 40  # Oil
      rhow = 62.238  # Water
      # Viscosities
      muo = 2e-4  # lb/ft-s
      muw = 6e-6

      conv = 9.1688e-8  # md to ft2
      const = -4e-4  # -1.1504e-07

      # Fractional flow curve
      fw = lambda S: krw(S) / (1 + krw(S) * muo / (muw * kro(S)))
      f = fw(u)
      vw = lambda S: g * (rhoo - rhow) / (phi * muw) * c * conv * fw(S)
      v = vw(u)

      # set equations
      self.equations = {}
      self.equations['closed_boundary'] = (u.diff(t, 1)
                                         + v.diff(x, 1)
                                         + f.diff(y, 1)
                                         + f.diff(z, 1))


# define open boundary conditions
class OpenBoundary(PDES):
  """
  Open boundary condition for wave problems
  Ref: http://hplgit.github.io/wavebc/doc/pub/._wavebc_cyborg002.html

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
  """

  name = 'OpenBoundary'

  def __init__(self, u='u', c='c', dim=3, time=True):
    # set params
    self.u = u
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # normal
    normal_x, normal_y, normal_z = Symbol('normal_x'), Symbol('normal_y'), Symbol('normal_z')

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

    nw = 2
    no = 2
    # Residual oil saturation
    Sor = 0.
    # Residual water saturation
    Swc = 0.
    # End points
    krwmax = 1
    kromax = 1
    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no
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
    muo = 2e-4  # lb/ft-s
    muw = 6e-6
    conv = 9.1688e-8  # md to ft2
    fw = lambda S: krw(S) / (1 + krw(S) * muo / (muw * kro(S)))
    f = fw(u)
    vw = lambda S: g * (rhoo - rhow) / (phi * muw) * c * conv * fw(S)
    v = vw(u)
    # set equations
    self.equations = {}
    self.equations['open_boundary'] = v**2 + f**2
    # (u.diff(t, 1)
    #  + v.diff(x, 1)
    #  + f.diff(y, 1)
    #  + f.diff(z, 1))

class GravityEquationWeighted(PDES):
  """
Gravity Segregation equation

Parameters
==========
u : str
    The dependent variable.
c : float, Sympy Symbol/Expr, str
    permeability coefficient. If `c` is a str then it is
    converted to Sympy Function of form 'c(x,y,z,t)'.
    If 'c' is a Sympy Symbol or Expression then this
    is substituted into the equation.
dim : int
    Dimension of the wave equation (1, 2, or 3). Default is 2.
time : bool
    If time-dependent equations or not. Default is True.
"""
  name = 'GravityEquationWeighted'

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
    nw = 2
    no = 2
    # Residual oil saturation
    Sor = 0.
    # Residual water saturation
    Swc = 0.
    # End points
    krwmax = 1
    kromax = 1
    # Relperms from Corey-Brooks
    Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
    krw = lambda S: krwmax * Sstar(S) ** nw
    kro = lambda S: kromax * (1 - Sstar(S)) ** no
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
    muo = 2e-4  # lb/ft-s
    muw = 6e-6
    conv = 9.1688e-8  # md to ft2
    const = -4e-4  # -1.1504e-07
    # Fractional flow curve
    fw = lambda S: krw(S) / (1 + krw(S) * muo / (muw * kro(S)))
    f = fw(u)
    vw = lambda S: g * (rhoo - rhow) / (phi * muw) * c * conv * fw(S)
    v = vw(u)

    # set equations
    self.equations = {}
    self.equations['gravity_equation'] = (u.diff(t, 1)
                                          + v.diff(x, 1)
                                          + f.diff(y, 1)
                                          + f.diff(z, 1)
                                          / (Function(self.weighting)(*input_variables) + 1))
    # self.equations['gravity_equation'] = ((u.diff(t) + v.diff(x))
    #                                       / (Function(self.weighting)(*input_variables) + 1))
