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

    conv = 9.1688e-15#9.1688e-17  # md to ft2

    # Fractional flow curve
    vo = lambda S: g * (rhow - rhoo) / (phi * muo) * c * conv * kro(S) / (1 + kro(S) * muw / (muo * krw(S)))

    v = vo(u)

    # set equations
    self.equations = {}
    self.equations['gravity_equation'] = (u.diff(t) + v.diff(x))


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


# define open boundary conditions
class OpenBoundary(PDES):
    """
  Open boundary condition for wave problems

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

    name = 'OpenBoundary'

    def __init__(self, u='u', c='c', dim=3, time=True):
        # set params
        self.u = u
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

        # normal
        normal_p = Symbol('normal_p')

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

        no = 2
        # Residual oil saturation
        Sor = 0.
        # Residual water saturation
        Swc = 0.
        # End points
        kromax = 1

        # Gravity number
        g = 32.2  # ft/s2

        # Relperms from Corey-Brooks
        Sstar = lambda S: (S - Swc) / (1 - Swc - Sor)
        kro = lambda S: kromax * (1 - Sstar(S)) ** no

        # Densities (lbm / scf)
        rhoo = 40  # Oil
        # Viscosities
        muo = 2e-4  # lb/ft-s
        phi = 0.25

        lam = kro(u) / (phi * muo)

        # set equations
        self.equations = {}
        self.equations['open_boundary'] = (u.diff(t)
                                           + lam * c * (u.diff(x) + rhoo * g)
                                           + lam * c * u.diff(y)
                                           + lam * c * u.diff(z))


class Darcy(PDES):
  """
  Compressible flow and transport in porous media equations

  Parameters
  ==========
  phi : float, Sympy Symbol/Expr, str
      The porosity. If `phi` is a str then it is
      converted to Sympy Function of form `phi(x,y,z,t)`.
      If `phi` is a Sympy Symbol or Expression then this
      is substituted into the equation. This allows for
      variable porosity.
  k : float, Sympy Symbol/Expr, str
      The permeability. If `k` is a str then it is
      converted to Sympy Function of form 'k(x,y,z,t)'.
      If 'k' is a Sympy Symbol or Expression then this
      is substituted into the equation to allow for
      variable permeability. Default is 1.
  dim : int
      Dimension of the flow and transport (2 or 3). Default is 2.
  time : bool
      If time-dependent equations or not. Default is True.

  Examples
  ========
  >>> da = Darcy(phi=0.01, k=100, dim=2)
  >>> da.pprint(preview=False)
    water: u__x + v__y
    momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
    momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
  >>> da = Darcy(nu='nu', rho=1, dim=2, time=False)
  >>> da.pprint(preview=False)
    continuity: u__x + v__y
    momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - nu__x*u__x - nu__y*u__y + p__x
    momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*v__x - nu__y*v__y + p__y
  """

  name = 'NavierStokes'

  def __init__(self, nu, rho=1, dim=3, time=True):
    # set params
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # time
    t = Symbol('t')

    # make input variables
    input_variables = {'x': x, 'y': y, 'z': z, 't': t}
    if self.dim == 2:
      input_variables.pop('z')
    if not self.time:
      input_variables.pop('t')

    # velocity componets
    u = Function('u')(*input_variables)
    v = Function('v')(*input_variables)
    if self.dim == 3:
      w = Function('w')(*input_variables)
    else:
      w = Number(0)

    # pressure
    p = Function('p')(*input_variables)

    # kinematic viscosity
    if isinstance(nu, str):
      nu = Function(nu)(*input_variables)
    elif isinstance(nu, (float, int)):
      nu = Number(nu)

    # density
    if isinstance(rho, str):
      rho = Function(rho)(*input_variables)
    elif isinstance(rho, (float, int)):
      rho = Number(rho)

    # dynamic viscosity
    mu = rho * nu

    # curl
    curl = Number(0) if rho.diff() == 0 else u.diff(x) + v.diff(y) + w.diff(z)

    # set equations
    self.equations = {}
    self.equations['continuity'] = rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
    self.equations['momentum_x'] = ((rho * u).diff(t)
                                    + (u * ((rho * u).diff(x)) + v * ((rho * u).diff(y)) + w * (
        (rho * u).diff(z)) + rho * u * (curl))
                                    + p.diff(x)
                                    - (-2 / 3 * mu * (curl)).diff(x)
                                    - (mu * u.diff(x)).diff(x)
                                    - (mu * u.diff(y)).diff(y)
                                    - (mu * u.diff(z)).diff(z)
                                    - (mu * (curl).diff(x)))
    self.equations['momentum_y'] = ((rho * v).diff(t)
                                    + (u * ((rho * v).diff(x)) + v * ((rho * v).diff(y)) + w * (
        (rho * v).diff(z)) + rho * v * (curl))
                                    + p.diff(y)
                                    - (-2 / 3 * mu * (curl)).diff(y)
                                    - (mu * v.diff(x)).diff(x)
                                    - (mu * v.diff(y)).diff(y)
                                    - (mu * v.diff(z)).diff(z)
                                    - (mu * (curl).diff(y)))
    self.equations['momentum_z'] = ((rho * w).diff(t)
                                    + (u * ((rho * w).diff(x)) + v * ((rho * w).diff(y)) + w * (
        (rho * w).diff(z)) + rho * w * (curl))
                                    + p.diff(z)
                                    - (-2 / 3 * mu * (curl)).diff(z)
                                    - (mu * w.diff(x)).diff(x)
                                    - (mu * w.diff(y)).diff(y)
                                    - (mu * w.diff(z)).diff(z)
                                    - (mu * (curl).diff(z)))

    if self.dim == 2:
      self.equations.pop('momentum_z')
