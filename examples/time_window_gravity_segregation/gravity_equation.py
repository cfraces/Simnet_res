"""Wave equation
Reference: https://en.wikipedia.org/wiki/Buckley-Leverett
"""

from sympy import Symbol, Function, Number, tanh, Piecewise, Max, Heaviside, DiracDelta, sqrt
import numpy as np

from simnet.pdes import PDES

class TwoPhaseFlow(PDES):
  """
  Darcy with gravity

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
  name = 'TwoPhaseFlow'

  def __init__(self, sw='sw', perm='perm', dim=3, time=True, added_diffusivity=1e-2):
    # set params
    self.sw = sw
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
    assert type(sw) == str, "u needs to be string"
    sw = Function(sw)(*input_variables)

    # permeability coefficient
    if type(perm) is str:
      perm = Function(perm)(*input_variables)
    elif type(perm) in [float, int]:
      perm = Number(perm)

    nw = 2
    no = 2

    # Residual oil saturation
    Sor = 0.0

    # Residual water saturation
    Swc = 0.0

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
    muo = 2e-4  # lb/ft-s
    muw = 6e-6
    conv = 9.1688e-8  # md to ft2
    fw = lambda S: krw(S) * kro(S) / (kro(S) + krw(S) * muo / muw)
    vw = g * (rhoo - rhow) / (phi * muw) * perm * conv * fw(sw)

    # Oil phase
    fo = lambda S: kro(S) * krw(S) / (krw(S) + kro(S) * muw / muo)
    vo = g * (rhow - rhoo) * perm * conv * fo(sw) / (muo * phi)

    # set equations
    self.equations = {}
    self.equations['darcy_equation'] = (sw.diff(t, 1)
                                      + vw.diff(x, 1)
                                      + added_diffusivity*sw.diff(x, 2))
                                      #+ added_diffusivity*vw.diff(x, 2))
    self.equations['counter_current'] = vw + vo
    self.equations['closed_boundary_w'] = vw
    self.equations['closed_boundary_o'] = vo
