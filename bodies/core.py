import abc

import numpy as np

from astropy import units as u
from astropy import constants as cnst
from astropy.coordinates import CartesianRepresentation, SphericalRepresentation


class Body:
    __metaclass__ = abc.ABCMeta

    def __init__(self, surface_area, luminance, mass, obliquity=0*u.deg):
        self.luminance = luminance  # power / area / solid angle
        self.surface_area = surface_area
        self.mass = mass
        self.obliquity = obliquity

        try:
            self.loc = CartesianRepresentation([0, 0, 0]*u.m)
        except AttributeError:
            pass  # this is OK, because it's when a read-only loc is present


    @property
    def intensity(self):
        return self.luminance * self.surface_area

    @abc.abstractproperty
    def luminosity(self):
        raise NotImplementedError()


class SphericalBody(Body):
    def __init__(self, radius, luminance, mass, obliquity=0*u.deg):
        super().__init__(0, luminance, mass, obliquity)
        self.radius = radius

    @property
    def surface_area(self):
        return 4 * np.pi * self.radius**2
    @surface_area.setter
    def surface_area(self, val):
        self.radius = (val/np.pi/4)**0.5

    @property
    def luminosity(self):
        return self.intensity * 4 * np.pi


class OrbitingMixin:
    def __init__(self, semimajor, eccentricity, inclination, parent_body,
                 phase0=0*u.deg):
        self.semimajor = semimajor
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.parent_body = parent_body
        self.phase0 = phase0

        self.t = 0*u.second

    @property
    def grav_mu(self):
        # assuming parent mass >> self mass.  Below is ~right if otherwise
        #return cnst.G * (self.mass + self.parent_body.mass)
        return cnst.G * self.parent_body.mass

    @property
    def r(self):
        numer = self.semimajor * (1 - self.eccentricity**2)
        denom = 1 + self.eccentricity * np.cos(self.theta)
        return numer/denom

    @property
    def theta(self):
        periodo2pi = (self.semimajor**3/self.grav_mu)**0.5
        return (self.t / periodo2pi)*u.radian

    @property
    def loc(self):
        sr = SphericalRepresentation(self.theta + self.phase0,
                                     self.inclination*np.cos(self.theta),
                                     self.r)
        return sr.to_cartesian() + self.parent_body.loc


class OrbitingSphericalBody(SphericalBody, OrbitingMixin):
    def __init__(self, radius, luminance, mass, obliquity, semimajor, eccentricity,
                 inclination, parent_body):
        SphericalBody.__init__(self, radius, luminance, mass)
        OrbitingMixin.__init__(self, semimajor, eccentricity, inclination, parent_body)


def plot_bodies(bodies, t=0*u.second, ax=None, unit=None, **kwargs):
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(projection='3d')

    for b in bodies:
        b.t = t
        l = b.loc
        if unit is not None:
            x = l.x.to(unit)
            y = l.y.to(unit)
            z = l.z.to(unit)
        else:
            x = l.x
            y = l.y
            z = l.z
        if l.x.isscalar:
            ax.scatter3D(x, y, z, **kwargs)
        else:
            ax.plot3D(b.loc.x, b.loc.y, b.loc.z, **kwargs)
    return ax
