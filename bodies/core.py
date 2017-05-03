import abc

import numpy as np
from scipy import integrate

from astropy import units as u
from astropy import constants as cnst
from astropy.coordinates import (CartesianRepresentation,
                                 SphericalRepresentation,
                                 UnitSphericalRepresentation,
                                 matrix_utilities)

TWOPI = 2*np.pi

class Body:
    __metaclass__ = abc.ABCMeta

    def __init__(self, surface_area, radiance, mass):
        self.radiance = radiance  # power / area / solid angle
        self.surface_area = surface_area
        self.mass = mass

        self.t = 0*u.second

        try:
            self.loc = CartesianRepresentation([0, 0, 0]*u.m)
        except AttributeError:
            pass  # this is OK, because it's when a read-only loc is present


    @property
    def intensity(self):
        return self.radiance * self.surface_area

    @abc.abstractproperty
    def luminosity(self):
        raise NotImplementedError()


class SphericalBody(Body):
    def __init__(self, radius, radiance, mass, obliquity=0*u.deg,
                 rotation_period=0*u.second):
        super().__init__(0, radiance, mass)
        self.radius = radius
        self.obliquity = obliquity
        try:
            self.rotation = 0*u.deg
        except AttributeError:
            pass # means it's a property
        self.rotation_period = rotation_period

    @property
    def surface_area(self):
        return 4 * np.pi * self.radius**2
    @surface_area.setter
    def surface_area(self, val):
        self.radius = (val/np.pi/4)**0.5

    @property
    def luminosity(self):
        return self.intensity * 4 * np.pi

    def surface_normal(self, lat, lon):
        usr = UnitSphericalRepresentation(lon=lon+self.rotation_angle, lat=lat)
        rotm = matrix_utilities.rotation_matrix(self.obliquity, axis='x')
        return usr.to_cartesian().transform(rotm)

    def surface_flux(self, lat, lon, source='parent'):
        if source == 'parent':
            source = self.parent_body

        source_vector = source.loc - self.loc
        dsource = source_vector.represent_as(SphericalRepresentation).distance
        source_normal = source_vector.represent_as(UnitSphericalRepresentation)

        fluxfraction = self.surface_normal(lat, lon).dot(source_normal)
        fluxfraction[fluxfraction < 0] = 0

        return fluxfraction * source.intensity * (dsource**-2 * u.sr)

    def average_surface_flux(self, lat, lon, source='parent',
                             timespan='rotation', t0=None,
                             nsamples=None):
        """
        Compute the average flux at a point on the surface

        `nsamples` == None means use the scipy integrate function instead of
        numerically estimating.
        """
        if timespan == 'rotation':
            timespan = self.rotation_period

        if t0 is None:
            t0 = self.t
            if not hasattr(t0, 'isscalar') or not t0.isscalar:
                raise ValueError("cannot set t0 because self.t is not a scalar")

        oldt = getattr(self, 't', 'not_present')
        try:
            if nsamples is None:
                def fint(tfrac):
                    self.t = t0 + tfrac * timespan
                    return self.surface_flux(lat, lon, source).value

                fluxu = self.surface_flux(lat, lon, source).unit
                return integrate.quad(fint, -0.5, 0.5)[0] * fluxu
            else:
                self.t = np.linspace(-0.5*timespan + t0, 0.5*timespan + t0, nsamples)
                flux = self.surface_flux(lat, lon, source)
                return flux.mean()
        finally:
            if oldt != 'not_present':
                self.t = oldt


    @property
    def rotation_angle(self):
        return TWOPI*(self.t/self.rotation_period)*u.radian


class OrbitingMixin:
    """
    Mixin class for a smaller body orbiting a larger one.

    Note that this assumes M_self << M_parent !
    """
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
    def orbital_period(self):
        return TWOPI*(self.semimajor**3/self.grav_mu)**0.5

    @property
    def theta(self):
        return (self.t / self.orbital_period)*TWOPI*u.radian

    @property
    def loc(self):
        sr = SphericalRepresentation(self.theta + self.phase0,
                                     self.inclination*np.cos(self.theta),
                                     self.r)
        return sr.to_cartesian() + self.parent_body.loc


class OrbitingSphericalBody(SphericalBody, OrbitingMixin):
    def __init__(self, radius, radiance, mass, obliquity, rotation_period,
                 semimajor, eccentricity, inclination, parent_body):
        SphericalBody.__init__(self, radius, radiance, mass,
                               obliquity=obliquity,
                               rotation_period=rotation_period)
        OrbitingMixin.__init__(self, semimajor, eccentricity, inclination,
                               parent_body)


def plot_bodies(bodies, t=0*u.second, ax=None, unit=None,
                alwaysscatter=False, **kwargs):
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
        if l.x.isscalar or alwaysscatter:
            res = ax.scatter3D(x, y, z, **kwargs)
        else:
            res = ax.plot3D(x, y, z, **kwargs)
    return ax, res
