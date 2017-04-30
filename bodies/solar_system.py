from . import SphericalBody, OrbitingSphericalBody


from astropy import time
from astropy import units as u
from astropy import constants as cnst
from astropy.coordinates import earth_orientation

sun = SphericalBody(cnst.R_sun, 1, cnst.M_sun)
# sneaky trick to use L_sun to compute luminance
sun.luminance = cnst.L_sun / sun.luminosity

_J2017 = time.Time('J2017')
ec = earth_orientation.eccentricity(_J2017.jd)
obliq = earth_orientation.obliquity(_J2017.jd)

earth = OrbitingSphericalBody(cnst.R_earth, 0, cnst.M_earth, obliq,
                              1*u.AU, ec, 0*u.deg, sun)
