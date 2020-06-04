from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import sys, os
import julian
import time
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from scipy.integrate import ode, odeint
import datetime

#sys.path.insert(0, '../../../../Aero_Funcs')


"""
Very temporary, rarely used script I used to verify that my transformations were correct. Matches the TLE data and calculated observation
vectors with the actual azimuth and elevation data provided by LMS.

Since this is just for debugging and I quickly moved on, this script is meant to be run in the file with all of the associated data like julian dates etc.
It also requires that there be a .txt file with both lines of the TLE inside

Not the most elegant file but it got the job done.
"""
import Aero_Funcs as AF

LAT = 37.1348
LON = -12.2110
ALT = 684

def propagate_orbit(t, state, mu = 398600):
    pos = state[0:3]
    vel = state[3:6]

    d_pos = vel
    d_vel = -pos*mu/norm(pos)**3

    return hstack([d_pos, d_vel])



tlefile = open('tle.txt','r')
line1 = tlefile.readline().replace('\n','')
line2 = tlefile.readline().replace('\n','')
Satellite = twoline2rv(line1, line2, wgs84)

julian_dates = load('julian_dates0.npy')

timesteps = diff(julian_dates)*24*3600
times = (julian_dates - julian_dates[0])*24*3600


date0 = julian.from_jd(julian_dates[0])
sc_pos, sc_vel = Satellite.propagate(*date0.timetuple()[0:6])
state0 = hstack([sc_pos, sc_vel])


positions = []
for t in times:
    date_t = date0 + datetime.timedelta(seconds = t)
    sc_pos, _ = Satellite.propagate(*date_t.timetuple()[0:6])
    positions.append(sc_pos)

obs_vecs = []
sun_vecs = []
sat_poss = []
site_poss = []
range_vecs = []


for t, state in zip(times, positions):
    date = date0 + datetime.timedelta(seconds = t)
    lst = AF.local_sidereal_time(date, LON)
    #site = AF.observation_site(LAT, lst, ALT)
    sc_pos, sc_vel = Satellite.propagate(*date.timetuple()[0:6])
    #sc_pos = state[0:3]
    sc_pos = asarray(sc_pos)
    range_vec = sc_pos - site
    sun_vec = AF.vect_earth_to_sun(date)

    telescope_ecef = AF.lla_to_ecef(LAT, LON, ALT, geodetic = True)
    site = AF.Cz(lst)@telescope_ecef

    obs_vecs.append(site - sc_pos)
    sun_vecs.append(sun_vec)
    sat_poss.append(sc_pos)
    site_poss.append(site)
    range_vecs.append(range_vec)

obs_vecs = vstack(obs_vecs)
sun_vecs = vstack(sun_vecs)
sat_poss = vstack(sat_poss)
site_poss = vstack(site_poss)
range_vecs = vstack(range_vecs)

azimuths, elevations = AF.pass_az_el(site_poss, sat_poss)
true_az = load('azimuth.npy')
true_el = load('elevation.npy')

plt.scatter(true_az, true_el)
plt.plot(azimuths0, elevations)
plt.xlabel('Azimuth [deg]')
plt.ylabel('Elevation [deg]')
plt.show()
