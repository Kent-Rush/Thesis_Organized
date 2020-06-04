from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from scipy.spatial.transform import Rotation as R
#import pymap3d as pm
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys
import os


sys.path.insert(0, 'Support_Files')

import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF
import Reflection_Funcs as RF

import json
#Luca was here

"""
This file generates simulated data. Requires a .json file as a parameter in the command line
"""

PASS_TIME = 600 #seconds

def propagate_mrps(t, state, inertia):
    mrps = state[0:3]
    omega = state[3:6]
    #inertia = diag([1, state[6], state[7]])

    #dcm_eci2body = CF.mrp2dcm(mrps).T
    d_mrps = CF.mrp_derivative(mrps, omega)
    d_omega = inv(inertia)@(-cross(omega, inertia@omega))

    #print(d_mrps, d_omega)
    return hstack([d_mrps, d_omega])

def propagate_orbit(t, state, mu = 398600):
    pos = state[0:3]
    vel = state[3:6]

    d_pos = vel
    d_vel = -pos*mu/norm(pos)**3

    return hstack([d_pos, d_vel])

def states_to_lightcurve(obs_vecs, sun_vecs, attitudes, spacecraft_geometry, Exposure_Time):
    '''
    @param times list of dates at which each observation was taken
    @param obs_vecs array of vectors from the observer to the spacecraft
    @param sun_vecs array of vectors from the sun to the spacecraft
    @param attitudes array of modified rodriguez parameters describing the spacecraft attitude
    @param spacecraft_geometry a Spacecraft Geometry object describing the spacecraft geometry
    '''

    lightcurve = []

    iters = len(obs_vecs)
    count = 0
    for obs_vec, sun_vec, attitude in zip(obs_vecs, sun_vecs, attitudes):

        dcm_body2eci = CF.mrp2dcm(attitude)
        sun_vec_body = dcm_body2eci.T@sun_vec
        obs_vec_body = dcm_body2eci.T@obs_vec
        power = spacecraft_geometry.calc_reflected_power(obs_vec_body, sun_vec_body, Exposure_Time)
        lightcurve.append(power)

        count += 1
        loading_bar(count/iters, 'Simulating Lightcurve')

    lightcurve = hstack(lightcurve)

    return lightcurve

def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} Loading:[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')


Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
if len(sys.argv) == 3:
    num_passes = int(sys.argv[2])
else:
    num_passes = 12

Satellite = twoline2rv(Simulation_Configuration['TLE Line1'],
                       Simulation_Configuration['TLE Line2'], wgs84)
Lat = Simulation_Configuration['Observation Site Lattitude']
Lon = Simulation_Configuration['Observation Site East Longitude']
Alt = Simulation_Configuration['Observation Site Altitude']
DT = Simulation_Configuration['Data Rate']
Inertia = asarray(Simulation_Configuration['Inertia Matrix'])
Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])
Noise_STD = Simulation_Configuration['Sensor STD']
Directory = Simulation_Configuration['Directory']
Exposure_Time = Simulation_Configuration['Exposure Time']
Real_Data = Simulation_Configuration['Real Data']

date = Satellite.epoch - datetime.timedelta(hours = 1)

if not os.path.exists(Directory):
    os.makedirs(Directory)

pass_list = []
mid_pass_flag = False
max_el = 0
while len(pass_list) < num_passes:
    date += datetime.timedelta(seconds = 1)
    lst = AF.local_sidereal_time(date, Lon)
    site = AF.observation_site(Lat, lst, Alt)
    sc_pos, sc_vel = Satellite.propagate(*date.timetuple()[0:6])
    sc_pos = asarray(sc_pos)
    range_vec = sc_pos - site
    sun_vec = AF.vect_earth_to_sun(date)

    illuminated  = AF.shadow(sc_pos, sun_vec)
    above_horizon = dot(range_vec, site) > 0
    SPA_less_than_90 = dot(sun_vec, site - sc_pos) > 0

    if above_horizon and illuminated and SPA_less_than_90:
        if mid_pass_flag == False:
            print('Pass encountered',date)
            mid_pass_flag = True
            current_pass = {'Date0': date}
            max_el = 0

        az, el = AF.pass_az_el(site, sc_pos)
        if el > max_el:
            max_el = el

    else:
        if mid_pass_flag:
            mid_pass_flag = False
            current_pass['Date1'] = date
            current_pass['Pass Length'] = (date - current_pass['Date0']).total_seconds()
            if max_el > 20:
                pass_list.append(current_pass)
                date += datetime.timedelta(days = 30)




for _pass in pass_list:
    pass_directory = Directory + '/' + str(_pass['Date0']).replace(':','-')
    if not os.path.exists(pass_directory):
        os.makedirs(pass_directory)

    r0, v0 = Satellite.propagate(*_pass['Date0'].timetuple()[0:6])
    state0 = hstack([r0, v0])

    solver = ode(propagate_orbit)
    solver.set_integrator('lsoda')
    solver.set_initial_value(state0, 0)
    #times = asarray(arange(0, _pass['Pass Length'], DT))

    positions = []
    times = []
    while solver.t < _pass['Pass Length']:
        positions.append(solver.integrate(solver.t + DT))
        times.append(solver.t)
    times = hstack(times)
    positions = vstack(positions)

    obs_vecs = []
    sun_vecs = []
    sat_poss = []
    site_poss = []
    range_vecs = []
    for t, state in zip(times, positions):
        date = _pass['Date0'] + datetime.timedelta(seconds = t)
        lst = AF.local_sidereal_time(date, Lon)
        site = AF.observation_site(Lat, lst, Alt)
        #sc_pos, sc_vel = Satellite.propagate(*date.timetuple()[0:6])
        sc_pos = state[0:3]
        #sc_pos = asarray(sc_pos)
        range_vec = sc_pos - site
        sun_vec = AF.vect_earth_to_sun(date)

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

    length_of_data = int(PASS_TIME/DT)
    if len(obs_vecs) > length_of_data:
            start_index = random.randint(0, len(obs_vecs) - length_of_data - 1)
            obs_vecs = obs_vecs[start_index:start_index+length_of_data, :]
            sun_vecs = sun_vecs[start_index:start_index+length_of_data, :]
            sat_poss = sat_poss[start_index:start_index+length_of_data, :]
            site_poss = site_poss[start_index:start_index+length_of_data, :]
            range_vecs = range_vecs[start_index:start_index+length_of_data, :]
            times = times[start_index:start_index+length_of_data]

    azimuths, elevations = AF.pass_az_el(site_poss[::100], sat_poss[::100])

    save(pass_directory+'/satpos.npy', sat_poss)
    save(pass_directory+'/sitepos.npy', site_poss)
    save(pass_directory+'/rangevec.npy', range_vecs)
    save(pass_directory+'/sunvec.npy', sun_vecs)
    save(pass_directory+'/obsvec.npy', obs_vecs)
    save(pass_directory+'/time.npy', times)

    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection = 'polar')
    ax2 = fig.add_subplot(212)
    ax1.scatter(radians(azimuths), 90-elevations)
    ax1.set_theta_zero_location('N')
    ax1.set_ylim([0,90])

    for i in range(3):

        magnitude = random.rand()*.2
        min_magnitude = .1
        omega = random.rand(3)
        omega = omega/norm(omega)*magnitude + ones(3)*min_magnitude
        mrp = random.rand(3)
        mrp /= norm(mrp)
        state0 = hstack([mrp, omega])

        solver = ode(propagate_mrps)
        solver.set_integrator('lsoda')
        solver.set_initial_value(state0, 0)
        solver.set_f_params(Inertia)

        attitudes = []
        while len(attitudes) < len(times):
            attitudes.append(solver.integrate(solver.t + DT))
        attitudes = vstack(attitudes)

        save(pass_directory+'/mrps'+str(i)+'.npy', attitudes[:,0:3])
        save(pass_directory+'/angular_rate'+str(i)+'.npy', attitudes[:,3:6])

        lightcurve = states_to_lightcurve(obs_vecs, sun_vecs, attitudes[:,0:3], Geometry, Exposure_Time)
        save(pass_directory+'/lightcurve'+str(i)+'.npy', lightcurve)
        ax2.plot(times, lightcurve)

    plt.title('Lightcurves ' + str(_pass['Pass Length']) +' ' + str(omega))
    plt.savefig(pass_directory+'/Lightcurves.png')
    plt.close()


















