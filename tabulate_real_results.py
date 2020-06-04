from numpy import *
from numpy.linalg import *
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import ode
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, MMAEFilterBank, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import sys, os
sys.path.insert(0, 'Support_Files')
sys.path.insert(0, '..')
import datetime
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

import Aero_Funcs as AF
import Aero_Plots as AP
from Attitude_Filter import Attitude_Filter
import Reflection_Funcs as RF
import Controls_Funcs as CF

import json
import pandas as pd

"""
This was a quick and dirty script I wrote to help interpret the data. You may find it useful but probably not. It summarizes the results of the real data trials in an excel table.
This script requies the json file for the trial as a parameter in the command line
"""

def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} :[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')

def propagate_mrps(t, state, true_inertia):
    mrps = state[0:3]
    omega = state[3:6]

    d_mrps = CF.mrp_derivative(mrps, omega)
    d_omega = inv(true_inertia)@(-cross(omega, true_inertia@omega))

    return hstack([d_mrps, d_omega])

def propagate_mrps_inertia(t, state):
    mrps = state[0:3]
    omega = state[3:6]

    est_inertia = diag([1, abs(state[6]), abs(state[7])])

    d_mrps = CF.mrp_derivative(mrps, omega)
    d_omega = inv(est_inertia)@(-cross(omega, est_inertia@omega))

    return hstack([d_mrps, d_omega, 0, 0])

def modified_rodrigues_prop(state, dt, inertia = None, est_inertia = False):

    #print(state)
    if not est_inertia:
        solver = ode(propagate_mrps)
    else:
        solver = ode(propagate_mrps_inertia)

    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)

    if not est_inertia:
        solver.set_f_params(inertia)

    solver.integrate(dt)

    return hstack([solver.y])

Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
# Satellite = twoline2rv(Simulation_Configuration['TLE Line1'],
#                        Simulation_Configuration['TLE Line2'], wgs84)
# Lat = Simulation_Configuration['Observation Site Lattitude']
# Lon = Simulation_Configuration['Observation Site East Longitude']
# Alt = Simulation_Configuration['Observation Site Altitude']
# DT = Simulation_Configuration['Data Rate']
# Inertia = asarray(Simulation_Configuration['Inertia Matrix'])
# Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])
# Noise_STD = Simulation_Configuration['Sensor STD']
Directory = Simulation_Configuration['Directory']

pass_directories = [file for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]

data = {}
columns = ['X Body', 'Y Body', 'Z Body', 'X ECI', 'Y ECI', 'Z ECI', 'OBS X', 'OBS Y', 'OBS Z', 'X Update', 'Y Update', 'Z Update']

for i, filename in enumerate(pass_directories):
    passfile = os.path.join(Directory, filename)
    loading_bar(i/len(pass_directories), text = 'Processing')
    for resultfile in [file for file in os.listdir(passfile) if os.path.isdir(os.path.join(passfile,file))]:



        run_number = resultfile[7]
        result_dir = os.path.join(passfile,resultfile)

        rowname = filename + '_' + str(run_number)
        data[rowname] = []

        times = load(os.path.join(passfile, 'time.npy'))
        dts = diff(times)

        obs_vecs = load(os.path.join(passfile, 'obsvec.npy'))
        sun_vecs = load(os.path.join(passfile, 'sunvec.npy'))

        est_lightcurve = load(os.path.join(result_dir, 'results'+run_number+'_estimated_curve.npy'))
        means = load(os.path.join(result_dir, 'results'+run_number+'_raw_means.npy'))
        covariances = load(os.path.join(result_dir, 'results'+run_number+'_raw_covariance.npy'))
        residuals = load(os.path.join(result_dir, 'results'+run_number+'_raw_residuals.npy'))


        eci_rate_ests = []
        for est in means:
            eci_rate_est = CF.mrp2dcm(est[0:3])@est[3:6]
            eci_rate_ests.append(eci_rate_est)
        eci_rate_ests = vstack(eci_rate_ests)

        obs_frame_est_rate = []
        for obs, sun, est in zip(obs_vecs, sun_vecs, eci_rate_ests):

            x = obs/norm(obs)
            z = cross(sun, obs)/norm(cross(sun,obs))
            y = cross(z, x)

            eci2obs = vstack([x, y, z])
            obs_frame_est_rate.append(eci2obs@est)

        obs_frame_est_rate = vstack(obs_frame_est_rate)

        for c in means[-50, 3:6]:
            data[rowname].append(c)
        for c in eci_rate_ests[-50]:
            data[rowname].append(c)
        for c in obs_frame_est_rate[-50]:
            data[rowname].append(c)

        updates = []
        least_few_means = means[-31:-2]
        last_few_dts = dts[-30:-2]
        for i, (_mean, dt) in enumerate(zip(least_few_means, last_few_dts)):

            pred = modified_rodrigues_prop(_mean, dt, inertia = identity(3), est_inertia = False)
            update = least_few_means[i+1] - pred

            updates.append(update)

        updates = vstack(updates)


        mean_update = mean(abs(updates[:, 3]))
        data[rowname].append(mean_update)
        std_update = std(updates[:, 3])
        #data[rowname].append(std_update)

        mean_update = mean(abs(updates[:, 4]))
        data[rowname].append(mean_update)
        std_update = std(updates[:, 4])
        #data[rowname].append(std_update)

        mean_update = mean(abs(updates[:, 5]))
        data[rowname].append(mean_update)
        std_update = std(updates[:, 5])
        #data[rowname].append(std_update)



df = pd.DataFrame.from_dict(data, orient = 'index', columns = columns)
df.to_csv(os.path.join(Directory, 'datatable.csv'))
df.to_pickle(os.path.join(Directory, 'datatable.pkl'))