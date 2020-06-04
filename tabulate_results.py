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
This was a quick and dirty script I wrote to help interpret the data. You may find it useful but probably not. It summarizes the results of the simulated data trials in an excel table.
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

pass_directories = [file for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]

data = {}
columns = ['X Rate Error', 'X STD', 'X % Error', 'Y Rate Error', 'Y STD', 'Y % Error', 'Z Rate Error', 'Z STD', 'Z % Error',
           'ECI X Rate Error', 'ECI X STD', 'ECI X % Error', 'ECI Y Rate Error', 'ECI Y STD', 'ECI Y % Error', 'ECI Z Rate Error', 'ECI Z STD', 'ECI Z % Error',
           'OBS X Rate Error', 'OBS X STD', 'OBS X % Error', 'OBS Y Rate Error', 'OBS Y STD', 'OBS Y % Error', 'OBS Z Rate Error', 'OBS Z STD', 'OBS Z % Error',
           'YY Inertia Error', 'YY Inertia Error STD', 'YY Inertia  % Error', 'ZZ Inertia Error', 'ZZ Inertia Error STD', 'ZZ Inertia  % Error',
           'Mean X KF Update', 'KF X Update STD', 'Mean Y KF Update', 'KF Y Update STD', 'Mean Z KF Update', 'KF Z Update STD', 'YY KF Update', 'ZZ KF Update',
           'Mean Residual', 'Residual STD', 'Residual % Error']

NUM_PTS_CHECKED = int(floor(50/DT))

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

        true_lightcurve = load(os.path.join(passfile, 'lightcurve'+run_number+'.npy'))
        true_mrps = load(os.path.join(passfile, 'mrps'+run_number+'.npy'))
        true_rates = load(os.path.join(passfile, 'angular_rate'+run_number+'.npy'))

        obs_vecs = load(os.path.join(passfile, 'obsvec.npy'))
        sun_vecs = load(os.path.join(passfile, 'sunvec.npy'))

        est_lightcurve = load(os.path.join(result_dir, 'results'+run_number+'_estimated_curve.npy'))
        means = load(os.path.join(result_dir, 'results'+run_number+'_raw_means.npy'))
        covariances = load(os.path.join(result_dir, 'results'+run_number+'_raw_covariance.npy'))
        residuals = load(os.path.join(result_dir, 'results'+run_number+'_raw_residuals.npy'))

        eci_rate_true = vstack(list([CF.mrp2dcm(m)@rate for m, rate in zip(true_mrps, true_rates)]))
        eci_rate_ests = []
        for est, tru in zip(means, eci_rate_true):
            eci_rate_est = CF.mrp2dcm(est[0:3])@est[3:6]
            a = norm(eci_rate_est - tru)
            b = norm(-eci_rate_est - tru)
            if b < a:
                eci_rate_ests.append(-eci_rate_est)
            else:
                eci_rate_ests.append(eci_rate_est)
        eci_rate_ests = vstack(eci_rate_ests)


        obs_frame_true_rate = []
        obs_frame_est_rate = []
        for obs, sun, truth, est in zip(obs_vecs[-NUM_PTS_CHECKED:-2], sun_vecs[-NUM_PTS_CHECKED:-2], eci_rate_true[-NUM_PTS_CHECKED:-2], eci_rate_ests[-NUM_PTS_CHECKED:-2]):

            x = obs/norm(obs)
            z = cross(sun, obs)/norm(cross(sun,obs))
            y = cross(z, x)

            eci2obs = vstack([x, y, z])

            obs_frame_true_rate.append(eci2obs@truth)
            obs_frame_est_rate.append(eci2obs@est)


        obs_frame_true_rate = vstack(obs_frame_true_rate)
        obs_frame_est_rate = vstack(obs_frame_est_rate)


        for i, (truth, _mean) in enumerate(zip(true_rates, means[:,3:6])):
            if dot(truth, _mean) < 0:
                means[i, 3:6] = -_mean

        for i, truth in enumerate(true_mrps):
            if norm(truth) > 1:
                true_mrps[i] = -truth/norm(truth)

        final_errors = true_rates[-NUM_PTS_CHECKED:-2, :] - means[-NUM_PTS_CHECKED:-2, 3:6]
        X_err = mean(abs(final_errors[:,0]))
        data[rowname].append(X_err)
        X_std = std(final_errors[:,0])
        data[rowname].append(X_std)
        X_percent_err = X_err/mean(abs(true_rates[-NUM_PTS_CHECKED:-2,0]))*100
        data[rowname].append(X_percent_err)

        Y_err = mean(abs(final_errors[:,1]))
        data[rowname].append(Y_err)
        Y_std = std(final_errors[:,1])
        data[rowname].append(Y_std)
        Y_percent_err = Y_err/mean(abs(true_rates[-NUM_PTS_CHECKED:-2,1]))*100
        data[rowname].append(Y_percent_err)

        Z_err = mean(abs(final_errors[:,2]))
        data[rowname].append(Z_err)
        Z_std = std(final_errors[:,2])
        data[rowname].append(Z_std)
        Z_percent_err = Z_err/mean(abs(true_rates[-NUM_PTS_CHECKED:-2,2]))*100
        data[rowname].append(Z_percent_err)


        final_errors = eci_rate_true[-NUM_PTS_CHECKED:-2, :] - eci_rate_ests[-NUM_PTS_CHECKED:-2, :]
        X_err = mean(abs(final_errors[:,0]))
        data[rowname].append(X_err)
        X_std = std(final_errors[:,0])
        data[rowname].append(X_std)
        X_percent_err = X_err/mean(abs(eci_rate_true[-NUM_PTS_CHECKED:-2,0]))*100
        data[rowname].append(X_percent_err)

        Y_err = mean(abs(final_errors[:,1]))
        data[rowname].append(Y_err)
        Y_std = std(final_errors[:,1])
        data[rowname].append(Y_std)
        Y_percent_err = Y_err/mean(abs(eci_rate_true[-NUM_PTS_CHECKED:-2,1]))*100
        data[rowname].append(Y_percent_err)

        Z_err = mean(abs(final_errors[:,2]))
        data[rowname].append(Z_err)
        Z_std = std(final_errors[:,2])
        data[rowname].append(Z_std)
        Z_percent_err = Z_err/mean(abs(eci_rate_true[-NUM_PTS_CHECKED:-2,2]))*100
        data[rowname].append(Z_percent_err)

        final_errors = obs_frame_true_rate - obs_frame_est_rate
        X_err = mean(abs(final_errors[:,0]))
        data[rowname].append(X_err)
        X_std = std(final_errors[:,0])
        data[rowname].append(X_std)
        X_percent_err = X_err/mean(abs(obs_frame_true_rate[-NUM_PTS_CHECKED:-2,0]))*100
        data[rowname].append(X_percent_err)

        Y_err = mean(abs(final_errors[:,1]))
        data[rowname].append(Y_err)
        Y_std = std(final_errors[:,1])
        data[rowname].append(Y_std)
        Y_percent_err = Y_err/mean(abs(obs_frame_true_rate[-NUM_PTS_CHECKED:-2,1]))*100
        data[rowname].append(Y_percent_err)

        Z_err = mean(abs(final_errors[:,2]))
        data[rowname].append(Z_err)
        Z_std = std(final_errors[:,2])
        data[rowname].append(Z_std)
        Z_percent_err = Z_err/mean(abs(obs_frame_true_rate[-NUM_PTS_CHECKED:-2,2]))*100
        data[rowname].append(Z_percent_err)


        if len(means[0]) == 8:
            
            final_errors = Inertia[1,1] - means[-NUM_PTS_CHECKED:-2, 6]
            X_err = mean(abs(final_errors))
            data[rowname].append(X_err)
            
            X_std = std(final_errors)
            data[rowname].append(X_std)
            
            X_percent_err = X_err/Inertia[1,1]*100
            data[rowname].append(X_percent_err)

            
            final_errors = Inertia[2,2] - means[-NUM_PTS_CHECKED:-2, 7]
            Y_err = mean(abs(final_errors))
            data[rowname].append(Y_err)
            
            Y_std = std(final_errors)
            data[rowname].append(Y_std)
           
            Y_percent_err = Y_err/Inertia[2,2]*100
            data[rowname].append(Y_percent_err)

        else:
            for i in range(6):
                data[rowname].append(None)


        if len(means[0]) == 8:
            est_inertia_flag = True
        else:
            est_inertia_flag = False

        updates = []
        least_few_means = means[-NUM_PTS_CHECKED:-2]
        last_few_dts = dts[-(NUM_PTS_CHECKED - 1):-2]
        for i, (_mean, dt) in enumerate(zip(least_few_means, last_few_dts)):

            pred = modified_rodrigues_prop(_mean, dt, inertia = Inertia, est_inertia = est_inertia_flag)
            update = least_few_means[i+1] - pred

            updates.append(update)


        updates = vstack(updates)

        mean_update = mean(abs(updates[:, 3]))
        data[rowname].append(mean_update)
        std_update = std(updates[:, 3])
        data[rowname].append(std_update)

        mean_update = mean(abs(updates[:, 4]))
        data[rowname].append(mean_update)
        std_update = std(updates[:, 4])
        data[rowname].append(std_update)

        mean_update = mean(abs(updates[:, 5]))
        data[rowname].append(mean_update)
        std_update = std(updates[:, 5])
        data[rowname].append(std_update)

        if len(means[0]) == 8:
            mean_update = mean(abs(updates[:,6]))
            data[rowname].append(mean_update)
            mean_update = mean(abs(updates[:,7]))
            data[rowname].append(mean_update)
        else:
            data[rowname].append(None)
            data[rowname].append(None)

        data[rowname].append(mean(abs(residuals[NUM_PTS_CHECKED: -2])))
        data[rowname].append(std(residuals[NUM_PTS_CHECKED:-2]))
        data[rowname].append(mean(abs(residuals[NUM_PTS_CHECKED:-2]/true_lightcurve[NUM_PTS_CHECKED:-2])))


df = pd.DataFrame.from_dict(data, orient = 'index', columns = columns)
df.to_csv(os.path.join(Directory, 'datatable.csv'))
df.to_pickle(os.path.join(Directory, 'datatable.pkl'))




