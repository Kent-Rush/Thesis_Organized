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
import Controls_Funcs as CF
from Attitude_Filter import Attitude_Filter
import Reflection_Funcs as RF

import json


"""
This file plotted a lot of the results in my thesis. I am including it for potential debugging in case I plotted something wrong and you want to know how I did it.
Also if you arent super familiar with matplotlib this can be a refference
"""

def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} :[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')

Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
Lat = Simulation_Configuration['Observation Site Lattitude']
Lon = Simulation_Configuration['Observation Site East Longitude']
Alt = Simulation_Configuration['Observation Site Altitude']
DT = Simulation_Configuration['Data Rate']
Inertia = asarray(Simulation_Configuration['Inertia Matrix'])
Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])
Geometry_Name = Simulation_Configuration['Spacecraft Geometry']
Noise_STD = Simulation_Configuration['Sensor STD']
Directory = Simulation_Configuration['Directory']
Real_Data = Simulation_Configuration['Real Data']

DPI = 200
LAST_INDEX = -2

est_inertia_flag = False
if '-i' in sys.argv:
    est_inertia_flag = True

pass_directories = [os.path.join(Directory,file) for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]

for i, passfile in enumerate(pass_directories):
    loading_bar(i/len(pass_directories), text = 'Plotting')
    for resultfile in [file for file in os.listdir(passfile) if os.path.isdir(os.path.join(passfile,file))]:


        run_number = resultfile[7]
        result_dir = os.path.join(passfile,resultfile)
        inertia_filtered = 'inertia' in resultfile

        times = load(os.path.join(passfile, 'time.npy'))

        true_lightcurve = load(os.path.join(passfile, 'lightcurve'+run_number+'.npy'))

        true_mrps = load(os.path.join(passfile, 'mrps'+run_number+'.npy'))
        true_rates = load(os.path.join(passfile, 'angular_rate'+run_number+'.npy'))

        obs_vecs = load(os.path.join(passfile, 'obsvec.npy'))
        sun_vecs = load(os.path.join(passfile, 'sunvec.npy'))

        est_lightcurve = load(os.path.join(result_dir, 'results'+run_number+'_estimated_curve.npy'))
        means = load(os.path.join(result_dir, 'results'+run_number+'_raw_means.npy'))
        covariances = load(os.path.join(result_dir, 'results'+run_number+'_raw_covariance.npy'))
        residuals = load(os.path.join(result_dir, 'results'+run_number+'_raw_residuals.npy'))

        stp = int(floor(1/DT))

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
        for obs, sun, truth, est in zip(obs_vecs, sun_vecs, eci_rate_true, eci_rate_ests):

            x = obs/norm(obs)
            z = cross(sun, obs)/norm(cross(sun,obs))
            y = cross(z, x)

            eci2obs = vstack([x, y, z])

            obs_frame_true_rate.append(eci2obs@truth)
            obs_frame_est_rate.append(eci2obs@est)


        obs_frame_true_rate = vstack(obs_frame_true_rate)
        obs_frame_est_rate = vstack(obs_frame_est_rate)


        # mrp_std = []
        # rate_std = []
        # inertia_std = []
        # for cov in covariances:
        #     mrp_vals, mrp_vecs = eig(cov[0:3, 0:3])
        #     mrp_std.append(abs(sum(sqrt(mrp_vals)*mrp_vecs, axis = 1)))

        #     rate_vals, rate_vecs = eig(cov[3:6,3:6])
        #     rate_std.append(abs(sum(sqrt(rate_vals)*rate_vecs, axis = 1)))

        #     if inertia_filtered:
        #         inertia_vals, inertia_vecs = eig(cov[6:8,6:8])
        #         inertia_std.append(abs(sum(sqrt(inertia_vals)*inertia_vecs, axis = 1)))
        # mrp_std = vstack(mrp_std)
        # rate_std = vstack(rate_std)
        # if inertia_filtered:
        #     inertia_std = vstack(inertia_std)

        raw_rates = means[:,3:6]
        if len(means[0]) != 8:
            for i, (truth, mean) in enumerate(zip(true_rates, means[:,3:6])):
                if dot(truth, mean) < 0:
                    means[i, 3:6] = -mean

        for i, truth in enumerate(true_mrps):
            if norm(truth) > 1:
                true_mrps[i] = -truth/norm(truth)**2


        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times[:LAST_INDEX:stp], true_rates[0:len(times),0][:LAST_INDEX:stp], 'k')
        ax1.plot(times[:LAST_INDEX:stp], means[:,3][:LAST_INDEX:stp], 'r--')

        ax2.plot(times[:LAST_INDEX:stp], true_rates[0:len(times),1][:LAST_INDEX:stp], 'k')
        ax2.plot(times[:LAST_INDEX:stp], means[:,4][:LAST_INDEX:stp], 'r--')

        ax3.plot(times[:LAST_INDEX:stp], true_rates[0:len(times),2][:LAST_INDEX:stp], 'k')
        ax3.plot(times[:LAST_INDEX:stp], means[:,5][:LAST_INDEX:stp], 'r--')

        ax1.grid()
        ax2.grid()
        ax3.grid()

        #ax1.set_title('X-rate Estimate vs Truth')
        ax1.legend(['Truth', 'Est.'],bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        ax1.set_title('X-rate Estimate vs Truth BODY Frame')
        ax2.set_title('Y-rate Estimate vs Truth BODY Frame')
        ax3.set_title('Z-rate Estimate vs Truth BODY Frame')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('Angular Rate [rad/s]')

        plt.savefig(os.path.join(result_dir,'A'),dpi = DPI, bbox_inches = 'tight')
        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times[:LAST_INDEX:stp], obs_frame_true_rate[0:len(times),0][:LAST_INDEX:stp], 'k')
        ax1.plot(times[:LAST_INDEX:stp], obs_frame_est_rate[:,0][:LAST_INDEX:stp],'r--')

        ax2.plot(times[:LAST_INDEX:stp], obs_frame_true_rate[0:len(times),1][:LAST_INDEX:stp], 'k')
        ax2.plot(times[:LAST_INDEX:stp], obs_frame_est_rate[:,1][:LAST_INDEX:stp],'r--')

        ax3.plot(times[:LAST_INDEX:stp], obs_frame_true_rate[0:len(times),2][:LAST_INDEX:stp], 'k')
        ax3.plot(times[:LAST_INDEX:stp], obs_frame_est_rate[:,2][:LAST_INDEX:stp],'r--')

        #ax1.set_title('X-rate Estimate vs Truth OBS Frame')
        ax1.legend(['Truth', 'Est.'])
        ax1.set_title('X-rate Estimate vs Truth OBS Frame')
        ax2.set_title('Y-rate Estimate vs Truth OBS Frame')
        ax3.set_title('Z-rate Estimate vs Truth OBS Frame')

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('Angular Rate [rad/s]')

        plt.savefig(os.path.join(result_dir,'B'),dpi = DPI, bbox_inches = 'tight')
        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times[:LAST_INDEX:stp], true_mrps[0:len(times),0][:LAST_INDEX:stp],'k')
        ax1.plot(times[:LAST_INDEX:stp], means[:,0][:LAST_INDEX:stp],'r--')

        ax2.plot(times[:LAST_INDEX:stp], true_mrps[0:len(times),1][:LAST_INDEX:stp],'k')
        ax2.plot(times[:LAST_INDEX:stp], means[:,1][:LAST_INDEX:stp],'r--')

        ax3.plot(times[:LAST_INDEX:stp], true_mrps[0:len(times),2][:LAST_INDEX:stp],'k')
        ax3.plot(times[:LAST_INDEX:stp], means[:,2][:LAST_INDEX:stp],'r--')

        ax1.set_title('MRP-1')
        ax1.legend(['Truth', 'Est.'],bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        ax2.set_title('MRP-2')
        ax3.set_title('MRP-3')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('MRP')

        plt.savefig(os.path.join(result_dir,'C'),dpi = 300, bbox_inches = 'tight')
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
        ax1.plot(times[:LAST_INDEX:stp], true_lightcurve[:LAST_INDEX:stp],'k')
        ax1.plot(times[:LAST_INDEX:stp], est_lightcurve[:LAST_INDEX:stp],'r--')
        ax2.plot(times[:LAST_INDEX:stp], residuals[:LAST_INDEX:stp],'k')
        ax1.set_title('Lightcurves')
        ax2.set_title('Residual Error')

        ax1.legend(['Truth', 'Estimate'],bbox_to_anchor=(0.5, 0., 0.5, 0.5))

        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Counts')

        plt.savefig(os.path.join(result_dir,'D'),dpi = 300, bbox_inches = 'tight')
        plt.close()

        if len(means[0]) == 8:
            fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
            ax1.plot([times[:LAST_INDEX:stp][0], times[:LAST_INDEX:stp][-1]], [Inertia[1,1], Inertia[1,1]],'k')
            ax1.plot(times[:LAST_INDEX:stp], means[:,6][:LAST_INDEX:stp],'r--')
            
            ax1.grid()
            ax1.set_title('YY-Inertia')
            ax1.legend(['Truth',r'Estimate'],bbox_to_anchor=(0.5, 0., 0.5, 0.5))

            ax2.plot([times[:LAST_INDEX:stp][0], times[:LAST_INDEX:stp][-1]], [Inertia[2,2], Inertia[2,2]],'k')
            ax2.plot(times[:LAST_INDEX:stp], means[:,7][:LAST_INDEX:stp],'r--')
            
            ax2.grid()
            ax2.set_title('ZZ-Inertia')

            plt.savefig(os.path.join(result_dir,'E'),dpi = DPI, bbox_inches = 'tight')
            plt.close()

        # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # rate_error = true_rates[0:len(times),:] - means[:,3:6]
        # ax1.plot(times[::stp], rate_error[:,0][::stp])
        # ax1.plot(times[::stp], 3*rate_std[:,0][::stp], 'k', alpha = .4)
        # ax1.plot(times[::stp], -3*rate_std[:,0][::stp], 'k', alpha = .4)
        # ax1.set_ylim(-.4, .4)
        # ax1.set_yticks(arange(-.4, .4, .2))
        # ax1.grid()
        # ax1.set_title('X-Rate Error')
        # ax1.legend(['Error',r'3$\sigma$ Error'])

        # ax2.plot(times[::stp], rate_error[:,1][::stp])
        # ax2.plot(times[::stp], 3*rate_std[:,1][::stp], 'k', alpha = .4)
        # ax2.plot(times[::stp], -3*rate_std[:,1][::stp], 'k', alpha = .4)
        # ax2.set_ylim(-.4, .4)
        # ax2.set_yticks(arange(-.4, .4, .2))
        # ax2.grid()
        # ax2.set_title('Y-Rate Error')

        # ax3.plot(times[::stp], rate_error[:,2][::stp])
        # ax3.plot(times[::stp], 3*rate_std[:,2][::stp], 'k', alpha = .4)
        # ax3.plot(times[::stp], -3*rate_std[:,2][::stp], 'k', alpha = .4)
        
        # ax3.set_ylim(-.4, .4)
        # ax3.set_yticks(arange(-.4, .4, .2))
        # ax3.grid()
        # ax3.set_title('Z-Rate Error')

        # ax3.set_xlabel('Time [s]')
        # ax2.set_ylabel('Error')

        # plt.savefig(os.path.join(result_dir,'Rate Errors.png'),dpi = DPI, bbox_inches = 'tight')
        # plt.close()



        # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # ax1.plot(times[:-2:stp], true_rates[0:len(times),0][:-2:stp], 'k')
        # ax1.plot(times[:-2:stp], means[:,3][:-2:stp], 'k--')

        # ax2.plot(times[:-2:stp], true_rates[0:len(times),1][:-2:stp], 'k')
        # ax2.plot(times[:-2:stp], means[:,4][:-2:stp], 'k--')

        # ax3.plot(times[:-2:stp], true_rates[0:len(times),2][:-2:stp], 'k')
        # ax3.plot(times[:-2:stp], means[:,5][:-2:stp], 'k--')

        # ax1.grid()
        # ax2.grid()
        # ax3.grid()

        # #ax1.set_title('X-rate Estimate vs Truth')
        # ax1.legend(['Truth', 'Est.'])
        # ax2.set_title('Y-rate Estimate vs Truth')
        # ax3.set_title('Z-rate Estimate vs Truth')

        # ax3.set_xlabel('Time [s]')
        # ax2.set_ylabel('Angular Rate [rad/s]')

        # plt.savefig(os.path.join(result_dir,Geometry_Name+' Rate Comparison.png'),dpi = DPI, bbox_inches = 'tight')
        # plt.close()

        
        # # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # # ax1.plot(times[:-2:stp], eci_rate_true[0:len(times),0][:-2:stp])
        # # ax1.plot(times[:-2:stp], eci_rate_ests[:,0][:-2:stp])

        # # ax2.plot(times[:-2:stp], eci_rate_true[0:len(times),1][:-2:stp])
        # # ax2.plot(times[:-2:stp], eci_rate_ests[:,1][:-2:stp])

        # # ax3.plot(times[:-2:stp], eci_rate_true[0:len(times),2][:-2:stp])
        # # ax3.plot(times[:-2:stp], eci_rate_ests[:,2][:-2:stp])

        # # ax1.grid()
        # # ax2.grid()
        # # ax3.grid()

        # # ax1.set_title('X-rate Estimate vs Truth ECI')
        # # ax1.legend(['Truth', 'Est.'])
        # # ax2.set_title('Y-rate Estimate vs Truth ECI')
        # # ax3.set_title('Z-rate Estimate vs Truth ECI')

        # # ax3.set_xlabel('Time [s]')
        # # ax2.set_ylabel('Angular Rate [rad/s]')

        # # plt.savefig(os.path.join(result_dir,'ECI Rate Comparison.png'),dpi = DPI, bbox_inches = 'tight')
        # # plt.close()

        # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # ax1.plot(times[:-2:stp], obs_frame_true_rate[0:len(times),0][:-2:stp], 'k')
        # ax1.plot(times[:-2:stp], obs_frame_est_rate[:,0][:-2:stp],'k--')

        # ax2.plot(times[:-2:stp], obs_frame_true_rate[0:len(times),1][:-2:stp], 'k')
        # ax2.plot(times[:-2:stp], obs_frame_est_rate[:,1][:-2:stp],'k--')

        # ax3.plot(times[:-2:stp], obs_frame_true_rate[0:len(times),2][:-2:stp], 'k')
        # ax3.plot(times[:-2:stp], obs_frame_est_rate[:,2][:-2:stp],'k--')

        # #ax1.set_title('X-rate Estimate vs Truth OBS Frame')
        # ax1.legend(['Truth', 'Est.'])
        # ax2.set_title('Y-rate Estimate vs Truth OBS Frame')
        # ax3.set_title('Z-rate Estimate vs Truth OBS Frame')

        # ax1.grid()
        # ax2.grid()
        # ax3.grid()

        # ax3.set_xlabel('Time [s]')
        # ax2.set_ylabel('Angular Rate [rad/s]')

        # plt.savefig(os.path.join(result_dir,Geometry_Name+' OBS Frame Rate Comparison.png'),dpi = DPI, bbox_inches = 'tight')
        # plt.close()




        # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # mrp_error = true_mrps[0:len(times),:] - means[:,0:3]
        # ax1.plot(times[::stp], mrp_error[:,0][::stp])
        # ax1.plot(times[::stp], 3*mrp_std[:,0][::stp], 'k', alpha = .4)
        # ax1.plot(times[::stp], -3*mrp_std[:,0][::stp], 'k', alpha = .4)
        # ax1.set_ylim(-.4, .4)
        # ax1.set_yticks(arange(-.4, .4, .2))
        # ax1.grid()
        # ax1.set_title('MRP-1 Error')
        # ax1.legend(['Error',r'3$\sigma$ Error'])

        # ax2.plot(times[::stp], mrp_error[:,1][::stp])
        # ax2.plot(times[::stp], 3*mrp_std[:,1][::stp], 'k', alpha = .4)
        # ax2.plot(times[::stp], -3*mrp_std[:,1][::stp], 'k', alpha = .4)
        # ax2.set_ylim(-.4, .4)
        # ax2.set_yticks(arange(-.4, .4, .2))
        # ax2.grid()
        # ax2.set_title('MRP-2 Error')

        # ax3.plot(times[::stp], mrp_error[:,2][::stp])
        # ax3.plot(times[::stp], 3*mrp_std[:,2][::stp], 'k', alpha = .4)
        # ax3.plot(times[::stp], -3*mrp_std[:,2][::stp], 'k', alpha = .4)
        # ax3.set_ylim(-.4, .4)
        # ax3.set_yticks(arange(-.4, .4, .2))
        # ax3.grid()
        # ax3.set_title('MRP-3 Error')

        # ax3.set_xlabel('Time [s]')
        # ax2.set_ylabel('Error')

        # plt.savefig(os.path.join(result_dir,'MRP Errors.png'),dpi = DPI, bbox_inches = 'tight')
        # plt.close()

        # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # ax1.plot(times[::stp], true_mrps[0:len(times),0][::stp])
        # ax1.plot(times[::stp], means[:,0][::stp])

        # ax2.plot(times[::stp], true_mrps[0:len(times),1][::stp])
        # ax2.plot(times[::stp], means[:,1][::stp])

        # ax3.plot(times[::stp], true_mrps[0:len(times),2][::stp])
        # ax3.plot(times[::stp], means[:,2][::stp])

        # ax1.set_title('MRP-1 Estimate vs Truth')
        # ax1.legend(['Truth', 'Est.'])
        # ax2.set_title('MRP-2 Estimate vs Truth')
        # ax3.set_title('MRP-3 Estimate vs Truth')

        # ax3.set_xlabel('Time [s]')
        # ax2.set_ylabel('MRP')

        # plt.savefig(os.path.join(result_dir,'MRP Comparison.png'),dpi = DPI, bbox_inches = 'tight')
        # plt.close()

        # fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        # ax1.plot(times[::stp], true_lightcurve[::stp])
        # ax2.plot(times[::stp], est_lightcurve[::stp])
        # ax3.plot(times[::stp], residuals[::stp])
        # ax1.set_title('True Lightcurve')
        # ax2.set_title('Estimated Lightcurve')
        # ax3.set_title('Resudial Error')

        # ax3.set_xlabel('Time [s]')
        # ax2.set_ylabel(r'$W/m^{2}$')

        # plt.savefig(os.path.join(result_dir,'LightCurve Residual.png'),dpi = DPI, bbox_inches = 'tight')
        # plt.close()

        # plt.figure()
        # angles = zeros(len(sun_vecs))
        # for i, (sv, ov) in enumerate(zip(sun_vecs, obs_vecs)):
        #     angles[i] = degrees(arccos(dot(sv, ov)/norm(sv)/norm(ov)))

        # plt.plot(times, angles)
        # plt.xlabel('Time [s]')
        # plt.ylabel('SPA [Deg]')
        # plt.title('Solar Phase Angle During Pass')
        # plt.savefig(os.path.join(result_dir,'SPA.png'), bbox_inches = 'tight', dpi = DPI)
        # plt.close()
