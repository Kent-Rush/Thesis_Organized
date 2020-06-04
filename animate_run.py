from numpy import *
from numpy.linalg import *

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys, os

sys.path.insert(0, 'Support_Files')

import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF
import Reflection_Funcs as RF

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
import json

"""
This file is for visualization and debugging purposes, it is very crude and I apologise.
When you call this file in the command line youll need to add a .json file as a parameter which specifies the geometry
Saves the animation in the same file as the data
"""

def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} :[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')




Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])

#This is jank, these are the directories where the data you want to animate are
truth_dir = 'BOX_WING_GEO/2020-09-11 11-21-52.681983'
result_dir = 'BOX_WING_GEO/2020-09-11 11-21-52.681983/results0'
result_num = result_dir[-1]

#Only renders every 100 datapoints, change this depending on how much data you want
skip = 100

obs_vecs = load(truth_dir+'/obsvec.npy')[::skip]
sun_vecs = load(truth_dir+'/sunvec.npy')[::skip]
attitudes = load(truth_dir+'/mrps'+result_num+'.npy')[::skip]
Exposure_Time = Simulation_Configuration['Exposure Time']

num_frames = len(obs_vecs)
START_INDEX = -5000

lightcurve = load(truth_dir+'/lightcurve'+result_num+'.npy')[::skip]
time = load(truth_dir+'/time.npy')[::skip]

for i in range(len(lightcurve)):
    plt.figure()
    ax = plt.axes()
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(min(lightcurve), max(lightcurve))
    ax.plot(time[0:i], lightcurve[0:i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Counts')
    plt.savefig(truth_dir+'/frames/frame{}.png'.format(i), bbox_inches = 'tight')
    plt.close()


frames = []
count = 0
max_val = 0
for mrps, obs_vec, sun_vec in zip(attitudes[START_INDEX:], obs_vecs[START_INDEX:], sun_vecs[START_INDEX:]):
    dcm_eci2body = CF.mrp2dcm(mrps).T
    #dcm_eci2body = CF.mrp2dcm(mrps).T
    image = RF.generate_image(Geometry, dcm_eci2body@obs_vec, dcm_eci2body@sun_vec, Exposure_Time, win_dim = (6,6), dpm = 20)
    im_max = amax(image)
    if im_max > max_val:
        max_val = im_max

    i, j = image.shape

    # for x in range(i):
    #     for y in range(j):
    #         if image[x,y] != 0:
    #             image[x,y] = 1
    frames.append(image)
    count += 1
    loading_bar(count/num_frames, 'Rendering gif')

frames = [frame/max_val for frame in frames]

imageio.mimsave(result_dir+'/true_rotation.gif',frames, fps = 10)




# attitudes = load(result_dir+'/results'+result_num+'_raw_means.npy')[:,0:3][::skip]

# frames = []
# count = 0
# max_val = 0
# for mrps, obs_vec, sun_vec in zip(attitudes[START_INDEX:], obs_vecs[START_INDEX:], sun_vecs[START_INDEX:]):
#     dcm_eci2body = CF.mrp2dcm(mrps).T
#     #dcm_eci2body = CF.mrp2dcm(mrps).T
#     image = RF.generate_image(Geometry, dcm_eci2body@obs_vec, dcm_eci2body@sun_vec, Exposure_Time, win_dim = (6,6), dpm = 20)
#     im_max = amax(image)
#     if im_max > max_val:
#         max_val = im_max

#     # i, j = image.shape

#     # for x in range(i):
#     #     for y in range(j):
#     #         if image[x,y] != 0:
#     #             image[x,y] = 1

#     frames.append(image)
#     count += 1
#     loading_bar(count/num_frames, 'Rendering gif')

# frames = [frame/max_val for frame in frames]

# imageio.mimsave(result_dir+'/estimated_rotation.gif',frames, fps = 10)