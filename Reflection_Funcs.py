from numpy import *
import numpy as np
from numpy.linalg import *

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys

sys.path.insert(0, 'Support_Files')

import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
import pickle

"""
This is an essential file for the thesis. This file is where all the measurement modelling happens.
This file contains class definitions for facets and spacecraft geometries as well as all the logic for ray tracing.
I really made sure this was well documented.
"""

# def lommel_seeliger(obs_vec, sun_vec, albedo, normal, area):

#     #determines the brightness of a facet at 1 meter distance
#     #obs_dot is the dot product of the facet normal and observation vector
#     #sun_dot is the dot product of the facet normal and sun vector
#     #solar_phase_angle is the angle between the sun and observation vector
#     #albedo is albedo
#     #area is the facet area
#     #from LIGHTCURVE INVERSION FOR SHAPE ESTIMATION OF GEO OBJECTS FROM
#     #SPACE-BASED SENSORS

#     obs_norm = obs_vec/norm(obs_vec)
#     sun_norm = sun_vec/norm(sun_vec)

#     obs_dot = dot(normal, obs_norm)
#     sun_dot = dot(normal, sun_norm)

#     if obs_dot <= 0 or sun_dot <= 0:
#         return 0

#     solar_phase_angle = arccos(dot(obs_norm, sun_norm))

#     #constants from above paper
#     c = .1
#     A0 = .5
#     D = .1
#     k = -.5


#     phase = A0*exp(-solar_phase_angle/D) + k*solar_phase_angle + 1

#     scattering = phase*obs_dot*sun_dot*(1/(obs_dot + sun_dot) + c)
#     brightness = scattering*albedo*area


#     return brightness


def phong_brdf(obs_vec, sun_vec, normal, area, exposure_time, r_specular = .2, r_diffuse = 0):
    #As implemented in INACTIVE SPACE OBJECT SHAPE ESTIMATION
    #VIA ASTROMETRIC AND PHOTOMETRIC DATA FUSION

    #Assumes specular lobe is even in all directions, nu = nv = N_PHONG
    C_SUN = 455 #Visible spectrum solar flux
    n_phong = 1
    CCD_GAIN = 4.8 #Electrons per CCD "Count" from datasheet
    TELESCOPE_DIAMETER = 1 #meter
    ELECTRON_ENERGY = 2.27 #Electron Volt [eV]
    J2eV = 1.6022e-19 #Joules per eV
    alpha = 1

    obs_norm = obs_vec/norm(obs_vec)
    sun_norm = sun_vec/norm(sun_vec)
    h_vec = (obs_norm + sun_norm)
    h_vec = h_vec/norm(h_vec)
    normal = normal/norm(normal)

    dot_ns = dot(normal, sun_norm)
    dot_no = dot(normal, obs_norm)
    dot_nh = dot(normal, h_vec)

    # Equations from:
    #INACTIVE SPACE OBJECT SHAPE ESTIMATION VIA ASTROMETRIC AND PHOTOMETRIC DATA FUSION
    exponent = n_phong
    F_reflect = r_specular + (1-r_specular)*(1 - dot(sun_norm, h_vec))**5
    denominator = dot_ns + dot_no - dot_ns*dot_no

    #calculate 
    specular = sqrt((n_phong+1)*(n_phong+1))/(8*pi)*(dot_nh**exponent)/denominator*F_reflect
    diffuse = 28*r_diffuse/(23*pi)*(1 - r_specular)*(1 - (1 - dot_ns/2)**5)*(1 - (1 - dot_no/2)**5)

    Fsun = C_SUN*(specular + diffuse)*dot_ns
    Fobs = Fsun*area*dot_no/norm(obs_vec*1e3)**2

    collecting_area = pi*(TELESCOPE_DIAMETER**2)/4
    collected_energy = Fobs*collecting_area*exposure_time
    photons_collected = collected_energy/J2eV/ELECTRON_ENERGY
    counts = photons_collected/CCD_GAIN #flux
    instrument_magnitude =  -26 - 2.5*log10(Fobs/C_SUN)

    return counts

class Facet():

    """
    This is a class which generalizes the properties of each individual surface of which SpacecraftGeometries are composed.
    This includes information such as their position and orientation, reflection properties, and vertex positions.
    All facets are rectangles

    Parameters:
    x_dim       = the dimension of the rectangle along the X axis
    y_dim       = the dimension of the rectangle along the Y axis
    center_pos  = the position of the center of the rectangle in the spacecraft body frame

    Keyword Parameters:
    name            = A string which can be accessed for debugging purposes
    facet2body      = A numpy array, rotation matrix describing the rotation from the facets frame to the spacecraft body frame.
                      The facet frame assumes that the X and Y axes are along the edges of the rectangle with the Z axis rising out from it.
    double_sided    = This is a boolean parameters which is TRUE if the facet can be illuminated from both sides. This should be FALSE
                      whenever possible to reduce computation time.
    specular_coef   = A float which represents the specular coefficient for the Phong BRDF model
    diffuse_coef    = A float which represents the diffuse coefficient for the Phong BRDF model

    Functions:
    intersects(source, direction, check_bounds)
        - Returns the distance of a point from a plane along a ray. Returns infinity if check_bounds is TRUE and
          the intersection of the ray and plane is outside of the dimensions of the facet.

        Parameters:
        source      = a 3D point in the spacecraft body frame where the ray originates.
        direction   = the ray along which the intersection is calculated
        check_bounds= boolean which causes the function to return infinity if the ray misses the facet. If False,
                      the function returns the distance to the intersection of the infinite plane of the facet.

    calc_vertices()
        - Internal function only. Calculates the vertices of the facet in the spacecraft body frame

    """

    def __init__(self, x_dim, y_dim, center_pos, name = '', facet2body = None, double_sided = False, specular_coef = 0.2, diffuse_coef = 0):
        self.center = center_pos
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.area = x_dim*y_dim
        self.double_sided = double_sided
        self.specular_coef = specular_coef
        self.diffuse_coef = diffuse_coef

        self.name = name

        # angle = arccos(dot(self.unit_normal, array([0,0,1])))
        # axis = cross(self.unit_normal, array([0,0,1]))
        # if facet2body == None:
        #     self.facet2body = axis_angle2dcm(-axis, angle)
        #     self.unit_normal = facet2body@array([0,0,1])
        # else:
        self.facet2body = facet2body
        self.unit_normal = facet2body@array([0,0,1])
        self.vertices = []

        self.calc_vertices()


    def intersects(self, source, direction, check_bounds = True):


        direction = direction/norm(direction)

        #check if ray and plane are perpendicular
        if dot(direction, self.unit_normal) == 0:
            return inf
        #check if the source is ON the plane
        elif dot((self.center - source), self.unit_normal) == 0:
            return 0

        else:
            distance_from_source = dot((self.center - source), self.unit_normal)/dot(direction, self.unit_normal)



            intersection = source + direction*distance_from_source

            intersection_in_plane = self.facet2body.T@(intersection - self.center)

            if check_bounds:
                within_x = (intersection_in_plane[0] <= self.x_dim/2) and (intersection_in_plane[0] >= -self.x_dim/2)
                within_y = (intersection_in_plane[1] <= self.y_dim/2) and (intersection_in_plane[1] >= -self.y_dim/2)

                if within_x and within_y:
                    return distance_from_source
                else:
                    return inf
            else:
                return distance_from_source

    def calc_vertices(self):
        for x in [-self.x_dim/2, self.x_dim/2]:
            for y in [-self.y_dim/2, self.y_dim/2]:
                pos = self.facet2body@array([x, y, 0]) + self.center
                self.vertices.append(pos)
        self.vertices = vstack(self.vertices)





class Spacecraft_Geometry():
    """

    A class composed of Facet objects. Enables ray tracing.

    Parameters:
    facets      = a list of Facet objects which compose the spacecraft model
    Keyword Parameters:
    sample_dim  = a float which represents the ideal edge length for the sample grid used in the ray tracing. Units are Meters.

    Funcions
    A_obscures_B(A, B)
        - Returns T/F If facet A is able to cast shadows on facet B

        Parameters:
        A = Facet object
        B = Facet object

    calc_obscuring_faces()
        - Internal function only, generates a dictionary where each key is a Facet object and each value is a list of 
          all other Facets that can obscure it.

    calc_sample_pts()
        - Internal function only. Generates a dictionary where each key is a Facet object and each value is a list of points
          in the spacecraft body frame which will be sampled by the ray tracing algorythm.

    calc_reflected_power(obs_vec_body, sun_vec_body, exposure_time)
        - depsite having power in the name, actually returns the units of the Phong BRDF model which is currently Counts.

        Parameters:
        obs_vec_body = unit vector in body frame representing the direction the spacecraft is being VIEWED from.
        sun_vec_body = unit vector in body frame representing the direction the spacecraft is being ILLUMINATED from.
        exposure_time= float representing the ammount of time the CCD was exposed for in seconds.

    calc_reflecting_area(obs_vec_body, sun_vec_body, facet)
        - Calculates the illuminated and visible area of a facet.

        Parameters:
        obs_vec_body = unit vector in body frame representing the direction the spacecraft is being VIEWED from.
        sun_vec_body = unit vector in body frame representing the direction the spacecraft is being ILLUMINATED from.
        facet        = A facet object whose reflecting area is to be evaluated.

    trace_ray(source, ray, sun_vec, exposure_time)
        - Returns the brightness of the point intersected, 0 if the ray did not hit anything.

        Parameters:
        source          = a 3D point in the spacecraft body frame where the ray originates.
        ray             = the unit vector describing the rays direction
        sun_vec         = unit vector in body frame representing the direction the spacecraft is being ILLUMINATED from.
        exposure_time   = float representing the ammount of time the CCD was exposed for in seconds.
    """

    def __init__(self, facets, sample_dim = .01):

        self.facets = facets
        self.obscuring_facets = {}
        self.sample_points = {}
        self.sample_nums = {}
        self.sample_dim = sample_dim


        self.calc_obscuring_faces()
        self.calc_sample_pts()

        

    def A_obscures_B(self, A, B):

        for vertex in A.vertices:
            v_test = vertex - B.center
            if dot(v_test, B.unit_normal) > 0.03:
                #print(dot(v_test, B.unit_normal), A.name, B.name)
                return True

        return False

    def calc_obscuring_faces(self):
        for B in self.facets:
            self.obscuring_facets[B] = []
            for A in self.facets:
                if (A != B) and self.A_obscures_B(A, B):
                    self.obscuring_facets[B].append(A)

    def calc_sample_pts(self):

        for facet in self.facets:

            self.sample_points[facet] = []
            x_num = int(ceil(facet.x_dim/self.sample_dim))
            y_num = int(ceil(facet.y_dim/self.sample_dim))

            self.sample_nums[facet] = x_num*y_num

            x_buff = facet.x_dim/x_num/2.0
            y_buff = facet.y_dim/y_num/2.0

            for x in linspace(-facet.x_dim/2.0 + x_buff, facet.x_dim/2.0 - x_buff, x_num):
                for y in linspace(-facet.y_dim/2.0 + y_buff, facet.y_dim/2.0 - y_buff, y_num):
                    self.sample_points[facet].append(facet.facet2body.T@array([x,y,0]))



    def calc_reflected_power(self, obs_vec_body, sun_vec_body, exposure_time):

        power = 0
        #poop = []
        for facet in self.facets:

            if dot(facet.unit_normal, sun_vec_body) > 0 and dot(facet.unit_normal, obs_vec_body) > 0:

                if len(self.obscuring_facets[facet]) == 0:
                    reflecting_area = facet.area
                else:
                    reflecting_area = self.calc_reflecting_area(obs_vec_body, sun_vec_body, facet)

                power += phong_brdf(obs_vec_body, sun_vec_body, facet.unit_normal, reflecting_area, exposure_time, facet.specular_coef, facet.diffuse_coef)


        return power

    def calc_reflecting_area(self, obs_vec_body, sun_vec_body, facet):

        num_invisible = 0
        for pt in self.sample_points[facet]:
            if len(self.obscuring_facets[facet]) == 0:
                area = facet.area
            else:
                for obscurer in self.obscuring_facets[facet]:
                    if (obscurer.intersects(pt, obs_vec_body) != inf) or (obscurer.intersects(pt, sun_vec_body) != inf):
                        num_invisible += 1
                        break
                area = facet.area*(1 - num_invisible/self.sample_nums[facet])

        return area

    def trace_ray(self, source, ray, sun_vec, exposure_time):

        obs_vec = -ray
        distances = asarray([f.intersects(source, ray) for f in self.facets])
        index = where(distances == amin(distances))[0][0]

        distance = distances[index]
        facet = self.facets[index]



        if facet.double_sided and dot(facet.unit_normal, obs_vec) < 0:
            unit_normal = -facet.unit_normal
        else:
            unit_normal = facet.unit_normal
            
        # if facet.name == '+X wing':
        #     print(dot(unit_normal, sun_vec) < 0, dot(unit_normal, obs_vec) < 0)

        # if distance != inf:
        #     print(facet.name, dot(unit_normal, sun_vec) < 0, dot(unit_normal, obs_vec) < 0)
        #     print(facet.name, facet.unit_normal)


        if distance == inf:
            return 0

        elif dot(unit_normal, sun_vec) < 0 or dot(unit_normal, obs_vec) < 0:
            return 0

        else:

            surface_pt = source + ray*distance
            if len(self.obscuring_facets[facet]) != 0:
                for obscurer in self.obscuring_facets[facet]:
                        dist = obscurer.intersects(surface_pt, sun_vec)
                        if (dist != inf) and (dist > 0):
                            #print(facet.name)
                            return 0
                            break
            return phong_brdf(obs_vec, sun_vec, unit_normal, 1, exposure_time, facet.specular_coef, facet.diffuse_coef)

        

def generate_image(spacecraft_geometry, obs_vec_body, sun_vec_body, exposure_time, win_dim = (2,2), dpm = 50, load_bar = False):
    """
    Debugging tool, used to make animations. Generates a 2D array of grayscale pixels representing an image of A Spacecraft Geometry Object.
    This assumes no FOV effects.

    Parameters:
    Spacecraft_Geometry     = The SpacecraftGeometry object whose image will be generated.
    obs_vec_body            = unit vector in body frame representing the direction the spacecraft is being VIEWED from.
    sun_vec_body            = unit vector in body frame representing the direction the spacecraft is being ILLUMINATED from.
    exposure_time           = float representing the ammount of time the CCD was exposed for in seconds.

    Keyword Parameters:
    win_dim                 = A tuple representing the dimensions of the projected area of the image in meters.
    dpm                     = INteger representing how many pixels per meter to generate. Larger numbers lead to signifnicant slower run times.
    load_bar                = True if the user wants a loading bar to appear when generating an image.
    """

    win_pix = (win_dim[0]*dpm, win_dim[1]*dpm)
    image = zeros(win_pix)
    perspective_distance = 5

    obs_vec_body = obs_vec_body/norm(obs_vec_body)
    camera_pos = obs_vec_body*5
    ray_direction = -obs_vec_body

    
    camera_angle = arccos(dot(ray_direction, array([0,0,1])))
    camera_rotation = CF.axis_angle2dcm(cross(array([0,0,1]), ray_direction), camera_angle)

    for y, row in enumerate(image):
        if load_bar:
                loading_bar(y/len(image), text = 'Recticulating Splines')
        for x, pix in enumerate(row):
            x_pos = (x - win_pix[0]/2)/dpm
            y_pos = (win_pix[1]/2 - y)/dpm
            pix_pos = camera_rotation@array([x_pos, y_pos, 0]) + camera_pos
            
            image[x,y] = spacecraft_geometry.trace_ray(pix_pos, ray_direction, sun_vec_body, exposure_time)

    m = amax(image)
    # if m == 0:
    #     print('m = 0',obs_vec_body, sun_vec_body)
    if m != 0:
        image = image/m

    return image


class Premade_Spacecraft():

    """
    A really dumb idea. Class contains Spacecraft Geometries which can be easily called elsewhere.

    Functions:
    get_geometry(name):
        - Returns the spacecraft geometry specified by name

        Parameters:
        name = A string representing the name of a spacecraft geometry. Spacecraft names included in this class are:
        BOX_WING
        BOX
        PLATE
        CYLINDER
        RECTANGLE
        LONG_RECTANGLE
        ARIANE40
        EXOCUBE
        SL8
        SERT2
        AJISAI
        SPLOTCHY_RECTANGLE

    """

    def __init__(self):
        pZ = Facet(1, 1, array([0,0, .5]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(1, 1, array([0,0,-.5]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(1, 1, array([ .5,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(1, 1, array([-.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(1, 1, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(1, 1, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y')
        wingnX = Facet(1, .5, array([-1, 0,0]), facet2body = CF.Cx(pi/2), name = '-X wing', double_sided = True)
        wingpX = Facet(1, .5, array([ 1, 0,0]), facet2body = CF.Cx(pi/2), name = '+X wing', double_sided = True)
        
        self.BOX_WING = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ,wingnX, wingpX], sample_dim = .1)

        self.BOX = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

        plate = Facet(1,1, array([0,0,0]), facet2body = identity(3), name = 'plate', double_sided = True)
        self.PLATE = Spacecraft_Geometry([plate])


        segments = 20
        angle = 2*pi/segments
        radius = 1.5
        side_length = radius*sin(angle/2)*radius
        lenght = 9
        cylinder_facets = []
        for theta in linspace(0, 2*pi, segments)[:-1]:
            pos = CF.Cz(theta)@array([1,0,0])
            facet2body = CF.Cz(theta)@CF.Cy(pi/2)
            cylinder_facets.append(Facet(lenght, side_length, pos, facet2body = facet2body, name = 'cylinder'))

        pZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,lenght/2]), name = '+Z', facet2body = identity(3))
        pZ.area = pi*radius**2
        nZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,-lenght/2]), name = '-Z', facet2body = CF.Cx(pi))
        nZ.area = pi*radius**2
        cylinder_facets.append(pZ)
        cylinder_facets.append(nZ)

        self.CYLINDER = Spacecraft_Geometry(cylinder_facets, sample_dim = .5)

        spec_coef = 0
        diffuse_coef = .002
        segments = 20
        angle = 2*pi/segments
        radius = 1.3
        side_length = radius*sin(angle/2)*2
        lenght = 11.6
        cylinder_facets = []
        for theta in linspace(0, 2*pi, segments)[:-1]:
            pos = CF.Cz(theta)@array([radius,0,0])
            facet2body = CF.Cz(theta)@CF.Cy(pi/2)
            cylinder_facets.append(Facet(lenght, side_length, pos, facet2body = facet2body, name = 'cylinder', specular_coef = spec_coef, diffuse_coef = diffuse_coef))

        pZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,lenght/2]), name = '+Z', facet2body = identity(3), specular_coef = spec_coef, diffuse_coef = diffuse_coef)
        pZ.area = pi*radius**2
        nZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,-lenght/2]), name = '-Z', facet2body = CF.Cx(pi), specular_coef = spec_coef, diffuse_coef = diffuse_coef)
        nZ.area = pi*radius**2
        cylinder_facets.append(pZ)
        cylinder_facets.append(nZ)

        #https://www.spacelaunchreport.com/ariane4.html
        #data from second stage
        self.ARIANE40 = Spacecraft_Geometry(cylinder_facets, sample_dim = .5)

        diffuse_coef = .0002

        segments = 20
        angle = 2*pi/segments
        radius = 1.2
        side_length = radius*sin(angle/2)*2
        lenght = 6.5
        cylinder_facets = []
        for theta in linspace(0, 2*pi, segments)[:-1]:
            pos = CF.Cz(theta)@array([radius,0,0])
            facet2body = CF.Cz(theta)@CF.Cy(pi/2)
            cylinder_facets.append(Facet(lenght, side_length, pos, facet2body = facet2body, specular_coef = 0, diffuse_coef = diffuse_coef, name = 'cylinder'))

        pZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,lenght/2]), name = '+Z', facet2body = identity(3), specular_coef = 0, diffuse_coef = diffuse_coef)
        pZ.area = pi*radius**2
        nZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,-lenght/2]), name = '-Z', facet2body = CF.Cx(pi), specular_coef = 0, diffuse_coef = diffuse_coef)
        nZ.area = pi*radius**2
        cylinder_facets.append(pZ)
        cylinder_facets.append(nZ)

        self.SL8 = Spacecraft_Geometry(cylinder_facets, sample_dim = .5)


        pZ = Facet(3.0, 1.0, array([0,0, 1.0]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(3.0, 1.0, array([0,0,-1.0]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(2.0, 1.0, array([ 1.5,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(2.0, 1.0, array([-1.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(3.0, 2.0, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(3.0, 2.0, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y')

        self.RECTANGLE = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

        pZ = Facet(5.0, 1.0, array([0,0, 1.0]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(5.0, 1.0, array([0,0,-1.0]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(2.0, 1.0, array([ 2.5,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(2.0, 1.0, array([-2.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(5.0, 2.0, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(5.0, 2.0, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y')

        self.LONG_RECTANGLE = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

        pZ = Facet(5.0, 1.0, array([0,0, 1.0]), facet2body = identity(3) , name = '+Z', diffuse_coef = .001, specular_coef = 0)
        nZ = Facet(5.0, 1.0, array([0,0,-1.0]), facet2body = CF.Cy(pi) , name = '-Z', diffuse_coef = .002, specular_coef = 0)
        pX = Facet(2.0, 1.0, array([ 2.5,0,0]), facet2body = CF.Cy(pi/2), name = '+X', diffuse_coef = .003, specular_coef = 0)
        nX = Facet(2.0, 1.0, array([-2.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X', diffuse_coef = .004, specular_coef = 0)
        pY = Facet(5.0, 2.0, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y', diffuse_coef = .005, specular_coef = 0)
        nY = Facet(5.0, 2.0, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y', diffuse_coef = .006, specular_coef = 0)

        self.SPLOTCHY_RECTANGLE = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

        xdim = .1
        ydim = .1
        zdim = .3
        pZ = Facet(xdim, ydim, array([0,0, zdim/2]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(xdim, ydim, array([0,0,-zdim/2]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(zdim, ydim, array([xdim/2,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(zdim, ydim, array([-xdim/2,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(xdim, zdim, array([0, ydim/2,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(xdim, zdim, array([0,-ydim/2,0]), facet2body = CF.Cx(pi/2), name = '-Y')

        self.EXOCUBE = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .01)

        spec_coef = .002

        

        segments = 20
        angle = 2*pi/segments
        radius = 1.5/2
        side_length = 2*sin(angle/2)*radius
        lenght = 7.9
        cylinder_facets = []
        for theta in linspace(0, 2*pi, segments)[:-1]:
            pos = CF.Cz(theta)@array([radius,0,0])
            facet2body = CF.Cz(theta)@CF.Cy(pi/2)
            outer_facet = Facet(lenght, side_length, pos, facet2body = facet2body, name = 'cylinder', specular_coef = spec_coef)
            cylinder_facets.append(outer_facet)

        


        pZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,lenght/2]), name = '+Z', facet2body = identity(3), specular_coef = spec_coef)
        pZ.area = pi*radius**2
        nZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,-lenght/2]), name = '-Z', facet2body = CF.Cx(pi), specular_coef = spec_coef)
        nZ.area = pi*radius**2
        cylinder_facets.append(pZ)
        cylinder_facets.append(nZ)

        pX_panel = Facet(6, 1.5, array([6/2 + radius, 0, lenght/2 - 1.5/2]), name = '+X Panel', facet2body = CF.Cx(pi/2), double_sided = True, specular_coef = spec_coef)
        nX_panel = Facet(6, 1.5, array([-6/2 - radius, 0, lenght/2 - 1.5/2]), name = '-X Panel', facet2body = CF.Cx(pi/2), double_sided = True, specular_coef = spec_coef)
        cylinder_facets.append(pX_panel)
        cylinder_facets.append(nX_panel)

        dumypanel = Facet(lenght, radius/2, array([0,0,0]), facet2body = CF.Cy(pi/2), name = 'Fake Shadow', specular_coef = 0)
        
        self.SERT2 = Spacecraft_Geometry(cylinder_facets, sample_dim = .1)

        self.SERT2.obscuring_facets[pX_panel] = [dumypanel]
        self.SERT2.obscuring_facets[nX_panel] = [dumypanel]

        # for f in self.SERT2.obscuring_facets.keys():
        #     print(f.name, [x.name for x in self.SERT2.obscuring_facets[f]])


    def AJISAI(self):

        ajisai_file = open('AJISAI_coords','rb')
        coords = pickle.load(ajisai_file)
        ajisai_file.close()

        Radius = 1.075
        facets = []
        for lat in coords.keys():
            for lon in coords[lat]:
                rlat = lat/180*pi
                rlon = lon/180*pi
                
                facet2body = CF.Cz(rlon).T@CF.Cy(pi/2 - rlat)
                position  =  CF.Cz(rlon)@CF.Cy(rlat)@array([1,0,0])
                facet = Facet(15/100, 15/100, position, facet2body = facet2body, specular_coef = 0.002)
                facets.append(facet)

        AJISAI = Spacecraft_Geometry(facets, sample_dim = 1)
        for facet in AJISAI.facets:
            AJISAI.obscuring_facets[facet] = []

        return AJISAI



    def get_geometry(self, name):
        name = name.upper()
        if name == 'BOX_WING':
            return self.BOX_WING
        elif name == "BOX":
            return self.BOX
        elif name == 'PLATE':
            return self.PLATE
        elif name == 'CYLINDER':
            return self.CYLINDER
        elif name == 'RECTANGLE':
            return self.RECTANGLE
        elif name == 'LONG_RECTANGLE':
            return self.LONG_RECTANGLE
        elif name == 'ARIANE40':
            return self.ARIANE40
        elif name == 'EXOCUBE':
            return self.EXOCUBE
        elif name == 'SL8':
            return self.SL8
        elif name == 'SERT2':
            return self.SERT2
        elif name == 'AJISAI':
            return self.AJISAI()
        elif name == 'SPLOTCHY_RECTANGLE':
            return self.SPLOTCHY_RECTANGLE
        else:
            print(name, 'is not a valid geometry.')


def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} Loading:[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')

if __name__ == '__main__':

    #Bug testing code here.

    obs_vecs = load('obsvec.npy')[:50]
    sun_vecs = load('obsvec.npy')[:50]
    attitudes = load('mrps0.npy')[:50]

    Geometry = Premade_Spacecraft().EXOCUBE

    num_frames = len(obs_vecs)

    frames = []
    count = 0
    max_val = 0
    for mrps, obs_vec, sun_vec in zip(attitudes, obs_vecs, sun_vecs):
        dcm_eci2body = CF.mrp2dcm(mrps).T
        #dcm_eci2body = CF.mrp2dcm(mrps).T
        image = generate_image(Geometry, dcm_eci2body@obs_vec, dcm_eci2body@sun_vec, win_dim = (1,1), dpm = 100)
        im_max = amax(image)
        if im_max > max_val:
            max_val = im_max
        frames.append(image)
        count += 1
        loading_bar(count/num_frames, 'Rendering gif')

    frames = [frame/max_val for frame in frames]

    imageio.mimsave('test.gif',frames, fps = 10)









