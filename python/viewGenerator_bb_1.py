###
# IMPORTS
###

import numpy as np
import os
import argparse
import math
import time

# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtCore, QtGui

from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders as shaders
from OpenGL.arrays import vbo
from OpenGL.GL.ARB.texture_buffer_object import *
from PyQt5 import QtWidgets, QtCore, QtGui, QtOpenGL

import scipy.misc
import pickle
from tqdm import *
import sys
import numbers
import copy
from PIL import Image

class ViewGeneratorLauncher:

    def __init__(self):
        self.app = QtWidgets.QApplication([])

    def launch(self,view_generator):
        self.view_generator = view_generator
        self.view_generator.show()
        self.app.exec_()

    def exit(self):
        self.app.exit()

class ViewGeneratorBase(QtOpenGL.QGLWidget):

    def __init__(self):
        QtOpenGL.QGLWidget.__init__(self)
        # create options
        self.opts = {}
        self.opts["imsize"] = 128
        self.opts["fov"] = 60
        self.count_camera = 0

        self.opts["near_plane"] = 1
        self.opts["far_plane"] = 100



    def initialize_acquisition(self,
            dir_mesh,
            dir_images,
            filename,
        ):


        # save the paths
        self.dir_mesh = dir_mesh # root directory of acquisition
        self.dir_images = dir_images # directory to contain the images
        self.filename = filename # acquisition name
        self.dir_images_views = os.path.join(self.dir_images,"views")
        voxels_directory = os.path.join(self.dir_mesh, "voxels")

        # create the directory to save the views
        if not os.path.exists(self.dir_images_views):
            os.makedirs(self.dir_images_views)

        # load the point cloud
        self.faces = np.load(os.path.join(voxels_directory, filename+"_faces.npz"))["arr_0"].astype(int)
        self.vertices = np.load(os.path.join(voxels_directory,self.filename+"_vertices.npz"))["arr_0"]

        # give a unique color to each face
        self.face_colors = np.zeros((self.faces.shape[0],4), dtype=float)
        self.face_colors[:,3] = 255 # opaque face
        for i in range(self.faces.shape[0]):
            # black is for back ground ==> i+1
            self.face_colors[i,2] = (i+1) % 256
            self.face_colors[i,1] = ((i+1)//256)%256
            self.face_colors[i,0] = (((i+1)//256)//256)%256
        # self.face_colors /= 255. # pyqtgraph takes float colors
        self.face_colors = self.face_colors.astype(np.uint8)

        self.vtx = self.vertices[self.faces.ravel()]
        self.vtx_cls = self.face_colors.reshape((self.face_colors.shape[0], 1, self.face_colors.shape[1]))[:,:,:3]
        self.vtx_cls = np.repeat(self.vtx_cls,3, axis=1)
        self.vtx_cls = self.vtx_cls.reshape((-1, 3))


    def set_camera_generator(self, camera_generator_function):
        self.cam_gen_function = camera_generator_function

    # compute cartesian given spherical coordinates in rad
    def sphericalToCartesian(self,radius, azimuth, elevation):
        X = radius * np.cos(elevation) * np.cos(azimuth)
        Y = radius * np.cos(elevation) * np.sin(azimuth)
        Z = radius * np.sin(elevation)
        return X, Y, Z

    # function to fit with previous
    def sphericalToCamera(self, center, radius, azimuth, elevation):
        eyeX, eyeY, eyeZ = self.sphericalToCartesian(radius, azimuth, elevation)
        eyeX += center[0]
        eyeY += center[1]
        eyeZ += center[2]
        cam = {}
        cam["eyeX"] = eyeX
        cam["eyeY"] = eyeY
        cam["eyeZ"] = eyeZ
        cam["centerX"] = center[0]
        cam["centerY"] = center[1]
        cam["centerZ"] = center[2]
        cam["upX"] = 0
        cam["upY"] = 0
        cam["upZ"] = 1
        return cam

    def cam_generator_random_vertical_cone(self, camera_num, vertices_bb):
        """Cam generator for isprs 3D dataset."""
        cams = []
        for i in range(0,camera_num):
            #pt = self.vertices[np.random.randint(0, self.vertices.shape[0])]
            pt = vertices_bb[np.random.randint(0, vertices_bb.shape[0])]

            #print ("vertice shape:", self.vertices.shape)
            #print ("vertices_box shape:", vertices_bb.shape)

            print ("pt", pt)
            azimuth = (360*np.random.rand())/180 * np.pi
            elevation = (90 - 20*np.random.rand())/180 * np.pi #50
            for distance in self.opts["distances"]:
                cams.append(self.sphericalToCamera(pt, distance, azimuth, elevation))
            #print ("cams_length", len(cams)) 
        return cams


    # generate cameras at different distances
    def generate_cameras_scales(self, cam_number, distances=[5,10,20]):
        self.opts["distances"] = distances
        self.cameras=[]

        #print ("vertices", self.vertices)
        #print ("vetice_shape", self.vertices.shape)
        #print ("cam_id", camera_id)
        
        vertices_max = np.amax(self.vertices, axis=0)
        vertices_min = np.amin(self.vertices, axis=0)
        #print ("vertices_max", vertices_max)
        #print ("vertices_min", vertices_min)

        div_x = vertices_max[0] - vertices_min[0]
        div_y = vertices_max[1] - vertices_min[1]        
        #print ("div_x", div_x)
        #print ("div_y", div_y)

        ratio = div_x/div_y
        
        if ratio < 1/32:
           x_block = 1
           y_block = 64
        elif ratio >= 1/32 and ratio < 1/16:
           x_block = 2
           y_block = 32           
        elif ratio >= 1/16 and ratio < 1/2:
           x_block = 4
           y_block = 16        
        elif ratio >= 1/2 and ratio < 2/1:
           x_block = 8
           y_block = 8
        elif ratio >= 2/1 and ratio < 16/1:
           x_block = 16
           y_block = 4
        elif ratio >= 16/1 and ratio < 32/1:
           x_block = 32
           y_block = 2
        else:
           x_block = 64
           y_block = 1          
        
        #print ("x, y block:", x_block, y_block)

        x_block_size = div_x/x_block
        y_block_size = div_y/y_block
        #print ("x, y block size:", x_block_size, y_block_size)

        block_map = dict()

        for i in range(0,self.vertices.shape[0]):
        #for i in range(100000,100005):
            x_pos = self.vertices[i][0]
            y_pos = self.vertices[i][1]
            #print ("x, y pos:", x_pos, y_pos)

            block_num = int((x_pos - vertices_min[0])/x_block_size) + x_block*int((y_pos - vertices_min[1])/y_block_size)
            #print ("block_num", block_num, int((y_pos - vertices_min[1])/y_block_size))

            if block_num in block_map:
                block_map[block_num].append(self.vertices[i])
            else: 
                block_map[block_num] = [self.vertices[i]]

        #print ("block_map", block_map)
        #print ("block_map_1", len(block_map[4]))
        vertices_num = [0] * 64
        
        for i in range(0,64):
            if i in block_map:
                vertices_num[i] = len(block_map[i]) 
        #print ("vertices_num", vertices_num)
        
        vertice_cam_num = [0] * 64
        vertices_num_sum = sum(vertices_num)
        for i in range(0,64):
            vertice_cam_num[i] = math.ceil(cam_number*vertices_num[i]/vertices_num_sum)
        
        print ("cam_num", sum(vertice_cam_num))

        for i in range(0,64):
            if vertice_cam_num[i] != 0:
                print ("block", i)
                vertices_bb = np.array(block_map[i])
                #print ("vertices_bb.shape", vertices_bb.shape)
                cams = self.cam_gen_function(self, vertice_cam_num[i], vertices_bb)
                self.cameras+=cams
        pickle.dump( self.cameras, open( os.path.join(self.dir_images, self.filename+"_cameras.p"), "wb" ) )

    def init(self):
        self.resize(self.opts["imsize"], self.opts["imsize"])
        self.t = time.time()
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.update)
        self._update_timer.start(1e3 / 60.)
        self.program_close = False

    def lookAtFromCam(self, cam):
        gluLookAt( cam["eyeX"], cam["eyeY"], cam["eyeZ"],
            cam["centerX"], cam["centerY"], cam["centerZ"],
            cam["upX"], cam["upY"], cam["upZ"])

    def initializeGL(self):
        glViewport(0, 0, self.opts["imsize"], self.opts["imsize"])

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.opts["fov"], 1, self.opts["near_plane"], self.opts["far_plane"])
        glMatrixMode(GL_MODELVIEW)


    def draw_points(self):


        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointerf(self.vtx)
        glColorPointer( 3, GL_UNSIGNED_BYTE, 0, self.vtx_cls );
        glDrawArrays( GL_POINTS , 0,self.vtx.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);

    def draw_mesh(self):

        glColor3f(1.0,1.0,1.0)
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointerf(self.vtx)
        glColorPointer( 3, GL_UNSIGNED_BYTE, 0, self.vtx_cls );
        glDrawArrays( GL_TRIANGLES , 0,self.vtx.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);



class ViewGenerator(ViewGeneratorBase):
    def __init__(self):
        ViewGeneratorBase.__init__(self)

    # render function
    def paintGL(self):

        # exit if all camera have been snapped
        if self.count_camera >= len(self.cameras):
            time.sleep(1)
            self.program_close = True
            self.close()
        else:
            glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glLoadIdentity()

            self.lookAtFromCam(self.cameras[self.count_camera])

            # self.draw_points()
            self.draw_mesh()

            buffer = glReadPixels(0, 0, self.opts["imsize"], self.opts["imsize"], GL_RGB, GL_UNSIGNED_BYTE)
            im = Image.frombytes(mode="RGB", size=(self.opts["imsize"], self.opts["imsize"]), data=buffer)
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

            # image.save(os.path.join(self.view_directory, self.name+("_%04d" % self.count_camera)+".png"))
            im.save(os.path.join(self.dir_images,"views", self.filename+("_%04d" % self.count_camera)+".png"))

            im = np.asarray(im).copy()
            im[:,:,0] *= 256*256
            im[:,:,1] *= 256
            im = im.sum(axis=2)
            im -= 1 #we added 1 to handle the background

            # save the matrix
            np.savez(os.path.join(self.dir_images,"views", self.filename+("_%04d" % self.count_camera)), im)

            self.count_camera += 1

        self.update()
        time.sleep(1)


class ViewGeneratorNoDisplay(ViewGeneratorBase):
    def __init__(self):
        ViewGeneratorBase.__init__(self)

    # render function
    def paintGL(self):

        for cam_id in self.cameras:
            glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glLoadIdentity()

            self.lookAtFromCam(self.cameras[self.count_camera])

            # self.draw_points()
            self.draw_mesh()

            buffer = glReadPixels(0, 0, self.opts["imsize"], self.opts["imsize"], GL_RGB, GL_UNSIGNED_BYTE)
            im = Image.frombytes(mode="RGB", size=(self.opts["imsize"], self.opts["imsize"]), data=buffer)
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

            # im.save(os.path.join(self.dir_images,"views", self.filename+("_%04d" % self.count_camera)+".png"))

            im = np.asarray(im).copy().astype(int)
            im[:,:,0] *= 256*256
            im[:,:,1] *= 256
            im = im.sum(axis=2)
            im -= 1 #we added 1 to handle the background

            # save the matrix
            np.savez(os.path.join(self.dir_images,"views", self.filename+("_%04d" % self.count_camera)), im)

            self.count_camera += 1

        self.program_close = True
        self.close()
