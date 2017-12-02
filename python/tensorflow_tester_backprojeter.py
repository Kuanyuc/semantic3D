import numpy as np
import os
# from plyfile import PlyData, PlyElement
import scipy.misc
import pickle
from tqdm import *
import tensorflow as tf
import shutil
#import matplotlib.pyplot as plt

import pointcloud_tools.lib.python.PcTools as PcTls

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

import matplotlib as mpl 
mpl.use('TkAgg') 
import matplotlib.pyplot as plt


class BackProjeter:

    def __init__(self, model_function1, model_function2, model_fusion):
            self.model_function1 = model_function1
            self.model_function2 = model_function2
            self.model_function_fusion = model_fusion

    def label_to_color_image(self, label_image):
        """ converts an image with grayscale values into a color image"""
        color = {1 : np.array([192, 192, 192]),
        2 : np.array([0, 255, 0]),
        3 : np.array([38, 214, 64]),
        4 : np.array([247, 247, 0]),
        5 : np.array([255, 3, 0]),
        6 : np.array([122, 0, 255]),
        7 : np.array([0, 255, 255]),
        8 : np.array([255, 110, 206])
        }
        #create a new image with 3 channels
        label_color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
        for i in range(1,9):
            indices = np.transpose(np.where(label_image==i))
            label_color_image[indices[:,0],indices[:,1] , :] = color[i]
        return label_color_image


    def backProj(self,
        filename,
        label_nbr,
        dir_data,
        dir_images,
        imsize,
        input_ch,
        batch_size,
        saver_directory1,
        saver_directory2,
        saver_directoryFusion,
        images_root1,
        images_root2,
        variable_scope1,
        variable_scope2,
        variable_scope_fusion):

        # load mesh
        vertices = np.load(os.path.join(dir_data,filename+"_vertices.npz"))["arr_0"]
        faces = np.load(os.path.join(dir_data, filename+"_faces.npz"))["arr_0"].astype(int)

        # create score matrix
        scores = np.zeros((vertices.shape[0],label_nbr))
        counts = np.zeros(vertices.shape[0])

        dir_views = os.path.join(dir_images, "views")
        ### LOAD THE MODEL
        with tf.Graph().as_default() as g:

            images2 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            images1 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            is_training = tf.placeholder(bool)

            with tf.variable_scope(variable_scope1) as scope:
                deconv_net1, net1 = self.model_function1(images1, label_nbr, is_training)

            with tf.variable_scope(variable_scope2) as scope:
                deconv_net2, net2 = self.model_function2(images2, label_nbr, is_training)

            # create corresponding saver

            with tf.variable_scope(variable_scope_fusion) as scope:
                net_fusion, net = self.model_function_fusion(deconv_net1, deconv_net2, label_nbr)
                predictions = net_fusion[-1]

            # create saver
            saver1 = tf.train.Saver([v for v in tf.global_variables() if variable_scope1 in v.name])
            saver2 = tf.train.Saver([v for v in tf.global_variables() if variable_scope2 in v.name])
            saverFusion = tf.train.Saver([v for v in tf.global_variables() if variable_scope_fusion in v.name])

            sess = tf.Session()
            #init = tf.global_variables_initializer()
            #sess.run(init)

            ckpt = tf.train.get_checkpoint_state(saver_directory1)
            if ckpt and ckpt.model_checkpoint_path:
                saver1.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Error ...no checkpoint found...")
            ckpt = tf.train.get_checkpoint_state(saver_directory2)
            if ckpt and ckpt.model_checkpoint_path:
                saver2.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Error ...no checkpoint found...")

            ckpt = tf.train.get_checkpoint_state(saver_directoryFusion)
            #########
            # TODO Don't know why, needed to change the syntax
            # compared to previous weight loads
            #########
            model_checkpoint_path = saver_directoryFusion
            print(model_checkpoint_path)
            saverFusion.restore(sess, os.path.join(model_checkpoint_path, "model.ckpt"))


            # create the list of images in the folder
            directory1 = os.path.join(dir_images, images_root1)
            directory2 = os.path.join(dir_images, images_root2)
            files = []
            for file in os.listdir(directory1):
                if file.endswith(".png") and filename+"_" in file:
                    file = file.split(".")[:-1]
                    file = ".".join(file)
                    files.append(file)


            # load to get the size
            imsize = scipy.misc.imread(os.path.join(directory1,files[0]+".png")).shape




            # create batches
            batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

            # iterate over the batches
            for batch_files in tqdm(batches):

                # create the batch container
                batch_1= np.zeros((len(batch_files),imsize[0], imsize[1], imsize[2]), dtype=float)
                batch_2= np.zeros((len(batch_files),imsize[0], imsize[1], imsize[2]), dtype=float)
                for im_id in range(len(batch_files)):
                    batch_1[im_id] = scipy.misc.imread(os.path.join(directory1, batch_files[im_id]+".png"))
                    batch_2[im_id] = scipy.misc.imread(os.path.join(directory2, batch_files[im_id]+".png"))

                    #show the images 

                batch_1 /= 255
                batch_2 /= 255

                fd = {images1:batch_1, images2:batch_2, is_training:True}
                preds = sess.run(predictions, fd)

                # save the results
                for im_id,file in enumerate(batch_files):

                    indices = np.load(os.path.join(dir_views, file+".npz"))["arr_0"]
                    preds_ = preds[im_id]

                    #show predictions
                    print("Show predictions\n")
                    print(preds_.shape)

                    #print as images 
                    pred_image = preds_.argmax(axis=2)


                    #do a crf post-processing step here 
                    softmax = preds_.squeeze()

                    softmax = np.rollaxis(softmax, 2)
                    unary = softmax_to_unary(softmax)
                    unary = unary.reshape((10,-1)) # Needs to be flat.

                    # print(" softmax to unary")

                    # The inputs should be C-continious -- we are using Cython wrapper
                    unary = np.ascontiguousarray(unary)
                    # print(" contiguous array ")
                    image = preds_

                    d = dcrf.DenseCRF(preds_.shape[0] * preds_.shape[1], 10)
                    # print(" DenseCRF ")
                    d.setUnaryEnergy(unary)
                    # print(" Set unary energy ")
                    # This potential penalizes small pieces of segmentation that are
                    # spatially isolated -- enforces more spatially consistent segmentations
                    feats = create_pairwise_gaussian(sdims=(10, 10), shape=preds_.shape[:2])

                    d.addPairwiseEnergy(feats, compat=3,
                                        kernel=dcrf.DIAG_KERNEL,
                                        normalization=dcrf.NORMALIZE_SYMMETRIC)

                    # This creates the color-dependent features --
                    # because the segmentation that we get from CNN are too coarse
                    # and we can use local color features to refine them
                    feats = create_pairwise_bilateral(sdims=(5, 5), schan=(20, 20, 20),
                                                       img=np.uint8(batch_1[im_id]*255), chdim=2)

                    d.addPairwiseEnergy(feats, compat=20,
                                         kernel=dcrf.DIAG_KERNEL,
                                         normalization=dcrf.NORMALIZE_SYMMETRIC)


                    feats = create_pairwise_bilateral(sdims=(5, 5), schan=(20, 20, 20),
                                                       img=np.uint8(batch_2[im_id]*255), chdim=2)

                    d.addPairwiseEnergy(feats, compat=20,
                                         kernel=dcrf.DIAG_KERNEL,
                                         normalization=dcrf.NORMALIZE_SYMMETRIC)

                    Q = d.inference(5)

                    print(np.array(Q).shape)
                    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
                    Q = np.array(Q)
                    Q_preds = np.rollaxis(Q, 1).reshape((image.shape[0], image.shape[1], 10))

                    # cmap = plt.get_cmap('bwr')

                    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                    # ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
                    # ax1.set_title('Segmentation with CRF post-processing')
                    # probability_graph = ax2.imshow(np.dstack((train_annotation,)*3)*100)
                    # ax2.set_title('Ground-Truth Annotation')
                    # plt.show()

                    res_image = self.label_to_color_image(res)

                    #create a color image with the label colors 
                    pred_image = self.label_to_color_image(pred_image)

                    
                    fig = plt.figure()

                    a=fig.add_subplot(1,4,1)
                    imgplot = plt.imshow(np.uint8(batch_1[im_id]*255))
                    a.set_title('RGB')

                    a=fig.add_subplot(1,4,2)
                    imgplot = plt.imshow(np.uint8(batch_2[im_id]*255))
                    a.set_title('Geometry')

                    a=fig.add_subplot(1,4,3)
                    imgplot = plt.imshow(pred_image)
                    a.set_title('Prediction')

                    a=fig.add_subplot(1,4,4)
                    imgplot = plt.imshow(res_image)
                    a.set_title('CRF Prediction')
                    plt.show()
                    
                    # input("Press Enter to continue...")

                    indices = indices.reshape((-1))
                    indices[indices>faces.shape[0]] = 0
                    # preds_ = preds_.reshape((-1, preds_.shape[2]))
                    # scores[faces[indices-1][:,0]] += preds_
                    # scores[faces[indices-1][:,1]] += preds_
                    # scores[faces[indices-1][:,2]] += preds_

                    Q_preds = Q_preds.reshape((-1, Q_preds.shape[2]))
                    scores[faces[indices-1][:,0]] += Q_preds
                    scores[faces[indices-1][:,1]] += Q_preds
                    scores[faces[indices-1][:,2]] += Q_preds

                    counts[faces[indices-1][:,0]] += 1
                    counts[faces[indices-1][:,1]] += 1
                    counts[faces[indices-1][:,2]] += 1

            counts[counts ==0] = 1
            scores /= counts[:,None]

            # force 0 label to unseen vertices
            scores[scores.sum(axis=1)==0][0] = 1

            scores = scores.argmax(axis=1)

            self.scores = scores

            # close session
            del sess



    def saveScores(self,filename):
        np.savez(filename, self.scores)

    def createLabelPLY(self, filename,
        dir_data,
        save_dir):

        # create the semantizer
        #semantizer = Sem3D()
        semantizer =  PcTls.Semantic3D()
        semantizer.set_voxel_size(0.1)

        # loading data
        semantizer.set_vertices_numpy(os.path.join(dir_data,filename+"_vertices.npz"))
        semantizer.set_labels_numpy(os.path.join(save_dir, filename+"_scores.npz"))

        # removing unlabeled points
        semantizer.remove_unlabeled_points()

        # saving the labels
        semantizer.savePLYFile_labels(os.path.join(save_dir, filename+".ply"))
