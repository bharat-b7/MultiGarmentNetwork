'''
This loads scan and visualizes segmentation and texture on it
'''
import os
from os.path import split, join, exists
from glob import glob
import cPickle as pkl
from shutil import copyfile
from psbody.mesh import Mesh, MeshViewer
import numpy as np
import cv2

if __name__ == '__main__':
    path = '/BS/bharat/work/MGN_release/Multi-Garment_dataset/125611508622317'

    scan = Mesh(filename=join(path, 'scan.obj'))
    seg = np.load(join(path, 'scan_labels.npy'))
    tex_file = join(path, 'scan_tex.jpg')

    scan.set_texture_image(tex_file)
    scan.show()

    scan2 = Mesh(filename=join(path, 'scan.obj'))
    scan2.set_vertex_colors_from_weights(seg.reshape(-1,))
    scan2.show()

    print('Done')