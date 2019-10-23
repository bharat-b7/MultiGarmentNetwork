'''
This code visualises registered garment on the original smpl body
If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
'''
import os
from os.path import exists, join, split
from glob import glob
import numpy as np
import cPickle as pkl
from psbody.mesh import Mesh, MeshViewer, MeshViewers

from utils.smpl_paths import SmplPaths
from lib.ch_smpl import Smpl
from dress_SMPL import load_smpl_from_file, pose_garment
from utils.interpenetration_ind import remove_interpenetration_fast

path = '/BS/bharat/work/MGN_release/Multi-Garment_dataset/'
all_scans = glob(path + '*')
garment_classes = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat', 'LongCoat']
gar_dict = {}
for gar in garment_classes:
    gar_dict[gar] = glob(join(path, '*', gar + '.obj'))

def visualize_garment(garment_path, with_tex = True):
    ## Load SMPL body for the garment
    path = split(garment_path)[0]
    garment_org_body = load_smpl_from_file(join(path, 'registration.pkl'))
    garment_org_body = Mesh(garment_org_body.v, garment_org_body.f)

    ## Load unposed garment
    garment_unposed = Mesh(filename=gar_dict[garment_type][index])
    garment_unposed.set_texture_image(join(path, 'multi_tex.jpg'))

    ## Pose garments
    dat = pkl.load(open(join(path, 'registration.pkl')))
    dat['gender'] = 'neutral'
    garment_posed = pose_garment(garment_unposed, vert_indices[garment_type], dat)
    garment_posed = remove_interpenetration_fast(garment_posed, garment_org_body)

    if with_tex:
        garment_posed.vt = garment_unposed.vt
        garment_posed.ft = garment_unposed.ft
        garment_posed.set_texture_image(join(path, 'multi_tex.jpg'))


    mvs = MeshViewers((1, 3), keepalive=True)
    mvs[0][2].set_background_color(np.array([1,1,1]))
    mvs[0][1].set_background_color(np.array([1,1,1]))
    mvs[0][0].set_background_color(np.array([1,1,1]))
    mvs[0][1].set_static_meshes([garment_org_body])
    mvs[0][2].set_static_meshes([garment_org_body, garment_posed])
    mvs[0][0].set_static_meshes([garment_unposed])

    return

if __name__ == '__main__':
    dp = SmplPaths()
    vt, ft = dp.get_vt_ft_hres()
    smpl = Smpl(dp.get_hres_smpl_model_data())

    ## This file contains correspondances between garment vertices and smpl body
    fts_file = 'assets/garment_fts.pkl'
    vert_indices, fts = pkl.load(open(fts_file))
    fts['naked'] = ft

    ## Choose any garmet type
    garment_type = 'Pants'
    index = np.random.randint(0, len(gar_dict[garment_type]))   ## Randomly pick from the digital wardrobe

    visualize_garment(gar_dict[garment_type][index])

    print('Done')