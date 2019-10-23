'''
Code to dress SMPL with registered garments.
Set the "path" variable in this code to the downloaded Multi-Garment Dataset

If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
Shout out to Chaitanya for intersection removal code
'''

from psbody.mesh import Mesh, MeshViewers
import numpy as np
import cPickle as pkl
from utils.smpl_paths import SmplPaths
from lib.ch_smpl import Smpl
from utils.interpenetration_ind import remove_interpenetration_fast
from os.path import join, split
from glob import glob

def load_smpl_from_file(file):
    dat = pkl.load(open(file))
    dp = SmplPaths(gender=dat['gender'])
    smpl_h = Smpl(dp.get_hres_smpl_model_data())

    smpl_h.pose[:] = dat['pose']
    smpl_h.betas[:] = dat['betas']
    smpl_h.trans[:] = dat['trans']

    return smpl_h

def pose_garment(garment, vert_indices, smpl_params):
    '''
    :param smpl_params: dict with pose, betas, v_template, trans, gender
    '''
    dp = SmplPaths(gender=smpl_params['gender'])
    smpl = Smpl(dp.get_hres_smpl_model_data() )
    smpl.pose[:] = 0
    smpl.betas[:] = smpl_params['betas']
    # smpl.v_template[:] = smpl_params['v_template']

    offsets = np.zeros_like(smpl.r)
    offsets[vert_indices] = garment.v - smpl.r[vert_indices]
    smpl.v_personal[:] = offsets
    smpl.pose[:] = smpl_params['pose']
    smpl.trans[:] = smpl_params['trans']

    mesh = Mesh(smpl.r, smpl.f).keep_vertices(vert_indices)
    return mesh

def retarget(garment_mesh, src, tgt):
    '''
    For each vertex finds the closest point and
    :return:
    '''
    from psbody.mesh import Mesh
    verts, _ = src.closest_vertices(garment_mesh.v)
    verts = np.array(verts)
    tgt_garment = garment_mesh.v - src.v[verts] + tgt.v[verts]
    return Mesh(tgt_garment, garment_mesh.f)

def dress(smpl_tgt, body_src, garment, vert_inds, garment_tex = None):
    '''
    :param smpl: SMPL in the output pose
    :param garment: garment mesh in t-pose
    :param body_src: garment body in t-pose
    :param garment_tex: texture file
    :param vert_inds: vertex association b/w smpl and garment
    :return:
    To use texture files, garments must have vt, ft
    '''
    tgt_params = {'pose': np.array(smpl_tgt.pose.r), 'trans': np.array(smpl_tgt.trans.r), 'betas': np.array(smpl_tgt.betas.r), 'gender': 'neutral'}
    smpl_tgt.pose[:] = 0
    body_tgt = Mesh(smpl_tgt.r, smpl_tgt.f)

    ## Re-target
    ret = retarget(garment, body_src, body_tgt)

    ## Re-pose
    ret_posed = pose_garment(ret, vert_inds, tgt_params)
    body_tgt_posed = pose_garment(body_tgt, range(len(body_tgt.v)), tgt_params)

    ## Remove intersections
    ret_posed_interp = remove_interpenetration_fast(ret_posed, body_tgt_posed)
    ret_posed_interp.vt = garment.vt
    ret_posed_interp.ft = garment.ft
    ret_posed_interp.set_texture_image(garment_tex)

    return ret_posed_interp

path = '/BS/bharat/work/MGN_release/Multi-Garment_dataset/'
all_scans = glob(path + '*')
garment_classes = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat', 'LongCoat']
gar_dict = {}
for gar in garment_classes:
    gar_dict[gar] = glob(join(path, '*', gar + '.obj'))

if __name__ == '__main__':
    dp = SmplPaths()
    vt, ft = dp.get_vt_ft_hres()
    smpl = Smpl(dp.get_hres_smpl_model_data())

    ## This file contains correspondances between garment vertices and smpl body
    fts_file = 'assets/garment_fts.pkl'
    vert_indices, fts = pkl.load(open(fts_file))
    fts['naked'] = ft

    ## Choose any garmet type as source
    garment_type = 'TShirtNoCoat'
    index = np.random.randint(0, len(gar_dict[garment_type]))   ## Randomly pick from the digital wardrobe
    path = split(gar_dict[garment_type][index])[0]


    garment_org_body_unposed = load_smpl_from_file(join(path, 'registration.pkl'))
    garment_org_body_unposed.pose[:] = 0
    garment_org_body_unposed.trans[:] = 0
    garment_org_body_unposed = Mesh(garment_org_body_unposed.v, garment_org_body_unposed.f)

    garment_unposed = Mesh(filename=join(path, garment_type + '.obj'))
    garment_tex = join(path, 'multi_tex.jpg')

    ## Generate random SMPL body (Feel free to set up ur own smpl) as target subject
    smpl.pose[:] = np.random.randn(72) *0.05
    smpl.betas[:] = np.random.randn(10) *0.01
    smpl.trans[:] = 0
    tgt_body = Mesh(smpl.r, smpl.f)

    vert_inds = vert_indices[garment_type]
    garment_unposed.set_texture_image(garment_tex)

    new_garment = dress(smpl, garment_org_body_unposed, garment_unposed, vert_inds, garment_tex)

    mvs = MeshViewers((1, 2))
    mvs[0][0].set_static_meshes([garment_unposed])
    mvs[0][1].set_static_meshes([new_garment, tgt_body])

    print('Done')

