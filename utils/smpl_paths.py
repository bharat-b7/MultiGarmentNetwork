import os
import numpy as np
from psbody.mesh import Mesh
from os.path import join
import cPickle as pkl
from lib.serialization import backwards_compatibility_replacements, load_model
from utils.geometry import get_hres
import scipy.sparse as sp

## Set your paths here
SMPL_PATH = '/BS/RVH/work/data/smpl_models/neutral/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
smpl_vt_ft_path = '/BS/bharat/work/MGN_final_release/assets/smpl_vt_ft.pkl'

class SmplPaths:
    def __init__(self, project_dir='', exp_name='', gender='neutral', garment=''):
        self.project_dir = project_dir
        # experiments name
        self.exp_name = exp_name
        self.gender = gender
        self.garment = garment

    def get_smpl_file(self):
        if self.gender == 'neutral':
            return SMPL_PATH

        else:
            raise(NotImplemented)

        return smpl_file

    def get_smpl(self):
        smpl_m = load_model(self.get_smpl_file())
        smpl_m.gender = self.gender
        return smpl_m

    def get_hres_smpl_model_data(self):

        dd = pkl.load(open(self.get_smpl_file()))
        backwards_compatibility_replacements(dd)

        hv, hf, mapping = get_hres(dd['v_template'], dd['f'])

        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'bs_style': dd['bs_style'],
            'J': dd['J'],
            'f': hf,
        }

        return model

    def get_hres_smpl(self):
        smpl_m = load_model(self.get_hres_smpl_model_data())
        smpl_m.gender = self.gender
        return smpl_m

    @staticmethod
    def get_vt_ft():
        vt, ft = pkl.load(open(smpl_vt_ft_path))
        return vt, ft

    @staticmethod
    def get_vt_ft_hres():
        vt, ft = SmplPaths.get_vt_ft()
        vt, ft, _ = get_hres(np.hstack((vt, np.ones((vt.shape[0], 1)))), ft)
        return vt[:, :2], ft

if __name__ == '__main__':
    dp = SmplPaths(gender='neutral')
    smpl_file = dp.get_smpl_file()

    print(smpl_file)
