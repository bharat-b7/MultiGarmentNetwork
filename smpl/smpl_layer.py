import tensorflow as tf

from batch_smpl import SMPL
import _pickle as pkl

from mesh.geometry import sparse_to_tensor, sparse_dense_matmul_batch_tile


class SmplBody25Layer(tf.keras.Model):

    def __init__(self, model='assets/neutral_smpl.pkl', theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, isHres=False, **kwargs):
        super(SmplBody25Layer, self).__init__(**kwargs)
        self.isHres = isHres
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx, isHres=isHres)
        self.body_25_reg = sparse_to_tensor(
                pkl.load(open('assets/J_regressor.pkl', 'rb', 'rb') , encoding='latin1').T)

    def joints_body25(self, v):
        with tf.device('cpu:0'):
            return sparse_dense_matmul_batch_tile(tf.cast(self.body_25_reg, v.dtype), v)

    def call(self, (pose, betas, trans)):
        if self.isHres:
            v_personal = tf.tile(tf.zeros((1, 27554, 3)), (tf.shape(betas)[0], 1, 1))
        else:
            v_personal = tf.tile(tf.zeros((1, 6890, 3)), (tf.shape(betas)[0], 1, 1))
        v, _, _, _ = self.smpl(pose, betas, trans, v_personal)

        return self.joints_body25(v[:,:6890,:])

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 25, 3
