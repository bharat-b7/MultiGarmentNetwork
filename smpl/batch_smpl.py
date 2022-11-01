""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import _pickle as pickle

import tensorflow as tf
# tf.enable_eager_execution()

from tensorflow.keras.layers import Layer

try:
    from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
except:
    from batch_lbs import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


def sparse_to_tensor(x, dtype=tf.float32):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, tf.convert_to_tensor(coo.data, dtype=dtype), coo.shape)


class SMPL(tf.keras.Model): #(object):
    def __init__(self, pkl_path, theta_in_rodrigues=True, theta_is_perfect_rotmtx=True, isHres = False, dtype=tf.float64, scale = True, name = None):
        super(SMPL, self).__init__(name = name)
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        self.isHres = isHres
        self.scale = scale
        self.data_type = dtype

        with open(pkl_path, 'r') as f:
            dd = pickle.load(f)
        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=dtype,
            trainable=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name='shapedirs', dtype=dtype, trainable=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = sparse_to_tensor(dd['J_regressor'], dtype=dtype)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name='posedirs', dtype=dtype, trainable=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        # self.weights = tf.Variable(
        #     undo_chumpy(dd['weights']),
        #     name='lbs_weights',
        #     dtype=dtype,
        #     trainable=False)
        self.weights_ = tf.Variable(undo_chumpy(dd['weights']), dtype=dtype, trainable=False)


        # expect theta in rodrigues form
        self.theta_in_rodrigues = theta_in_rodrigues

        # if in matrix form, is it already rotmax?
        self.theta_is_perfect_rotmtx = theta_is_perfect_rotmtx

        with open(os.path.join(os.path.dirname(__file__), '../assets/hresMapping.pkl'), 'rb') as f:
            mapping, nf = pickle.load(f)

        self.weights_hres = tf.cast(tf.Variable(np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]), trainable=False ), dtype= dtype)

        self.mapping = sparse_to_tensor(mapping, dtype=dtype)

    ##HighRes
    def fn(self, b):
        shape = tf.keras.backend.int_shape(b)
        with tf.device('/cpu:0'):
            return tf.reshape(tf.sparse_tensor_dense_matmul(self.mapping, tf.reshape(b, (-1, 1))), (-1, shape[-1]))


    def call(self, theta, beta, trans, v_personal):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3 (low res) N x 27554 x 3 (hres)
        """

        ## Cast inputs to float64
        theta, beta, trans, v_personal = tf.cast(theta, self.data_type), tf.cast(beta, self.data_type), tf.cast(trans, self.data_type), tf.cast(v_personal, self.data_type)

        if not self.isHres:
            v_personal = v_personal[:,:6890,:]

        # num_batch = int(tf.shape(beta)[0])
        num_batch = tf.keras.backend.int_shape(beta)[0]

        # 1. Add shape blend shapes
        v_shaped_scaled = tf.reshape(
            tf.matmul(beta, self.shapedirs, name='shape_bs'),
            [-1, self.size[0], self.size[1]]) + self.v_template

        if self.scale:
            body_height = (v_shaped_scaled[:, 2802, 1] + v_shaped_scaled[:, 6262, 1]) - (v_shaped_scaled[:, 2237, 1] + v_shaped_scaled[:, 6728, 1])
            scale = tf.reshape(1.66 / body_height, (-1, 1, 1))
        else:
            scale = tf.reshape(tf.ones_like(v_shaped_scaled[:, 2802, 1]), (-1, 1, 1))

        ## Scale to 1.66m height
        v_shaped_scaled *= scale

        v_shaped = v_shaped_scaled

        if self.isHres:
            v_shaped = tf.map_fn(self.fn, v_shaped, dtype = self.data_type)

        v_shaped_personal = v_shaped + v_personal

        # 2. Infer shape-dependent joint locations.
        ## Some gpu dont support float64 operations
        with tf.device('/cpu:0'):
            Jx = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_regressor, tf.transpose(v_shaped_scaled[:, :, 0])))
            Jy = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_regressor, tf.transpose(v_shaped_scaled[:, :, 1])))
            Jz = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_regressor, tf.transpose(v_shaped_scaled[:, :, 2])))
        J =  tf.stack([Jx, Jy, Jz], axis=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        if self.theta_in_rodrigues:
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
        else:
            if self.theta_is_perfect_rotmtx:
                Rs = theta
            else:
                s, u, v = tf.svd(theta)
                Rs = tf.matmul(u, tf.transpose(v, perm=[0, 1, 3, 2]))

        # with tf.name_scope("lrotmin"):
            # Ignore global rotation.
        pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3, dtype = Rs.dtype), [-1, 207])

        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = tf.reshape(
            tf.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]])

        if self.isHres:
            v_posed = tf.map_fn(self.fn, v_posed)
        v_posed += v_shaped_personal

        #4. Get the global joint location
        J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)
        J_transformed += tf.expand_dims(trans, axis=1)

        # 5. Do skinning:
        if self.isHres:
            W = tf.reshape(
                tf.tile(self.weights_hres, [num_batch, 1]), [num_batch, -1, 24])
        else:
        # W is N x 6890 x 24
            W = tf.reshape(
                tf.tile(self.weights_, [num_batch, 1]), [num_batch, -1, 24])
        # (N x 6890 x 24) x (N x 24 x 16)
        T = tf.reshape(
            tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
            [num_batch, -1, 4, 4])
        v_posed_homo = tf.concat(
            [v_posed, tf.ones([num_batch, v_posed.shape[1], 1], dtype = v_posed.dtype)], 2)

        v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0] #/ v_homo[:, :, 3, 0:1]

        verts_t = verts + tf.expand_dims(trans, axis=1)

        ##Return verts, unposed verts, unposed naked verts, joints
        return verts_t, v_shaped_personal, v_shaped, J_transformed