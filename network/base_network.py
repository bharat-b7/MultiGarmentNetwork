import cv2
import numpy as np
import _pickle as pkl
import matplotlib.pyplot as plt

from config_ver1 import config, NUM, IMG_SIZE, FACE
FOCAL_LENGTH, CAMERA_CENTER = [IMG_SIZE, IMG_SIZE], [IMG_SIZE/2., IMG_SIZE/2.]

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Conv2D, Reshape, MaxPool2D, Average, Dropout, Concatenate, \
    Add, Maximum, Layer, Activation, Conv1D, TimeDistributed, GlobalAvgPool2D
from tensorflow.keras import initializers


from render.generic_renderer import render_colored_batch, render_shaded_batch, perspective_projection
from smpl.smpl_layer import SmplBody25Layer
from smpl.batch_lbs import batch_rodrigues
from smpl.batch_smpl import SMPL
from mesh.geometry import compute_laplacian_diff

from custom_layers import PCA_, NameLayer, Scatter_

def reprojection(ytrue, ypred):
    b_size = tf.shape(ypred)[0]
    projection_matrix = perspective_projection(FOCAL_LENGTH, CAMERA_CENTER, IMG_SIZE, IMG_SIZE, .1, 10)
    projection_matrix = tf.tile(tf.expand_dims(projection_matrix, 0), (b_size, 1, 1))

    ypred_h = tf.concat([ypred, tf.ones_like(ypred[:, :, -1:])], axis=2)
    ypred_proj = tf.matmul(ypred_h, projection_matrix)
    ypred_proj /= tf.expand_dims(ypred_proj[:, :, -1], -1)

    return K.mean(K.square((ytrue[:, :, :2] - ypred_proj[:, :, :2]) * tf.expand_dims(ytrue[:, :, 2], -1)))

class GarmentNet(tf.keras.Model):
    def __init__(self, no_comp, garment_key, garmparams_sz):
        super(GarmentNet, self).__init__(name=garment_key)
        with open(
                'assets/garment_basis_{}_temp20/{}_param_{}_corrected.pkl'.format(no_comp, garment_key, no_comp),
                'rb') as f:
            pca = pkl.load(f)

        ## Define layers
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(512, activation='relu')
        self.d4 = Dense(1024, activation='relu', name = 'extra')

        self.pca_comp = Dense(garmparams_sz, kernel_initializer=initializers.RandomNormal(0, 0.0005), name='pca_comp')
        self.PCA_ = PCA_(pca.components_.astype('float32'), pca.mean_.astype('float32'))
        self.bypass = Dense(len(pca.mean_), kernel_initializer=initializers.RandomNormal(0, 0.0005),
                           activation='tanh', name='byPass')
        self.limit_sz = Lambda(lambda z: z*0.05, name = 'limit_sz')
        # self.limit_sz = Lambda(lambda z: tf.clip_by_value(z, -0.005, 0.005), name='limit_sz')
        self.reshape = Reshape((len(pca.mean_)/3, 3))

        # self.leakyrelu = tf.keras.layers.ReLU()

    def call(self, inp):
        with tf.device('/gpu:3'):
            x_ = self.d1(inp)
            x_ = self.d2(x_)
            x_ = self.d3(x_)

            pca_comp = self.pca_comp(x_)
            x = self.PCA_(pca_comp)

            bypass = self.d4(x_)
            bypass = self.bypass(bypass)
            bypass = self.limit_sz(bypass)
            bypass = self.reshape(bypass)

            y = x + bypass

            return [y, pca_comp]


class SingleImageNet(tf.keras.Model):
    def __init__(self, latent_code_garms_sz, latent_code_betas_sz, name=None):
        super(SingleImageNet, self).__init__(name=name)

        ## Define layers
        if IMG_SIZE >= 1000:
            self.conv1 = Conv2D(8, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')
            self.conv1_1 = Conv2D(16, (3, 3), strides=(2, 2), kernel_initializer='he_normal', padding='same',
                                  activation='relu')
        else:
            self.conv1 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
            self.conv1_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')

        self.conv2 = Conv2D(16, (3, 3),  # strides=(2, 2),
                            kernel_initializer='he_normal', padding='same', activation='relu')
        self.conv2_1 = Conv2D(32, (3, 3),  # strides=(2, 2),
                              kernel_initializer='he_normal', padding='same', activation='relu')

        self.conv3 = Conv2D(16, (3, 3),  # strides=(2, 2),
                            kernel_initializer='he_normal', padding='same', activation='relu')
        self.conv3_1 = Conv2D(32, (3, 3),  # strides=(2, 2),
                              kernel_initializer='he_normal', padding='same', activation='relu')
        sz = 16
        self.conv4 = Conv2D(sz, (3, 3),  # strides=(2, 2),
                            kernel_initializer='he_normal', padding='same', activation='relu')
        self.conv4_1 = Conv2D(sz, (3, 3),  # strides=(2, 2),
                              kernel_initializer='he_normal', padding='same', activation='relu')

        self.conv5 = Conv2D(sz, (3, 3), kernel_initializer='he_normal', padding='same',
                            activation='relu')
        self.conv5_1 = Conv2D(sz, (3, 3), kernel_initializer='he_normal', padding='same',
                              activation='relu')

        self.conv6 = Conv2D(sz, (3, 3), kernel_initializer='he_normal', padding='same',
                            activation='relu')
        self.conv6_1 = Conv2D(sz, (3, 3), kernel_initializer='he_normal', padding='same',
                              activation='relu')
        self.conv7 = Conv2D(sz, (3, 3), kernel_initializer='he_normal', padding='same',
                            activation='relu')
        self.conv7_1 = Conv2D(sz, (3, 3), kernel_initializer='he_normal', padding='same',
                              activation='relu')

        self.split_shape = Lambda(lambda z: z[..., :int(sz / 2)])
        self.split_garms = Lambda(lambda z: z[..., int(sz / 2):])

        self.flatten = Flatten()

        self.dg = Dense(latent_code_garms_sz, kernel_initializer='he_normal', activation='relu')
        self.db = Dense(latent_code_betas_sz, kernel_initializer='he_normal', activation='relu')

        self.dg2 = Dense(int(latent_code_garms_sz / 2), kernel_initializer='he_normal', activation='relu')
        self.db2 = Dense(latent_code_betas_sz, kernel_initializer='he_normal', activation='relu')

        self.dg3 = Dense(int(latent_code_garms_sz / 2), kernel_initializer='he_normal', activation='relu')
        self.db3 = Dense(latent_code_betas_sz, kernel_initializer='he_normal', activation='relu')

        self.latent_code_garms = Dense(int(latent_code_garms_sz / 2), kernel_initializer='he_normal',
                                       name='latent_garms', activation='relu')
        self.latent_code_shape = Dense(latent_code_betas_sz, kernel_initializer='he_normal', name='latent_shape',
                                       activation='relu')

        self.concat = Concatenate()

    def append_coord(self, x):
        a, b = tf.meshgrid(range(K.int_shape(x)[1]), range(K.int_shape(x)[2]))
        a = tf.cast(a, tf.float32) / K.int_shape(x)[1]
        b = tf.cast(b, tf.float32) / K.int_shape(x)[2]
        a = tf.tile(tf.expand_dims(tf.stack([a, b], axis=-1), 0), [K.int_shape(x)[0], 1, 1, 1])
        x = self.concat([x, a])
        return x

    def call(self, inp):
        inp, J_2d = inp
        with tf.device('/gpu:1'):
            x = self.conv1(inp)
            x = self.append_coord(x)
            x = self.conv1_1(x)
            z = MaxPool2D((2, 2))(x)

            z = self.append_coord(z)
            x = self.conv2(z)
            x = self.append_coord(x)
            x = self.conv2_1(x)
            z = MaxPool2D((2, 2))(x)

        with tf.device('/gpu:1'):
            z = self.append_coord(z)
            x = self.conv3(z)
            x = self.append_coord(x)
            x = self.conv3_1(x)
            z = MaxPool2D((2, 2))(x)

            z = self.append_coord(z)
            x = self.conv5(z)
            x = self.append_coord(x)
            x = self.conv5_1(x)

            split_shape = self.split_shape(x)
            split_garms = self.split_garms(x)

            split_shape = self.append_coord(split_shape)
            split_shape = self.conv6(split_shape)

            split_garms = self.append_coord(split_garms)
            split_garms = self.conv7(split_garms)
            flat_garms = self.flatten(split_garms)
            flat_shape = self.flatten(split_shape)

            flat_garms = Concatenate()([flat_garms, J_2d])
            flat_garms = self.dg(flat_garms)
            flat_garms = self.dg2(flat_garms)
            flat_garms = self.dg3(flat_garms)
            latent_code_garms = self.latent_code_garms(flat_garms)

            flat_shape = Concatenate()([flat_shape, J_2d])
            flat_shape = self.db(flat_shape)
            flat_shape = self.db2(flat_shape)
            flat_shape = self.db3(flat_shape)
            latent_code_shape = self.latent_code_shape(flat_shape)

            return [latent_code_garms, latent_code_shape]

class BaseModel(tf.keras.Model):
    def __init__(self, name = None):
        super(BaseModel, self).__init__(name=name)

        self.config = config
        self.model = None
        self.garmentModels = []
        if self.config:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.train.lr)
        else:
            self.optimizer = tf.train.AdamOptimizer(0.001)
        self.vertSpread = pkl.load(open('assets/vert_spread.pkl', 'rb') , encoding='latin1')
        self.garmentModels = []

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")


class PoseShapeOffsetModel(BaseModel):

    def __init__(self, config, latent_code_garms_sz=1024, garmparams_sz=config.PCA_, name=None):
        super(PoseShapeOffsetModel, self).__init__(name=name)

        self.config = config
        self.latent_code_garms_sz = latent_code_garms_sz
        self.garmparams_sz = garmparams_sz
        self.latent_code_betas_sz = 128

        ##ToDo: Minor: Remove hard coded colors. Should be same as rendered colors in input
        self.colormap = tf.cast(
            [np.array([255, 255, 255]), np.array([65, 0, 65]), np.array([0, 65, 65]), np.array([145, 65, 0]),
             np.array([145, 0, 65]),
             np.array([0, 145, 65])], tf.float32) / 255.
        with open('assets/hresMapping.pkl', 'rb') as f:
            _, self.faces = pkl.load(f)
        self.faces = np.int32(self.faces)

        ## Define network layers
        self.top_ = SingleImageNet(self.latent_code_garms_sz, self.latent_code_betas_sz)

        for n in self.config.garmentKeys:
            gn = GarmentNet(self.config.PCA_, n, self.garmparams_sz)
            self.garmentModels.append(gn)
        self.smpl = SMPL('assets/neutral_smpl.pkl',
                         theta_in_rodrigues=False, theta_is_perfect_rotmtx=False, isHres=True, scale=True)
        self.smpl_J = SmplBody25Layer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=False, isHres=True)
        self.J_layers = [NameLayer('J_{}'.format(i)) for i in range(NUM)]

        self.lat_betas = Dense(self.latent_code_betas_sz, kernel_initializer=initializers.RandomNormal(0, 0.00005),
                               activation='relu')
        self.betas = Dense(10, kernel_initializer=initializers.RandomNormal(0, 0.000005), name='betas')

        init_trans = np.array([0, 0.2, -2.])
        init_pose = np.load('assets/mean_a_pose.npy')
        init_pose[:3] = 0
        init_pose = tf.reshape(batch_rodrigues(init_pose.reshape(-1, 3).astype(np.float32)), (-1,))
        self.pose_trans = tf.concat((init_pose, init_trans), axis=0)

        self.lat_pose = Dense(self.latent_code_betas_sz, kernel_initializer=initializers.RandomNormal(0, 0.000005),
                              activation='relu')
        self.lat_pose_layer = Dense(24 * 3 * 3 + 3, kernel_initializer=initializers.RandomNormal(0, 0.000005),
                                    name='pose_trans')
        self.cut_trans = Lambda(lambda z: z[:, -3:])
        self.trans_layers = [NameLayer('trans_{}'.format(i)) for i in range(NUM)]

        self.cut_poses = Lambda(lambda z: z[:, :-3])
        self.reshape_pose = Reshape((24, 3, 3))
        self.pose_layers = [NameLayer('pose_{}'.format(i)) for i in range(NUM)]

        ## Optional: Condition garment on betas, probably not
        self.latent_code_offset_ShapeMerged = Dense(self.latent_code_garms_sz + self.latent_code_betas_sz,
                                                    activation='relu')
        self.latent_code_offset_ShapeMerged_2 = Dense(self.latent_code_garms_sz + self.latent_code_betas_sz,
                                                      activation='relu', name='latent_code_offset_ShapeMerged')

        self.avg = Average()
        self.flatten = Flatten()
        self.concat = Concatenate()

        self.scatters = []
        for vs in self.vertSpread:
            self.scatters.append(Scatter_(vs, self.config.NVERTS))

    def call(self, inp):
        images, vertexlabel, Js_in = inp
        out_dict = {}
        images = [tf.Variable(x, dtype=tf.float32, trainable=False) for x in images]
        vertexlabel = tf.cast(tf.Variable(vertexlabel, trainable=False), tf.int32)
        if FACE:
            Js = [Lambda(lambda j: j[:, :25])(J) for J in Js_in]
        else:
            Js = [self.flatten(tf.cast(tf.Variable(x, trainable=False), tf.float32)) for x in Js_in]

        with tf.device('/gpu:1'):
            lat_codes = [self.top_([q, j]) for q, j in zip(images, Js)]
            latent_code_offset = self.avg([q[0] for q in lat_codes])
            latent_code_betas = self.avg([q[1] for q in lat_codes])
            latent_code_pose = [tf.concat([q[1], x], axis=-1) for q, x in zip(lat_codes, Js)]

        with tf.device('/gpu:2'):
            latent_code_betas = self.lat_betas(latent_code_betas)
            betas = self.betas(latent_code_betas)

            latent_code_pose = [self.lat_pose(x) for x in latent_code_pose]

            pose_trans_init = tf.tile(tf.expand_dims(self.pose_trans, 0), (K.int_shape(betas)[0], 1))

            poses_ = [self.lat_pose_layer(x) + pose_trans_init for x in latent_code_pose]
            trans_ = [self.cut_trans(x) for x in poses_]
            trans = [la(i) for la, i in zip(self.trans_layers, trans_)]

            poses_ = [self.cut_poses(x) for x in poses_]
            poses_ = [self.reshape_pose(x) for x in poses_]
            poses = [la(i) for la, i in zip(self.pose_layers, poses_)]

            ##
            out_dict['betas'] = betas
            for i in range(NUM):
                out_dict['pose_{}'.format(i)] = poses[i]
                out_dict['trans_{}'.format(i)] = trans[i]

            latent_code_offset_ShapeMerged = self.latent_code_offset_ShapeMerged(latent_code_offset)
            latent_code_offset_ShapeMerged = self.latent_code_offset_ShapeMerged_2(latent_code_offset_ShapeMerged)

            garm_model_outputs = [fe(latent_code_offset_ShapeMerged) for fe in self.garmentModels]
            garment_verts_all = [fe[0] for fe in garm_model_outputs]
            garment_pca = [fe[1] for fe in garm_model_outputs]
            garment_pca = tf.stack(garment_pca, axis=1)

            ##
            out_dict['pca_verts'] = garment_pca

            lis = []
            for go, vs in zip(garment_verts_all, self.scatters):
                lis.append(vs(go))
            garment_verts_all_scattered = tf.stack(lis, axis=-1)

            ## Get naked smpl to compute garment offsets
            zerooooooo = K.zeros_like(garment_verts_all_scattered[..., 0])
            pooooooooo = [K.zeros_like(p) for p in poses]
            tooooooooo = [K.zeros_like(p) for p in trans]

            smpls_base = []
            for i, (p, t) in enumerate(zip(pooooooooo, tooooooooo)):
                v, _, n, _ = self.smpl(p, betas, t, zerooooooo)
                smpls_base.append(v)
                if i == 0:
                    vertices_naked_ = n

            ## Append Skin offsets
            garment_verts_all_scattered = tf.concat(
                [K.expand_dims(vertices_naked_, -1), tf.cast(garment_verts_all_scattered, vertices_naked_.dtype)],
                axis=-1)
            garment_verts_all_scattered = tf.transpose(garment_verts_all_scattered, perm=[0, 1, 3, 2])
            clothed_verts = tf.batch_gather(garment_verts_all_scattered, vertexlabel)
            clothed_verts = tf.squeeze(tf.transpose(clothed_verts, perm=[0, 1, 3, 2]))

            offsets_ = clothed_verts - vertices_naked_

            smpls = []
            for i, (p, t) in enumerate(zip(poses, trans)):
                v, t, n, _ = self.smpl(p, betas, t, offsets_)
                smpls.append(v)
                if i == 0:
                    vertices_naked = n
                    vertices_tposed = t

            Js = [jl(self.smpl_J([p, betas, t])) for jl, p, t in zip(self.J_layers, poses, trans)]
            vertices = tf.concat([tf.expand_dims(smpl, axis=-1) for i, smpl in enumerate(smpls)], axis=-1)

            ##
            out_dict['vertices'] = vertices
            out_dict['vertices_tposed'] = vertices_tposed
            out_dict['vertices_naked'] = vertices_naked

            ##
            out_dict['vertices'] = vertices
            out_dict['vertices_tposed'] = vertices_tposed
            out_dict['vertices_naked'] = vertices_naked
            out_dict['offsets_h'] = offsets_
            for i in range(NUM):
                out_dict['J_{}'.format(i)] = Js[i]

            vert_cols = tf.reshape(tf.gather(self.colormap, tf.reshape(vertexlabel, (-1,))), (-1, config.NVERTS, 3))
            renderered_garms_all = []

        for view in range(NUM):
            renderered_garms_all.append(render_colored_batch(vertices[..., view], self.faces, vert_cols,  # [bat],
                                                             IMG_SIZE, IMG_SIZE, FOCAL_LENGTH,
                                                             CAMERA_CENTER,
                                                             np.zeros(3, dtype=np.float32),
                                                             num_channels=3))

        renderered_garms_all = tf.transpose(renderered_garms_all, [1, 2, 3, 4, 0])
        out_dict['rendered'] = renderered_garms_all

        lap = compute_laplacian_diff(vertices_tposed, vertices_naked, self.faces)
        ##
        out_dict['laplacian'] = lap
        return out_dict

    @staticmethod
    def loss_model(gt_dict, pred_dict, wt_dict):
        loss = {}
        for k in wt_dict:
            if k == 'pca_verts':
                loss[k] = tf.losses.absolute_difference(gt_dict[k] * gt_dict['garments'][..., np.newaxis],
                                                        pred_dict[k] * gt_dict['garments'][..., np.newaxis],
                                                        weights=wt_dict[k])
            elif 'J_2d' in k:
                loss[k] = reprojection(tf.cast(gt_dict[k.replace('_2d', '')], tf.float32),
                                       tf.cast(pred_dict[k.replace('_2d', '')], tf.float32)) * wt_dict[k]
            else:
                loss[k] = tf.losses.absolute_difference(gt_dict[k], pred_dict[k], weights=wt_dict[k])

        return loss

    def train(self, inp_dict, gt_dict, loss_dict, vars2opt=None):
        images = [inp_dict['image_{}'.format(i)].astype('float32') for i in range(NUM)]
        vertex_label = inp_dict['vertexlabel'].astype('int64')
        J_2d = [inp_dict['J_2d_{}'.format(i)].astype('float32') for i in range(NUM)]

        with tf.GradientTape() as gtape:
            out_dict = self.call([images, vertex_label, J_2d])

            loss = self.loss_model(gt_dict, out_dict, wt_dict=loss_dict)
            loss_ = 0
            for k in loss:
                loss_ += loss[k]

            grad = gtape.gradient(loss_, self.trainable_variables)

            if vars2opt is not None:
                opt_var, opt_grad = [], []
                for x, g in zip(self.trainable_variables, grad):
                    if x.name in vars2opt:
                        opt_var.append(x)
                        opt_grad.append(g)
                self.optimizer.apply_gradients(
                    zip(opt_grad, opt_var))
            else:
                self.optimizer.apply_gradients(
                    zip(grad, self.trainable_variables))
        return loss