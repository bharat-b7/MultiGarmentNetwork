import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def sparse_to_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse_reorder(tf.SparseTensor(indices, coo.data, coo.shape))


def sparse_dense_matmul_batch(a, b):
    num_b = tf.shape(b)[0]
    shape = a.dense_shape

    indices = tf.reshape(a.indices, (num_b, -1, 3))
    values = tf.reshape(a.values, (num_b, -1))

    def matmul((i, bb)):
        sp = tf.SparseTensor(indices[i, :, 1:], values[i], shape[1:])
        return i, tf.sparse_tensor_dense_matmul(sp, bb)

    _, p = tf.map_fn(matmul, (tf.range(num_b), b))

    return p


def sparse_dense_matmul_batch_tile(a, b):
    return tf.map_fn(lambda x: tf.sparse_tensor_dense_matmul(a, x), b)


def project_pool(v, img_feat):
    shape = tf.shape(img_feat)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    dim = shape[-1]
    num_v = tf.shape(v)[1]

    v /= tf.expand_dims(v[:, :, -1], -1)

    x = (v[:, :, 0] + 1) / 2. * tf.cast(width - 1, tf.float32)
    y = (1 - (v[:, :, 1] + 1) / 2.) * tf.cast(height - 1, tf.float32)

    # x = tf.Print(x, [tf.stack([x, y], 2)[0]], summarize=20)

    x1 = tf.floor(x)
    x2 = tf.ceil(x)
    y1 = tf.floor(y)
    y2 = tf.ceil(y)

    b = tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), (1, num_v))

    Q11 = tf.gather_nd(img_feat, tf.stack([b, tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 2))
    Q12 = tf.gather_nd(img_feat, tf.stack([b, tf.cast(x1, tf.int32), tf.cast(y2, tf.int32)], 2))
    Q21 = tf.gather_nd(img_feat, tf.stack([b, tf.cast(x2, tf.int32), tf.cast(y1, tf.int32)], 2))
    Q22 = tf.gather_nd(img_feat, tf.stack([b, tf.cast(x2, tf.int32), tf.cast(y2, tf.int32)], 2))

    weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y2, y))
    Q11 = tf.multiply(tf.tile(tf.expand_dims(weights, 2), [1, 1, dim]), Q11)

    weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y2, y))
    Q21 = tf.multiply(tf.tile(tf.expand_dims(weights, 2), [1, 1, dim]), Q21)

    weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y, y1))
    Q12 = tf.multiply(tf.tile(tf.expand_dims(weights, 2), [1, 1, dim]), Q12)

    weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y, y1))
    Q22 = tf.multiply(tf.tile(tf.expand_dims(weights, 2), [1, 1, dim]), Q22)

    outputs = tf.add_n([Q11, Q21, Q12, Q22])
    return outputs


def edge_lengths(v, e_idx):
    num_b = tf.shape(v)[0]
    num_e = tf.shape(e_idx)[0]

    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_e))

    e_idx_0 = tf.tile(tf.expand_dims(e_idx[:, 0], 0), (num_b, 1))
    e_idx_1 = tf.tile(tf.expand_dims(e_idx[:, 1], 0), (num_b, 1))

    indices_0 = tf.stack((batch_dim, e_idx_0), axis=2)
    indices_1 = tf.stack((batch_dim, e_idx_1), axis=2)

    v0 = tf.gather_nd(v, indices_0)
    v1 = tf.gather_nd(v, indices_1)

    return tf.reduce_sum(tf.pow(v0 - v1, 2), 2)


def batch_laplacian(v, f, return_sparse=True):
    # v: B x N x 3
    # f: M x 3

    num_b = tf.shape(v)[0]
    num_v = tf.shape(v)[1]
    num_f = tf.shape(f)[0]

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    a = tf.gather(v, v_a, axis=1)
    b = tf.gather(v, v_b, axis=1)
    c = tf.gather(v, v_c, axis=1)

    ab = a - b
    bc = b - c
    ca = c - a

    cot_a = -1 * tf.reduce_sum(ab * ca, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(ab, ca) ** 2, axis=-1))
    cot_b = -1 * tf.reduce_sum(bc * ab, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(bc, ab) ** 2, axis=-1))
    cot_c = -1 * tf.reduce_sum(ca * bc, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(ca, bc) ** 2, axis=-1))

    I = tf.tile(tf.expand_dims(tf.concat((v_a, v_c, v_a, v_b, v_b, v_c), axis=0), 0), (num_b, 1))
    J = tf.tile(tf.expand_dims(tf.concat((v_c, v_a, v_b, v_a, v_c, v_b), axis=0), 0), (num_b, 1))

    W = 0.5 * tf.concat((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a), axis=1)

    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_f * 6))

    indices = tf.reshape(tf.stack((batch_dim, J, I), axis=2), (num_b, 6, -1, 3))
    W = tf.reshape(W, (num_b, 6, -1))

    l_indices = [tf.cast(tf.reshape(indices[:, i], (-1, 3)), tf.int64) for i in range(6)]
    shape = tf.cast(tf.stack((num_b, num_v, num_v)), tf.int64)
    sp_L_raw = [tf.sparse_reorder(tf.SparseTensor(l_indices[i], tf.reshape(W[:, i], (-1,)), shape)) for i in range(6)]

    L = sp_L_raw[0]
    for i in range(1, 6):
        L = tf.sparse_add(L, sp_L_raw[i])

    dia_values = tf.sparse_reduce_sum(L, axis=-1) * -1

    I = tf.tile(tf.expand_dims(tf.range(num_v), 0), (num_b, 1))
    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_v))
    indices = tf.reshape(tf.stack((batch_dim, I, I), axis=2), (-1, 3))

    dia = tf.sparse_reorder(tf.SparseTensor(tf.cast(indices, tf.int64), tf.reshape(dia_values, (-1,)), shape))

    return tf.sparse_add(L, dia)


def compute_laplacian_diff(v0, v1, f):
    L0 = batch_laplacian(v0, f)
    L1 = batch_laplacian(v1, f)

    return sparse_dense_matmul_batch(L0, v0) - sparse_dense_matmul_batch(L1, v1)


def cpu_laplacian(v, f):
    n = len(v)

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    ab = v[v_a] - v[v_b]
    bc = v[v_b] - v[v_c]
    ca = v[v_c] - v[v_a]

    cot_a = -1 * (ab * ca).sum(axis=1) / np.sqrt(np.sum(np.cross(ab, ca) ** 2, axis=-1))
    cot_b = -1 * (bc * ab).sum(axis=1) / np.sqrt(np.sum(np.cross(bc, ab) ** 2, axis=-1))
    cot_c = -1 * (ca * bc).sum(axis=1) / np.sqrt(np.sum(np.cross(ca, bc) ** 2, axis=-1))

    I = np.concatenate((v_a, v_c, v_a, v_b, v_b, v_c))
    J = np.concatenate((v_c, v_a, v_b, v_a, v_c, v_b))
    W = 0.5 * np.concatenate((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a))

    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    L = L - sp.spdiags(L * np.ones(n), 0, n, n)

    return L


if __name__ == "__main__":
    # from psbody.mesh import Mesh
    # m0 = Mesh(filename='assets/sphube.obj')
    # m1 = Mesh(filename='assets/sphube.obj')
    # m1.v *= np.array([0.5, 1., 2.])

    from utils.smpl_paths import SmplPaths
    smp = SmplPaths()
    m0 = smp.get_mesh(smp.get_smpl())

    L0 = cpu_laplacian(m0.v.astype(np.float32), m0.f)
    lap0 = L0.dot(m0.v.astype(np.float32))

    tf_v0 = tf.expand_dims(m0.v.astype(np.float32), 0)
    tf_v = tf.tile(tf_v0, (5, 1, 1))

    tf_L = batch_laplacian(tf_v, m0.f.astype(np.int32))
    tf_L0 = batch_laplacian(tf_v0, m0.f.astype(np.int32))

    tf_lap = sparse_dense_matmul_batch(tf_L, tf_v)
    # tf_diff = tf.reduce_max(tf.abs(f_L[0] - tf_L[-1]))

    with tf.Session():
        tf_L_e = tf.sparse_tensor_to_dense(tf_L).eval()
        tf_lap_e = tf_lap.eval()

    print np.max(np.abs(tf_L_e[0] - L0.toarray()))
    print np.max(np.abs(tf_L_e[-1] - L0.toarray()))
    print tf_L_e.shape
    # print np.max(np.abs(tf_L0_e[0] - L0.toarray()))

    print np.max(np.abs(tf_lap_e[0] - lap0))
    print np.max(np.abs(tf_lap_e[-1] - lap0))
    print np.max(np.abs(tf_lap_e[0] - tf_lap_e[-1]))

    # from opendr.topology import get_vertices_per_edge
    # e_idx = get_vertices_per_edge(m.v, m.f)
    #
    # tf_v = tf.tile(tf.expand_dims(m.v.astype(np.float32), 0), (2, 1, 1))
    #
    # el = edge_lengths(tf_v, e_idx)
    #
    # with tf.Session():
    #     print(el.eval())
    #     print(el.eval().shape)
