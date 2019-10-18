import sys
# sys.path.append('/BS/alldieck-3dpeople/work/lib/dirt/')
import numpy as np
import tensorflow as tf


from tensorflow.python.framework import ops
import dirt
from dirt import matrices
sys.stderr.write('Using dirt renderer.\n')

from lighting import split_vertices_by_face, diffuse_directional, vertex_normals_pre_split, diffuse_point

def perspective_projection(f, c, w, h, near=0.1, far=10., name=None):
    """Constructs a perspective projection matrix.
    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.
    Returns:
        a 4x4 `Tensor` containing the projection matrix
    """
    # import dirt
    import tensorflow as tf
    from tensorflow.python.framework import ops
    with ops.name_scope(name, 'PerspectiveProjection', [f, c, w, h, near, far]) as scope:
        f = 0.5 * (f[0] + f[1])
        pixel_center_offset = 0.5
        right = (w - (c[0] + pixel_center_offset)) * (near / f)
        left = -(c[0] + pixel_center_offset) * (near / f)
        top = (c[1] + pixel_center_offset) * (near / f)
        bottom = -(h - c[1] + pixel_center_offset) * (near / f)

        elements = [
            [2. * near / (right - left), 0., (right + left) / (right - left), 0.],
            [0., 2. * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]

        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))

def project_points_perspective(m_v, camera_f, camera_c, camera_t, camera_rt, width, height, near = 0.1, far = 10):
    projection_matrix = perspective_projection(camera_f, camera_c, width, height, near, far)

    view_matrix = matrices.compose(
        matrices.rodrigues(camera_rt.astype(np.float32)),
        matrices.translation(camera_t.astype(np.float32)),
    )
    m_v = tf.cast(m_v, tf.float32)
    m_v = tf.concat([m_v, tf.ones_like(m_v[:, :, -1:])], axis=2)
    m_v = tf.matmul(m_v, tf.tile(view_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))
    m_v = tf.matmul(m_v, tf.tile(projection_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))

    return m_v
def render_colored(m_v, m_f, m_vc, width, height, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                   num_channels=3, camera_t=np.zeros(3, dtype=np.float32), camera_rt=np.zeros(3, dtype=np.float32),
                   name=None):
    with ops.name_scope(name, "render", [m_v]) as name:
        assert (num_channels == m_vc.shape[-1] == bgcolor.shape[0])

        projection_matrix = perspective_projection(camera_f, camera_c, width, height, .1, 10)
        # projection_matrix = matrices.perspective_projection(near=0.1, far=20., right=0.1, aspect=1.)

        view_matrix = matrices.compose(
                        matrices.rodrigues(camera_rt.astype(np.float32)),
                        matrices.translation(camera_t.astype(np.float32)),
                      )

        bg = tf.tile(bgcolor.astype(np.float32)[np.newaxis, np.newaxis, :], (height, width, 1))

        m_v = tf.cast(m_v, tf.float32)
        m_v = tf.concat([m_v, tf.ones_like(m_v[:, -1:])], axis=1)

        m_v = tf.matmul(m_v, view_matrix)
        m_v = tf.matmul(m_v, projection_matrix)

        return dirt.rasterise(bg, m_v, tf.cast(m_vc, tf.float32), tf.cast(m_f, tf.int32), name=name)

def render_colored_batch(m_v, m_f, m_vc, width, height, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None):
    with ops.name_scope(name, "render_batch", [m_v]) as name:
        # print(name)
        assert (num_channels == m_vc.shape[-1] == bgcolor.shape[0])

        # projection_matrix = perspective_projection(camera_f, camera_c, width, height, .1, 10)

        # view_matrix = matrices.compose(
        #     matrices.rodrigues(camera_rt.astype(np.float32)),
        #     matrices.translation(camera_t.astype(np.float32)),
        # )

        bg = tf.tile(bgcolor.astype(np.float32)[np.newaxis, np.newaxis, np.newaxis, :],
                     (tf.shape(m_v)[0], height, width, 1))

        if m_vc.ndim < m_v.ndim:
            m_vc = tf.tile(tf.cast(m_vc, tf.float32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        m_v = project_points_perspective(m_v, camera_f, camera_c, camera_t, camera_rt, width, height, near=0.1, far=10)
        # m_v = tf.cast(m_v, tf.float32)
        # m_v = tf.concat([m_v, tf.ones_like(m_v[:, :, -1:])], axis=2)
        # m_v = tf.matmul(m_v, tf.tile(view_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))
        # m_v = tf.matmul(m_v, tf.tile(projection_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))

        m_f = tf.tile(tf.cast(m_f, tf.int32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        return dirt.rasterise_batch(bg, m_v, m_vc, m_f, name=name)

def render_textured_batch(v_, f_, vt_, ft_, texture_image, width, height, camera_f, camera_c,
                          bgcolor=np.zeros(3, dtype=np.float32), camera_t=np.zeros(3, dtype=np.float32),
                          camera_rt=np.zeros(3, dtype=np.float32)):
    """
    v: (B x V x 3)
    f: (F x 3)
    vt: (VT x 3)
    ft: (FT x 3)
    texture_image: (H x W x 3)
    """
    # f = tf.cast(f, tf.int32)
    # ft = tf.cast(ft, tf.int32)
    v, f = split_vertices_by_face(v_, f_)
    vt, ft = split_vertices_by_face(tf.expand_dims(vt_, 0), ft_)

    uv = tf.concat([vt[0], tf.ones_like(vt[0, :, -1:])], axis=1)

    uv_image = render_colored_batch(v, f, uv, width, height, camera_f, camera_c, bgcolor,
                                    camera_t=camera_t, camera_rt=camera_rt)
    mask_image = render_colored_batch(v, f, tf.ones_like(uv), width, height, camera_f, camera_c,
                                      bgcolor=np.zeros(3, dtype=np.float32),
                                      camera_t=camera_t, camera_rt=camera_rt)

    idx_u = tf.round(uv_image[:, :, :, 0] * tf.cast(tf.shape(texture_image)[0] - 1, tf.float32))
    idx_v = tf.round((1 - uv_image[:, :, :, 1]) * tf.cast(tf.shape(texture_image)[1] - 1, tf.float32))

    image = tf.gather_nd(texture_image, tf.cast(tf.stack((idx_v, idx_u), axis=3), tf.int32))

    return image * mask_image + (1 - mask_image) * tf.reshape(bgcolor, (1, 1, 3))

def render_shaded_batch(v, f, vc, width, height, camera_f, camera_c, light_direction = np.zeros((3,)), light_color = np.ones((3,)),
                        bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None):
    """
    v: (B x V x 3)
    f: (F x 3)
    vt: (VT x 3)
    ft: (FT x 3)
    """
    org_f = f
    vc, _ = split_vertices_by_face(vc, org_f)
    v, f = split_vertices_by_face(v, org_f)
    vn = vertex_normals_pre_split(v, f, name=None, static=False)

    if light_direction.ndim < v.ndim:
        light_direction = tf.tile(tf.cast(light_direction, tf.float32)[np.newaxis, ...], (tf.shape(v)[0], 1))

    if light_color.ndim < v.ndim:
        light_color = tf.tile(tf.cast(light_color, tf.float32)[np.newaxis, ...], (tf.shape(v)[0], 1))

    if vc.ndim < v.ndim:
        vc = tf.tile(tf.cast(vc, tf.float32)[np.newaxis, ...], (tf.shape(v)[0], 1, 1))

    # shaded_vc = diffuse_directional(tf.cast(vn, tf.double), tf.cast(vc, tf.float64), tf.cast(light_direction, tf.double),
    #                                 tf.cast(light_color, tf.double), double_sided=True, name=None)
    shaded_vc = diffuse_point(v, vn, vc, light_direction, light_color)

    im = render_colored_batch(v, f, shaded_vc, width, height, camera_f, camera_c, bgcolor=bgcolor,
                         num_channels=num_channels, camera_t=camera_t,
                         camera_rt=camera_rt, name=name)
    return im

def render_normals_batch(v, f, width, height, camera_f, camera_c,
                        bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None):
    org_f = f
    v, f = split_vertices_by_face(v, org_f)
    vn = vertex_normals_pre_split(v, f, name=None, static=False)

    im = render_colored_batch(v, f, vn, width, height, camera_f, camera_c, bgcolor=bgcolor,
                              num_channels=num_channels, camera_t=camera_t,
                              camera_rt=camera_rt, name=name)

    return im

