import tensorflow as tf
import dxchange
import h5py
import matplotlib.pyplot as plt
import matplotlib
import autograd.numpy as np

import warnings
try:
    import sys
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import fourier_shift
except:
    warnings.warn('Some dependencies are screwed up.')
import os
import pickle
import glob
from scipy.special import erf

PI = 3.1415927


def fftshift(tensor):
    ndim = len(tensor.shape)
    dim_ls = range(ndim - 2, ndim)
    for i in dim_ls:
        n = tensor.shape[i].value
        p2 = (n+1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def ifftshift(tensor):
    ndim = len(tensor.shape)
    dim_ls = range(ndim - 2, ndim)
    for i in dim_ls:
        n = tensor.shape[i].value
        p2 = n - (n + 1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def total_variation_3d(arr):
    """
    Calculate total variation of a 3D array.
    :param arr: 3D Tensor.
    :return: Scalar.
    """
    res = np.sum(np.abs(np.roll(arr, 1, axis=0) - arr))
    res = res + np.sum(np.abs(np.roll(arr, 1, axis=1) - arr))
    res = res + np.sum(np.abs(np.roll(arr, 1, axis=2) - arr))
    return res


def gen_mesh(max, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape):
    """Get Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    k = 2 * PI / lmbda_nm
    u_max = 1. / (2. * voxel_nm[0])
    v_max = 1. / (2. * voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], grid_shape[0:2])
    # H = np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm**2 * (u**2 + v**2)))
    try:
        H = np.exp(1j * k * dist_nm) * np.exp(-1j * PI * lmbda_nm * dist_nm * (u**2 + v**2))
    except:
        H = tf.exp(1j * k * dist_nm) * tf.exp(-1j * PI * lmbda_nm * dist_nm * (u ** 2 + v ** 2))

    return H


def get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape):

    """
    Get Fresnel propagation kernel for IR algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    size_nm = np.array(voxel_nm) * np.array(grid_shape)
    k = 2 * PI / lmbda_nm
    ymin, xmin = np.array(size_nm)[:2] / -2.
    dy, dx = voxel_nm[0:2]
    x = np.arange(xmin, xmin + size_nm[1], dx)
    y = np.arange(ymin, ymin + size_nm[0], dy)
    x, y = np.meshgrid(x, y)
    h = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))
    H = np.fft.fftshift(np.fft.fft2(h)) * voxel_nm[0] * voxel_nm[1]

    return H


def get_kernel_ir_real(dist_nm, lmbda_nm, voxel_nm, grid_shape):

    """
    Get Fresnel propagation kernel for IR algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    size_nm = np.array(voxel_nm) * np.array(grid_shape)
    k = 2 * PI / lmbda_nm
    y_half, x_half = (np.array(size_nm)[:2] - 1) / 2.
    dy, dx = voxel_nm[0:2]
    x = np.arange(0, size_nm[1], dx) - x_half
    y = np.arange(0, size_nm[0], dy) - y_half
    x, y = np.meshgrid(x, y)
    try:
        h = np.exp(1j / (dist_nm * lmbda_nm) * (x ** 2 + y ** 2))
    except:
        h = tf.exp(1j / (dist_nm * lmbda_nm) * (x ** 2 + y ** 2))
        # h = tf.convert_to_tensor(h, dtype='complex64')

    return h


def get_kernel_spherical(dist_nm, lmbda_nm, r_nm, theta_max, phi_max, probe_shape):

    k_theta = PI / theta_max * (np.arange(probe_shape[0]) - float(probe_shape[0] - 1) / 2)
    k_phi = PI / phi_max * (np.arange(probe_shape[1]) - float(probe_shape[1] - 1) / 2)
    k_theta = k_theta.astype(np.complex64)
    k_phi = k_phi.astype(np.complex64)
    k_phi, k_theta = tf.meshgrid(k_phi, k_theta)
    k = 2 * PI / lmbda_nm
    H = tf.exp(-1j / (2 * k) * (k_theta ** 2 + k_phi ** 2) * tf.cast(1. / (r_nm + dist_nm) - 1. / r_nm, tf.complex64))
    return H


def rescale_image(arr, m, original_shape):
    """
    :param arr: 3D image array [NHW]
    :param m:
    :param original_shape:
    :return:
    """
    n_batch = arr.shape[0]
    arr_shape = tf.cast(arr.shape[-2:], tf.float32)
    y_newlen = arr_shape[0] / m
    x_newlen = arr_shape[1] / m
    # tf.linspace shouldn't be used since it does not support gradient
    y = tf.range(0, arr_shape[0], 1, dtype=tf.float32)
    y = y / m + (original_shape[1] - y_newlen) / 2.
    x = tf.range(0, arr_shape[1], 1, dtype=tf.float32)
    x = x / m + (original_shape[2] - x_newlen) / 2.
    y = tf.clip_by_value(y, 0, arr_shape[0])
    x = tf.clip_by_value(x, 0, arr_shape[1])
    x_resample, y_resample = tf.meshgrid(x, y, indexing='ij')
    warp = tf.transpose(tf.stack([x_resample, y_resample]))
    # warp = tf.transpose(tf.stack([tf.reshape(y_resample, (np.prod(original_shape), )), tf.reshape(x_resample, (np.prod(original_shape), ))]))
    # warp = tf.cast(warp, tf.int32)
    # arr = arr * tf.reshape(warp[:, 0], original_shape)
    # arr = tf.gather_nd(arr, warp)
    warp = tf.expand_dims(warp, 0)
    warp = tf.tile(warp, [n_batch, 1, 1, 1])
    arr = tf.contrib.resampler.resampler(tf.expand_dims(arr, -1), warp)
    arr = tf.reshape(arr, original_shape)

    return arr


def preprocess(dat, blur=None, normalize_bg=False):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    # if normalize_bg:
    #     dat = tomopy.normalize_bg(dat)
    dat = -np.log(dat)
    dat[np.where(np.isnan(dat) == True)] = 0
    if blur is not None:
        dat = gaussian_filter(dat, blur)

    return dat


def realign_image(arr, shift):
    """
    Translate and rotate image via Fourier

    Parameters
    ----------
    arr : ndarray
        Image array.

    shift: tuple
        Mininum and maximum values to rescale data.

    angle: float, optional
        Mininum and maximum values to rescale data.

    Returns
    -------
    ndarray
        Output array.
    """
    # if both shifts are integers, do circular shift; otherwise perform Fourier shift.
    if np.count_nonzero(np.abs(np.array(shift) - np.round(shift)) < 0.01) == 2:
        temp = np.roll(arr, int(shift[0]), axis=0)
        temp = np.roll(temp, int(shift[1]), axis=1)
        temp = temp.astype('float32')
    else:
        temp = fourier_shift(np.fft.fftn(arr), shift)
        temp = np.fft.ifftn(temp)
        temp = np.abs(temp).astype('float32')
    return temp


def print_flush(a, designate_rank=None, this_rank=None):

    if designate_rank is not None:
        if this_rank == designate_rank:
            print(a)
    else:
        print(a)
    sys.stdout.flush()
    return


def real_imag_to_mag_phase(realpart, imagpart):

    a = realpart + 1j * imagpart
    return np.abs(a), np.angle(a)


def mag_phase_to_real_imag(mag, phase):

    a = mag * np.exp(1j * phase)
    return a.real, a.imag


def split_tasks(arr, split_size):
    res = []
    ind = 0
    while ind < len(arr):
        res.append(arr[ind:min(ind + split_size, len(arr))])
        ind += split_size
    return res


def apply_gradient_adam(x, g, i_batch, m=None, v=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):

    g = np.array(g)
    if m is None or v is None:
        m = np.zeros_like(x)
        v = np.zeros_like(v)
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i_batch + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i_batch + 1))
    x = x - step_size * mhat / (np.sqrt(vhat) + eps)
    return x, m, v


def save_rotation_lookup(array_size, n_theta, dest_folder=None):

    image_center = [np.floor(x / 2) for x in array_size]

    coord0 = np.arange(array_size[0])
    coord1 = np.arange(array_size[1])
    coord2 = np.arange(array_size[2])

    coord2_vec = np.tile(coord2, array_size[1])

    coord1_vec = np.tile(coord1, array_size[2])
    coord1_vec = np.reshape(coord1_vec, [array_size[1], array_size[2]])
    coord1_vec = np.reshape(np.transpose(coord1_vec), [-1])

    coord0_vec = np.tile(coord0, [array_size[1] * array_size[2]])
    coord0_vec = np.reshape(coord0_vec, [array_size[1] * array_size[2], array_size[0]])
    coord0_vec = np.reshape(np.transpose(coord0_vec), [-1])

    # move origin to image center
    coord1_vec = coord1_vec - image_center[1]
    coord2_vec = coord2_vec - image_center[2]

    # create matrix of coordinates
    coord_new = np.stack([coord1_vec, coord2_vec]).astype(np.float32)

    # create rotation matrix
    theta_ls = np.linspace(0, 2 * np.pi, n_theta)
    coord_old_ls = []
    for theta in theta_ls:
        m_rot = np.array([[np.cos(theta),  -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        coord_old = np.matmul(m_rot, coord_new)
        coord1_old = np.round(coord_old[0, :] + image_center[1]).astype(np.int)
        coord2_old = np.round(coord_old[1, :] + image_center[2]).astype(np.int)
        # clip coordinates
        coord1_old = np.clip(coord1_old, 0, array_size[1]-1)
        coord2_old = np.clip(coord2_old, 0, array_size[2]-1)
        coord_old = np.stack([coord1_old, coord2_old], axis=1)
        coord_old_ls.append(coord_old)
    if dest_folder is None:
        dest_folder = 'arrsize_{}_{}_{}_ntheta_{}'.format(array_size[0], array_size[1], array_size[2], n_theta)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    for i, arr in enumerate(coord_old_ls):
        np.save(os.path.join(dest_folder, '{:04}'.format(i)), arr)

    coord1_vec = coord1_vec + image_center[1]
    coord1_vec = np.tile(coord1_vec, array_size[0])
    coord2_vec = coord2_vec + image_center[2]
    coord2_vec = np.tile(coord2_vec, array_size[0])
    for i, coord in enumerate([coord0_vec, coord1_vec, coord2_vec]):
        np.save(os.path.join(dest_folder, 'coord{}_vec'.format(i)), coord)

    return coord_old_ls


def upsample_2x(arr):

    if arr.ndim == 4:
        out_arr = np.zeros([arr.shape[0] * 2, arr.shape[1] * 2, arr.shape[2] * 2, arr.shape[3]])
        for i in range(arr.shape[3]):
            out_arr[:, :, :, i] = upsample_2x(arr[:, :, :, i])
    else:
        out_arr = np.zeros([arr.shape[0] * 2, arr.shape[1] * 2, arr.shape[2] * 2])
        out_arr[::2, ::2, ::2] = arr[:, :, :]
        out_arr = gaussian_filter(out_arr, 1)
    return out_arr


def read_origin_coords(src_folder, index):

    coords = np.load(os.path.join(src_folder, '{:04}.npy'.format(index)))
    return coords


def read_all_origin_coords(src_folder, n_theta):

    coord_ls = []
    for i in range(n_theta):
        coord_ls.append(read_origin_coords(src_folder, i))
    return coord_ls


def apply_rotation(obj, coord_old, src_folder):

    coord_vec_ls = []
    for i in range(3):
        f = os.path.join(src_folder, 'coord{}_vec.npy'.format(i))
        coord_vec_ls.append(np.load(f))
    s = obj.shape
    coord0_vec, coord1_vec, coord2_vec = coord_vec_ls

    coord_old = np.tile(coord_old, [s[0], 1])
    coord1_old = coord_old[:, 0]
    coord2_old = coord_old[:, 1]
    coord_old = np.stack([coord0_vec, coord1_old, coord2_old], axis=1).transpose()
    # print(sess.run(coord_old))


    obj_channel_ls = np.split(obj, s[3], 3)
    obj_rot_channel_ls = []
    for channel in obj_channel_ls:
        channel_flat = channel.flatten()
        ind = coord_old[0] * (s[1] * s[2]) + coord_old[1] * s[2] + coord_old[2]
        ind = ind.astype('int')
        obj_chan_new_val = channel_flat[ind]
        obj_rot_channel_ls.append(np.reshape(obj_chan_new_val, s[:-1]))
    obj_rot = np.stack(obj_rot_channel_ls, axis=3)
    return obj_rot


def create_probe_initial_guess(data_fname, dist_nm, energy_ev, psize_nm):

    f = h5py.File(data_fname, 'r')
    dat = f['exchange/data'][...]
    # NOTE: this is for toy model
    wavefront = np.mean(np.abs(dat), axis=0)
    lmbda_nm = 1.24 / energy_ev
    h = get_kernel(-dist_nm, lmbda_nm, [psize_nm, psize_nm], wavefront.shape)
    wavefront = np.fft.fftshift(np.fft.fft2(wavefront)) * h
    wavefront = np.fft.ifft2(np.fft.ifftshift(wavefront))
    return wavefront
