import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import dxchange
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftn, ifftn
from pyfftw.interfaces.numpy_fft import fftshift as np_fftshift
from pyfftw.interfaces.numpy_fft import ifftshift as np_ifftshift

from util import get_kernel, get_kernel_ir

PI = 3.1415927


def multislice_propagate_batch_numpy(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm, free_prop_cm=None, obj_batch_shape=None):

    minibatch_size = obj_batch_shape[0]
    grid_shape = obj_batch_shape[1:]
    voxel_nm = np.array([psize_cm] * 3) * 1.e7
    wavefront = np.zeros([minibatch_size, obj_batch_shape[1], obj_batch_shape[2]], dtype='complex64')
    wavefront += (probe_real + 1j * probe_imag)

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm

    n_slice = obj_batch_shape[-1]
    delta_nm = voxel_nm[-1]

    # h = get_kernel_ir(delta_nm, lmbda_nm, voxel_nm, grid_shape)
    h = get_kernel(delta_nm, lmbda_nm, voxel_nm, grid_shape)
    k = 2. * PI * delta_nm / lmbda_nm

    probe_array = []

    for i in range(n_slice):
        delta_slice = grid_delta_batch[:, :, :, i]
        beta_slice = grid_beta_batch[:, :, :, i]
        c = np.exp(1j * k * delta_slice) * np.exp(-k * beta_slice)
        wavefront = wavefront * c
        if i < n_slice - 1:
            wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
        probe_array.append(wavefront)

    if free_prop_cm is not None:
        #1dxchange.write_tiff(abs(wavefront), '2d_1024/monitor_output/wv', dtype='float32', overwrite=True)
        if free_prop_cm == 'inf':
            wavefront = np_fftshift(fft2(wavefront), axes=[1, 2])
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            algorithm = 'TF' if mean_voxel_nm > crit_samp else 'IR'
            # print(algorithm)
            algorithm = 'TF'
            if algorithm == 'TF':
                h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
            else:
                h = get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
            # dxchange.write_tiff(abs(wavefront), '2d_512/monitor_output/wv', dtype='float32', overwrite=True)
            # dxchange.write_tiff(np.angle(h), '2d_512/monitor_output/h', dtype='float32', overwrite=True)

    return wavefront, np.array(probe_array)


if __name__ == '__main__':

    grid_delta = np.load('cone_256_foam/phantom/grid_delta.npy')
    grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    wavefield, probe_array = multislice_propagate_batch_numpy(grid_delta, grid_beta, probe_real, probe_imag, 5000, 1e-7, obj_batch_shape=grid_delta.shape)
    dxchange.write_tiff(np.abs(probe_array), 'test/np_array', overwrite=True, dtype='float32')

