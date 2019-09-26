import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
from autograd.numpy.fft import fft2, ifft2
from autograd.numpy.fft import fftshift as np_fftshift
from autograd.numpy.fft import ifftshift as np_ifftshift
from scipy.interpolate import RegularGridInterpolator
from util import get_kernel, get_kernel_ir
from constants import *
import dxchange
import time

from util import *

def multislice_propagate_batch_numpy(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm, free_prop_cm=None, obj_batch_shape=None, return_fft_time=True, starting_slice=0, debug=True, debug_save_path=None, rank=0, t_init=0):

    minibatch_size = obj_batch_shape[0]
    grid_shape = obj_batch_shape[1:]
    voxel_nm = np.array(psize_cm) * 1.e7
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

    t_tot = t_init
    for i in range(starting_slice, n_slice):
        if i % 5 == 0 and debug:
            np.savetxt(os.path.join(debug_save_path, 'current_islice_rank_{}.txt'.format(rank)), np.array([i, t_tot]))
            dxchange.write_tiff(wavefront.real, os.path.join(debug_save_path, 'probe_real_rank_{}.tiff'.format(rank)), dtype='float32', overwrite=True)
            dxchange.write_tiff(wavefront.imag, os.path.join(debug_save_path, 'probe_imag_rank_{}.tiff'.format(rank)), dtype='float32', overwrite=True)
        # Use np.array to convert memmap to memory object
        delta_slice = np.array(grid_delta_batch[:, :, :, i])
        beta_slice = np.array(grid_beta_batch[:, :, :, i])
        t0 = time.time()
        c = np.exp(1j * k * delta_slice) * np.exp(-k * beta_slice)
        wavefront = wavefront * c
        if i < n_slice - 1:
            wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
        t_tot += (time.time() - t0)

    if free_prop_cm not in [0, None]:
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
    if return_fft_time:
        return wavefront, t_tot
    else:
        return wavefront


if __name__ == '__main__':

    grid_delta = np.load('cone_256_foam/phantom/grid_delta.npy')
    grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    t0 = time.time()
    wavefield = multislice_propagate_batch_numpy(grid_delta, grid_beta, probe_real, probe_imag, 5000, [1e-7, 1e-7, 1e-7], free_prop_cm=1e-4, obj_batch_shape=grid_delta.shape)
    print('Delta t = {} s.'.format(time.time() - t0))
    plt.imshow(abs(wavefield[0]))
    plt.show()

