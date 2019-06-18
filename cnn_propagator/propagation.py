import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import dxchange
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.signal
from autograd import grad
convolve = autograd.scipy.signal.convolve
import scipy.signal
import time

from util import *


def multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, energy_ev, psize_cm, kernel_size=17, free_prop_cm=None, debug=False):

    n_batch, shape_y, shape_x, n_slice = grid_delta.shape
    lmbda_nm = 1240. / energy_ev
    voxel_nm = np.array(psize_cm) * 1.e7
    delta_nm = voxel_nm[-1]
    k = 2. * np.pi * delta_nm / lmbda_nm
    grid_shape = np.array(grid_delta.shape[1:])
    size_nm = voxel_nm * grid_shape
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)

    print('Critical distance is {} cm.'.format(psize_cm[0] * psize_cm[1] * grid_delta.shape[1] / (lmbda_nm * 1e-7)))

    if kernel_size % 2 == 0:
        warnings.warn('Kernel size should be odd.')
    # kernel = get_kernel(delta_nm, lmbda_nm, voxel_nm, np.array(grid_delta.shape[1:]))
    kernel = get_kernel(delta_nm, lmbda_nm, voxel_nm, grid_shape - 1)
    kernel = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kernel)))
    # dxchange.write_tiff(np.abs(kernel), 'test/kernel_abs', dtype='float32')
    # dxchange.write_tiff(np.angle(kernel), 'test/kernel_phase', dtype='float32')
    # raise Exception

    kernel_mid = ((np.array(kernel.shape) - 1) / 2).astype('int')
    half_kernel_size = int((kernel_size - 1) / 2)
    kernel = kernel[kernel_mid[0] - half_kernel_size:kernel_mid[0] + half_kernel_size + 1,
                    kernel_mid[1] - half_kernel_size:kernel_mid[1] + half_kernel_size + 1]
    # kernel = get_kernel_ir_real(delta_nm, lmbda_nm, voxel_nm, [kernel_size, kernel_size, 256])
    # kernel /= kernel.size
    pad_len = (kernel_size - 1) // 2

    probe = probe_real + 1j * probe_imag
    probe = np.tile(probe, [n_batch, 1, 1])

    probe_array = []


    t0 = time.time()

    for i_slice in range(n_slice):
        print(i_slice)
        this_delta_batch = grid_delta[:, :, :, i_slice]
        this_beta_batch = grid_beta[:, :, :, i_slice]
        # this_delta_batch = np.squeeze(this_delta_batch)
        # this_beta_batch = np.squeeze(this_beta_batch)
        c = np.exp(1j * k * this_delta_batch - k * this_beta_batch)
        probe = probe * c
        # print(probe.shape, kernel.shape)
        # probe = scipy.signal.convolve2d(probe, kernel, mode='valid', boundary='wrap')
        probe = convolve(probe, kernel, mode='valid', axes=([1, 2], [0, 1]))
        # print(probe.shape)
        # probe = ifft2(np_ifftshift(np_fftshift(fft2(probe)) * np_fftshift(fft2(kernel))))
        # probe = ifft2(np_ifftshift(np_fftshift(fft2(probe)) * kernel))
        probe = np.pad(probe, [[0, 0], [pad_len, pad_len], [pad_len, pad_len]], mode='constant', constant_values=1)
        probe_array.append(np.abs(probe))

    if free_prop_cm is not None:
        #1dxchange.write_tiff(abs(wavefront), '2d_1024/monitor_output/wv', dtype='float32', overwrite=True)
        if free_prop_cm == 'inf':
            probe = np.fft.fftshift(np.fft.fft2(probe), axes=[1, 2])
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            algorithm = 'TF' if mean_voxel_nm > crit_samp else 'IR'
            # print(algorithm)
            algorithm = 'TF'
            if algorithm == 'TF':
                h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                probe = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(probe), axes=[1, 2]) * h, axes=[1, 2]))
            else:
                h = get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                probe = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(probe), axes=[1, 2]) * h, axes=[1, 2]))

    if debug:
        return probe, probe_array, time.time() - t0
    else:
        return probe



if __name__ == '__main__':

    import pandas

    grid_delta = np.load('cone_256_foam/phantom/grid_delta.npy')
    grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    # f = open('test/conv_ir_report.csv', 'a')
    # f.write('kernel_size,time\n')

    wavefield, probe_array, t = multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, 5000,
                                                         [1e-7, 1e-7, 1e-7], kernel_size=17, free_prop_cm=1e-4, debug=True)

    dxchange.write_tiff(np.array(probe_array), 'test/array_conv', dtype='float32', overwrite=True)
    dxchange.write_tiff(np.abs(wavefield), 'test/det', dtype='float32', overwrite=True)


    # for kernel_size in np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 2 + 1:
    #     wavefield, probe_array, t = multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, 5000, [1e-7, 1e-7, 1e-7], kernel_size=kernel_size)
    #     # print(wavefield.shape)
    #     # plt.imshow(abs(wavefield))
    #     # plt.show()
    #     # dxchange.write_tiff(np.abs(wavefield), 'test/test', dtype='float32', overwrite=True)
    #     dxchange.write_tiff(np.array(probe_array), 'test/array_conv_{}_itf_bound_wrap'.format(kernel_size), dtype='float32', overwrite=True)
    #     f.write('{},{}\n'.format(kernel_size, t))
    # f.close()
    # print('Delta t = {} s.'.format(t))


