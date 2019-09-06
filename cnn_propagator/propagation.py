import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import dxchange
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy
from autograd.scipy.signal import convolve
from autograd import grad
import time
import warnings
import math
import numpy as nnp
from tqdm import trange

from util import *


warnings.catch_warnings()
warnings.simplefilter('ignore')

try:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    mpi_ok = True
except:
    from pseudo import Mpi
    comm = Mpi()
    size = 1
    rank = 0
    mpi_ok = False

def multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, energy_ev, psize_cm, kernel_size=17, free_prop_cm=None, n_line_per_rank=2, debug=False):

    assert kernel_size % 2 == 1, 'kernel_size must be an odd number.'
    n_batch, shape_y, shape_x, n_slice = grid_delta.shape
    lmbda_nm = 1240. / energy_ev
    voxel_nm = np.array(psize_cm) * 1.e7
    delta_nm = voxel_nm[-1]
    k = 2. * np.pi * delta_nm / lmbda_nm
    grid_shape = np.array(grid_delta.shape[1:])
    size_nm = voxel_nm * grid_shape
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)

    # print('Critical distance is {} cm.'.format(psize_cm[0] * psize_cm[1] * grid_delta.shape[1] / (lmbda_nm * 1e-7)))

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
    probe_size = probe.shape
    probe = np.tile(probe, [n_batch, 1, 1])


    probe_array = []

    # Build cyclic convolution matrix for kernel
    # kernel_mat = np.zeros([np.prod(probe_size)] * 2)
    # kernel_full_00 = np.zeros(probe_size)
    # kernel_full_00[:kernel_size, :kernel_size] = kernel
    # kernel_full_00 = np.roll(kernel_full_00, -half_kernel_size, axis=0)
    # kernel_full_00 = np.roll(kernel_full_00, -half_kernel_size, axis=1)
    # kernel_mat[0, :] = kernel_full_00.flatten()
    # for i in trange(probe_size[0]):
    #     for j in range(probe_size[1]):
    #         if i != 0 or j != 0:
    #             kernel_temp = np.roll(kernel_full_00, i, axis=0)
    #             kernel_temp = np.roll(kernel_temp, j, axis=1)
    #             kernel_mat[i * probe_size[1] + j, :] = kernel_temp.flatten()


    t0 = time.time()

    edge_val = 1.0
    kernel_sum = np.sum(kernel)

    initial_int = probe[0, 0, 0]
    t00 = time.time()
    t_comm = 0.
    for i_slice in trange(n_slice):
        this_delta_batch = grid_delta[:, :, :, i_slice]
        this_beta_batch = grid_beta[:, :, :, i_slice]

        # Probe modulation
        c = np.exp(1j * k * this_delta_batch - k * this_beta_batch)
        probe = probe * c

        # Distributed probe propagation
        # temp_probe_full = np.zeros_like(probe)
        temp_probe_holder = np.zeros_like(probe)
        temp_probe_ls = []
        line_spacing = n_line_per_rank * (size - 1)
        line_ls = range(rank * n_line_per_rank, probe_size[0], n_line_per_rank * size)
        n_lines = len(line_ls)

        temp_probe_ls.append(np.zeros([n_batch, rank * n_line_per_rank, probe_size[1]]))

        for i, l in enumerate(line_ls):
            temp_probe = probe[:, max([0, l - pad_len]):min([l + n_line_per_rank + pad_len, probe_size[0]]), :]
            # Pad bottom if needed
            if l + n_line_per_rank + pad_len > probe_size[0]:
                temp_probe = np.pad(temp_probe, [[0, 0], [0, l + n_line_per_rank + pad_len - probe_size[0]], [0, 0]], mode='constant', constant_values=edge_val)
            # Pad top if needed
            if l - pad_len < 0:
                temp_probe = np.pad(temp_probe, [[0, 0], [pad_len - l, 0], [0, 0]], mode='constant', constant_values=edge_val)
            # Pad left and right
            temp_probe = np.pad(temp_probe, [[0, 0], [0, 0], [pad_len, pad_len]], mode='constant', constant_values=edge_val)
            temp_probe = convolve(temp_probe, kernel, mode='valid', axes=([1, 2], [0, 1]))
            # print(temp_probe)
            # temp_probe_full[:, l:l + n_line_per_rank, :] = temp_probe
            # Expand and append to line list
            # if i == 0:
            #     temp_probe = np.pad(temp_probe, [[0, 0], [rank * n_line_per_rank, line_spacing], [0, 0]],
            #                         mode='constant', constant_values=0)
            # elif i == n_lines - 1:
            #     temp_probe = np.pad(temp_probe, [[0, 0], [0, probe_size[0] - (l + n_line_per_rank)], [0, 0]],
            #                         mode='constant', constant_values=0)
            # else:
            #     temp_probe = np.pad(temp_probe, [[0, 0], [0, line_spacing], [0, 0]],
            #                         mode='constant', constant_values=0)
            temp_probe_ls.append(temp_probe)
            if i < n_lines - 1:
                temp_probe_ls.append(np.zeros([n_batch, line_spacing, probe_size[1]]))
            else:
                temp_probe_ls.append(np.zeros([n_batch, probe_size[0] - (l + n_line_per_rank), probe_size[1]]))
        temp_probe_full = np.concatenate(temp_probe_ls, axis=1)
        comm.Barrier()
        t_comm_0 = time.time()
        dxchange.write_tiff(abs(temp_probe_full), 'test/temp_probe_full/{}_{}'.format(i_slice, rank), dtype='float32', overwrite=True)
        comm.Allreduce(temp_probe_full, temp_probe_holder)
        dxchange.write_tiff(abs(temp_probe_holder), 'test/temp_probe_holder/{}_{}'.format(i_slice, rank), dtype='float32', overwrite=True)

        t_comm += time.time() - t_comm_0
        probe = np.copy(temp_probe_holder)

        # probe = np.reshape(probe, [n_batch, np.prod(probe_size)])
        # probe = probe.dot(kernel_mat.T)
        # probe = np.reshape(probe, [n_batch, *probe_size])

        edge_val *= kernel_sum
        probe_array.append(np.abs(probe))

    if debug:
        print_flush('Multislice time: {} s.'.format(time.time() - t00))
        print_flush('Allreduce overhead: {} s.'.format(t_comm))

    final_int = probe[0, 0, 0]
    probe *= (initial_int / final_int)

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

    # grid_delta = np.load('adhesin/phantom/grid_delta.npy')
    # grid_beta = np.load('adhesin/phantom/grid_beta.npy')
    # grid_delta = np.load('cone_256_foam/phantom/grid_delta.npy')
    # grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = dxchange.read_tiff('cone_256_foam/test0/intermediate/current.tiff')
    grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    # f = open('test/conv_ir_report.csv', 'a')
    # f.write('kernel_size,time\n')

    wavefield, probe_array, t = multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, 5000,
                                                         [1e-7] * 3, kernel_size=17, free_prop_cm=1e-4, debug=True, n_line_per_rank=2)

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


