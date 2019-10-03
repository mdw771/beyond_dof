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
from math import ceil, floor

from util import *

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


def multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, energy_ev, psize_cm, kernel_size=17, free_prop_cm=None, original_grid_shape=None, return_fft_time=True, starting_slice=0, debug=True, debug_save_path=None, rank=0, t_init=0, verbose=False):

    assert kernel_size % 2 == 1, 'kernel_size must be an odd number.'
    n_batch, shape_y, shape_x, n_slice = grid_delta.shape
    lmbda_nm = 1240. / energy_ev
    voxel_nm = np.array(psize_cm) * 1.e7
    delta_nm = voxel_nm[-1]
    k = 2. * np.pi * delta_nm / lmbda_nm
    if original_grid_shape is not None:
        grid_shape = np.array(original_grid_shape)
    else:
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

    # probe_array = []

    edge_val = 1.0

    t_tot = t_init

    initial_int = probe[0, 0, 0]
    for i_slice in trange(starting_slice, n_slice, disable=(not verbose)):
        if i_slice % 5 == 0 and debug:
            np.savetxt(os.path.join(debug_save_path, 'current_islice_rank_{}.txt'.format(rank)), np.array([i_slice, t_tot]))
            dxchange.write_tiff(probe.real, os.path.join(debug_save_path, 'probe_real_rank_{}.tiff'.format(rank)), dtype='float32', overwrite=True)
            dxchange.write_tiff(probe.imag, os.path.join(debug_save_path, 'probe_imag_rank_{}.tiff'.format(rank)), dtype='float32', overwrite=True)
        # Use np.array to convert memmap to memory object
        delta_slice = np.array(grid_delta[:, :, :, i_slice])
        beta_slice = np.array(grid_beta[:, :, :, i_slice])
        t0 = time.time()
        c = np.exp(1j * k * delta_slice - k * beta_slice)
        probe = probe * c
        probe = np.pad(probe, [[0, 0], [pad_len, pad_len], [pad_len, pad_len]], mode='constant', constant_values=edge_val)
        probe = convolve(probe, kernel, mode='valid', axes=([1, 2], [0, 1]))
        # Phase shift to incident wave induced by truncated kernel
        edge_val = sum(kernel.flatten() * edge_val)
        t_tot += (time.time() - t0)
        # probe_array.append(np.abs(probe))

    #  Correct intensity offset
    final_int = probe[0, 0, 0]
    probe *= (initial_int / final_int)

    if free_prop_cm not in [None, 0]:
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
                h = get_kernel(dist_nm, lmbda_nm, voxel_nm, np.array(grid_delta.shape[1:]))
                probe = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(probe), axes=[1, 2]) * h, axes=[1, 2]))
            else:
                h = get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, np.array(grid_delta.shape[1:]))
                probe = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(probe), axes=[1, 2]) * h, axes=[1, 2]))

    if return_fft_time:
        return probe, t_tot
    else:
        return probe



if __name__ == '__main__':

    # f = open('test/safe_width_error.csv', 'w')
    # f.write('safe_zone_width,error\n')
    # ref = dxchange.read_tiff('test/det_std.tiff')

    # for safe_zone_width in range(30, 130, 10):


    energy_ev = 5000
    psize_cm = 1e-7
    kernel_size = 17
    free_prop_cm = 1e-4

    # grid_delta = np.load('adhesin/phantom/grid_delta.npy')
    # grid_beta = np.load('adhesin/phantom/grid_beta.npy')
    grid_delta = np.load('cone_256_foam/phantom/grid_delta.npy')
    grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    # grid_delta = dxchange.read_tiff('cone_256_foam/test0/intermediate/current.tiff')
    # grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
    n_batch = grid_delta.shape[0]
    original_grid_shape = grid_delta.shape[1:]

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    # f = open('test/conv_ir_report.csv', 'a')
    # f.write('kernel_size,time\n')

    lmbda_nm = 1.24 / (energy_ev / 1e3)
    safe_zone_width = ceil(4.0 * np.sqrt((psize_cm * 1e7 * grid_delta.shape[-1] + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7)) + (kernel_size // 2) + 1
    # safe_zone_width = 64
    print(safe_zone_width)

    # Calculate the block range to be processed by each rank.
    # If the number of ranks is smaller than the number of lines, each rank will take 1 or more
    # whole lines.
    n_lines = grid_delta.shape[1]
    n_pixels_per_line = grid_delta.shape[2]
    if size <= n_lines:
        n_ranks_spill = n_lines % size
        line_st = n_lines // size * rank + min([n_ranks_spill, rank])
        line_end = line_st + n_lines // size
        if rank < n_ranks_spill:
            line_end += 1
        px_st = 0
        px_end = n_pixels_per_line
        safe_zone_width_side = 0
    else:
        n_lines_spill = size % n_lines
        n_ranks_per_line_base = size // n_lines
        n_ranks_spill = n_lines_spill * (n_ranks_per_line_base + 1)
        line_st = rank // (n_ranks_per_line_base + 1) if rank < n_ranks_spill else (rank - n_ranks_spill) // n_ranks_per_line_base + n_lines_spill
        line_end = line_st + 1
        n_ranks_per_line = n_ranks_per_line_base if line_st < n_lines_spill else n_ranks_per_line_base + 1
        i_seg = rank % n_ranks_per_line
        px_st = int(n_pixels_per_line * (float(i_seg) / n_ranks_per_line))
        px_end = min([int(n_pixels_per_line * (float(i_seg + 1) / n_ranks_per_line)) + 1, n_pixels_per_line])
        safe_zone_width_side = safe_zone_width

    # sub_grids are memmaps
    sub_grid_delta = grid_delta[:, max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, n_lines), max([0, px_st - safe_zone_width_side]):min([px_end + safe_zone_width_side, n_pixels_per_line]), :]
    sub_grid_beta = grid_beta[:, max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, n_lines), max([0, px_st - safe_zone_width_side]):min([px_end + safe_zone_width_side, n_pixels_per_line]), :]

    # During padding, sub_grids are read into the RAM
    pad_top, pad_bottom = (0, 0)
    if line_st < safe_zone_width:
        sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [safe_zone_width - line_st, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
        sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [safe_zone_width - line_st, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
        probe_real = np.pad(probe_real, [[safe_zone_width - line_st, 0], [0, 0]], mode='edge')
        probe_imag = np.pad(probe_imag, [[safe_zone_width - line_st, 0], [0, 0]], mode='edge')
        pad_top = safe_zone_width - line_st
    if (n_lines - line_end + 1) < safe_zone_width:
        sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [0, line_end + safe_zone_width - n_lines], [0, 0], [0, 0]], mode='constant', constant_values=0)
        sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [0, line_end + safe_zone_width - n_lines], [0, 0], [0, 0]], mode='constant', constant_values=0)
        probe_real = np.pad(probe_real, [[0, line_end + safe_zone_width - n_lines], [0, 0]], mode='edge')
        probe_imag = np.pad(probe_imag, [[0, line_end + safe_zone_width - n_lines], [0, 0]], mode='edge')
        pad_bottom = safe_zone_width
    if safe_zone_width_side > 0:
        sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [0, 0], [safe_zone_width_side, safe_zone_width_side], [0, 0]], mode='constant', constant_values=0)
        sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [0, 0], [safe_zone_width_side, safe_zone_width_side], [0, 0]], mode='constant', constant_values=0)
        probe_real = np.pad(probe_real, [[0, 0], [safe_zone_width_side, safe_zone_width_side]], mode='edge')
        probe_imag = np.pad(probe_imag, [[0, 0], [safe_zone_width_side, safe_zone_width_side]], mode='edge')

    print(sub_grid_beta.shape)
    print(line_st, line_end)

    t0 = time.time()
    wavefield = multislice_propagate_cnn(sub_grid_delta, sub_grid_beta,
                                         probe_real[pad_top + line_st - safe_zone_width:pad_top + line_end + safe_zone_width, px_st:px_end + 2 * safe_zone_width_side],
                                         probe_imag[pad_top + line_st - safe_zone_width:pad_top + line_end + safe_zone_width, px_st:px_end + 2 * safe_zone_width_side],
                                         energy_ev, [psize_cm] * 3, kernel_size=kernel_size, free_prop_cm=free_prop_cm, debug=False,
                                         original_grid_shape=original_grid_shape)

    this_full_wavefield = np.zeros([n_batch, *original_grid_shape[:-1]], dtype='complex64')
    this_full_wavefield[:, line_st:line_end, px_st:px_end] = wavefield[:, safe_zone_width:safe_zone_width + (line_end - line_st),
                                                                       safe_zone_width_side:safe_zone_width_side + (px_end - px_st)]
    full_wavefield = np.zeros_like(this_full_wavefield, dtype='complex64')
    comm.Allreduce(this_full_wavefield, full_wavefield)
    t = time.time() - t0

    if rank == 0:
        # dxchange.write_tiff(np.array(probe_array), 'test/array_conv', dtype='float32', overwrite=True)
        dxchange.write_tiff(np.abs(full_wavefield), 'test/det', dtype='float32', overwrite=True)


    # for kernel_size in np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 2 + 1:
    #     wavefield, probe_array, t = multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, 5000, [1e-7, 1e-7, 1e-7], kernel_size=kernel_size)
    #     # print(wavefield.shape)
    #     # plt.imshow(abs(wavefield))
    #     # plt.show()
    #     # dxchange.write_tiff(np.abs(wavefield), 'test/test', dtype='float32', overwrite=True)
    #     dxchange.write_tiff(np.array(probe_array), 'test/array_conv_{}_itf_bound_wrap'.format(kernel_size), dtype='float32', overwrite=True)
    #     f.write('{},{}\n'.format(kernel_size, t))
    # f.close()
    print('Delta t = {} s.'.format(t))

        # error = np.mean((abs(full_wavefield) - ref) ** 2)
        # f.write('{},{}\n'.format(safe_zone_width, error))
    #
    # f.close()
