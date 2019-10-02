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
import sys
import os
from math import ceil, floor
from propagation import multislice_propagate_cnn
from propagation_fft import multislice_propagate_batch_numpy


try:
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    mpi_ok = True
except:
    from pseudo import Mpi
    comm = Mpi()
    n_ranks = 1
    rank = 0
    mpi_ok = False

energy_ev = 3000
lmbda_nm = 1.24 / (energy_ev / 1e3)
psize_min_cm = 100e-7
kernel_size_ls = 2 ** np.array([2, 3, 4, 5, 6]) + 1
free_prop_cm = 0
######################################################################
size_ls = np.array([256, 512, 1024, 2048, 4096])
#####################################################################
path_prefix = os.path.join(os.getcwd(), 'charcoal')
n_repeats = 100

# Create report
if rank == 0:
    f = open(os.path.join(path_prefix, 'report.csv'), 'w')
    f.write('algorithm,object_size,kernel_size,safezone_width,avg_time,mse_with_fft\n')

# Benchmark partial FFT propagation
for this_size in size_ls:
    
    size_factor = size_ls[-1] // this_size
    psize_cm = psize_min_cm * size_factor

    verbose = False if rank != 0 else True

    grid_delta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_delta.tiff').format(this_size))
    grid_beta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_beta.tiff').format(this_size))
    grid_delta = np.swapaxes(np.swapaxes(grid_delta, 0, 1), 1, 2)
    grid_beta = np.swapaxes(np.swapaxes(grid_beta, 0, 1), 1, 2)
    dxchange.write_tiff(grid_delta, 'charcoal/size_256/test.tiff', dtype='float32', overwrite=True)

    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
    n_batch = grid_delta.shape[0]
    original_grid_shape = grid_delta.shape[1:]

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    ref = dxchange.read_tiff(
        os.path.join(path_prefix, 'size_{}'.format(this_size), 'fft_output.tiff'))

    safe_zone_width = ceil(
        4.0 * np.sqrt((psize_cm * 1e7 * grid_delta.shape[-1] + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7)) + 1

    # Must satisfy:
    # 1. n_block_x * n_block_y = n_ranks
    # 2. block_size * n_block_y = wave_shape[0]
    # 3. block_size * n_block_x = wave_shape[1]
    n_blocks_x = int(np.sqrt(float(n_ranks) * original_grid_shape[1] / original_grid_shape[0])) + 1
    n_blocks_y = int(np.sqrt(float(n_ranks) * original_grid_shape[0] / original_grid_shape[1])) + 1
    n_blocks = n_blocks_x * n_blocks_y
    block_size = int(float(original_grid_shape[0]) / n_blocks_y) + 1
    print(block_size, n_blocks_y, n_blocks_x, safe_zone_width)

    this_pos_ind_ls = range(rank, n_blocks, n_ranks)
    block_delta_batch = np.zeros([len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width, original_grid_shape[-1]])
    block_beta_batch = np.zeros([len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width, original_grid_shape[-1]])
    block_probe_real_batch = np.zeros([len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width])
    block_probe_imag_batch = np.zeros([len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width])

    for ind, i_pos in enumerate(this_pos_ind_ls):
        line_st = i_pos // n_blocks_x * block_size
        line_end = line_st + block_size
        px_st = i_pos % n_blocks_x * block_size
        px_end = px_st + block_size

        # sub_grids are memmaps
        sub_grid_delta = grid_delta[0, max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, original_grid_shape[0]),
                         max([0, px_st - safe_zone_width]):min([px_end + safe_zone_width, original_grid_shape[1]]),
                         :]
        sub_grid_beta = grid_beta[0, max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, original_grid_shape[0]),
                        max([0, px_st - safe_zone_width]):min([px_end + safe_zone_width, original_grid_shape[1]]),
                        :]
        sub_probe_real = probe_real[max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, original_grid_shape[0]), max([0, px_st - safe_zone_width]):min([px_end + safe_zone_width, original_grid_shape[1]])]
        sub_probe_imag = probe_imag[max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, original_grid_shape[0]), max([0, px_st - safe_zone_width]):min([px_end + safe_zone_width, original_grid_shape[1]])]

        print(i_pos, sub_grid_delta.shape, 'before pad', line_st, line_end, px_st, px_end)

        # During padding, sub_grids are read into the RAM
        pad_top, pad_bottom, pad_left, pad_right = (0, 0, 0, 0)
        if line_st < safe_zone_width:
            sub_grid_delta = np.pad(sub_grid_delta, [[safe_zone_width - line_st, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta, [[safe_zone_width - line_st, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
            pad_top = safe_zone_width - line_st
        if (original_grid_shape[0] - line_end + 1) < safe_zone_width:
            sub_grid_delta = np.pad(sub_grid_delta, [[0, line_end + safe_zone_width - original_grid_shape[0]], [0, 0], [0, 0]], mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta, [[0, line_end + safe_zone_width - original_grid_shape[0]], [0, 0], [0, 0]], mode='constant', constant_values=0)
            pad_bottom = line_end + safe_zone_width - original_grid_shape[0]
        if px_st < safe_zone_width:
            sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [safe_zone_width - px_st, 0], [0, 0]], mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [safe_zone_width - px_st, 0], [0, 0]], mode='constant', constant_values=0)
            pad_left = safe_zone_width - px_st
        if (original_grid_shape[1] - px_end + 1) < safe_zone_width:
            sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [0, px_end + safe_zone_width - original_grid_shape[1]], [0, 0]], mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [0, px_end + safe_zone_width - original_grid_shape[1]], [0, 0]], mode='constant', constant_values=0)
            pad_right = px_end + safe_zone_width - original_grid_shape[1]
        sub_probe_real = np.pad(sub_probe_real, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='edge')
        sub_probe_imag = np.pad(sub_probe_imag, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='edge')


        block_delta_batch[ind, :, :, :] = sub_grid_delta
        block_beta_batch[ind, :, :, :] = sub_grid_beta
        block_probe_real_batch[ind, :, :] = sub_probe_real
        block_probe_imag_batch[ind, :, :] = sub_probe_imag

    dt_ls = np.zeros(n_repeats)
    for i in range(n_repeats):

        wavefield, dt = multislice_propagate_batch_numpy(block_delta_batch, block_beta_batch, block_probe_real_batch, block_probe_imag_batch, energy_ev,
                                                         [psize_cm] * 3, obj_batch_shape=block_delta_batch.shape,
                                                         return_fft_time=True, starting_slice=0, t_init=0,
                                                         debug=False, debug_save_path=None,
                                                         rank=rank, verbose=verbose)

        t0 = time.time()
        this_full_wavefield = np.zeros([n_batch, *original_grid_shape[:-1]], dtype='complex64')
        for ind, i_pos in enumerate(this_pos_ind_ls):
            line_st = i_pos // n_blocks_x * block_size
            line_end = min([line_st + block_size, original_grid_shape[0]])
            px_st = i_pos % n_blocks_x * block_size
            px_end = min([px_st + block_size, original_grid_shape[1]])
            this_full_wavefield[0, line_st:line_end, px_st:px_end] = wavefield[ind,
                                                                     safe_zone_width:safe_zone_width + (line_end - line_st),
                                                                     safe_zone_width:safe_zone_width + (px_end - px_st)]
        full_wavefield = np.zeros_like(this_full_wavefield, dtype='complex64')
        comm.Allreduce(this_full_wavefield, full_wavefield)
        dt += time.time() - t0
        dt_ls[i] = dt

        if rank == 0 and i == 0:
            dxchange.write_tiff(abs(full_wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size), 'partial_fft_output'), dtype='float32', overwrite=True)
        comm.Barrier()

    dt_avg = np.mean(dt_ls)
    if rank == 0:
        print('PFFT: For size {}, average dt = {} s.'.format(this_size, dt_avg))
        img = dxchange.read_tiff(os.path.join(path_prefix, 'size_{}'.format(this_size), 'partial_fft_output.tiff'))
        f.write('pfft,{},0,{},{},{}\n'.format(this_size, safe_zone_width, dt_avg, np.mean((img - ref) ** 2)))
        f.flush()
        os.fsync(f.fileno())

    comm.Barrier()

if rank == 0:
    f.close()