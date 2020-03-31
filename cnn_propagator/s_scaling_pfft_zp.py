from mpi4py import MPI
import dxchange
import time
import warnings
import math
import numpy as np
import h5py
from tqdm import trange
import sys
import os
import pickle
from math import ceil, floor
from propagation_fft import multislice_propagate_batch_numpy
import argparse

converging_steps_dict = {4096: 24,
                         8192: 24,
                         16384: 23,
                         32768: 23,
                         65536: 22}

n_repeats = 10

parser = argparse.ArgumentParser()
parser.add_argument('--size', default='None')
parser.add_argument('--nodes', default=256)
args = parser.parse_args()
this_size = int(args.size)
n_nodes = int(args.nodes)

n_slices = converging_steps_dict[this_size]

t_limit = 999
t_zero = time.time()
hdf5 = True

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


path_prefix = os.path.join(os.getcwd(), 'zp')

verbose = True if rank == 0 else False

# Create folder
if rank == 0:
    try:
        os.makedirs(os.path.join(path_prefix, 'size_{}'.format(this_size), 'nd_{}'.format(n_nodes)))
    except:
        print('Target folder exists.')

# Create report
if rank == 0:
    f = open(os.path.join(path_prefix, 'size_{}'.format(this_size), 'nd_{}'.format(n_nodes), 'report_pfft.csv'), 'w')
    if os.path.getsize(os.path.join(path_prefix, 'size_{}'.format(this_size), 'nd_{}'.format(n_nodes), 'report_pfft.csv')) == 0:
        f.write('i_repeat,n_nodes,this_size,n_slices,safe_zone_width,n_blocks_y,n_blocks_x,block_size,n_ranks,dt_total,dt_read_div,dt_write,dt_fft_prop\n')

# Benchmark partial FFT propagation

parameters = pickle.load(
    open(os.path.join(path_prefix, 'size_{}'.format(this_size), 'parameters.pickle'), 'rb'))
beta = parameters['beta']
delta = parameters['delta']
psize_cm = parameters['step_xy'] * 1e2
lmbda_nm = parameters['wavelength in m'] * 1e9
energy_ev = parameters['energy(in eV)']
focal_len_m = parameters['focal_length']
thick_zp_cm = 30.808e-4
free_prop_cm = 0
slice_spacing_cm = thick_zp_cm / n_slices

if rank == 0: print('This size is {}. This n_slices is {}'.format(this_size, n_slices))

for i in range(n_repeats):

    t_tot_0 = time.time()

    t_read_div_0 = time.time()
    img = np.load(os.path.join(path_prefix, 'size_{}', 'zp.npy').format(this_size), mmap_mode='r')
    img_shape = img.shape
    # grid_delta = np.ones([1, *img_shape, 1]) * img * delta
    # grid_beta = np.ones([1, *img_shape, 1]) * img * beta
    # grid_delta = np.swapaxes(np.swapaxes(grid_delta, 0, 1), 1, 2)
    # grid_beta = np.swapaxes(np.swapaxes(grid_beta, 0, 1), 1, 2)
    # grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    # grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
    n_batch = 1
    original_grid_shape = [img.shape[0], img.shape[1], 1]

    safe_zone_width = ceil(
        4.0 * np.sqrt((slice_spacing_cm * 1e7 * n_slices + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7))
    # z_nm = slice_spacing_cm * 1e7 * n_slices + free_prop_cm * 1e7
    # safe_zone_width = np.sqrt(z_nm ** 2 / (4 * (psize_cm * 1e7) ** 2 / lmbda_nm ** 2 - 1))
    # safe_zone_width = ceil(1.1 * safe_zone_width)
    if rank == 0: print('  Safe zone width is {}.'.format(safe_zone_width))

    # Must satisfy:
    # 1. n_block_x * n_block_y = n_ranks
    # 2. block_size * n_block_y = wave_shape[0]
    # 3. block_size * n_block_x = wave_shape[1]

    # n_blocks_x = int(np.sqrt(float(n_ranks) * original_grid_shape[1] / original_grid_shape[0])) + 1
    # n_blocks_y = int(np.sqrt(float(n_ranks) * original_grid_shape[0] / original_grid_shape[1])) + 1
    # n_blocks = n_blocks_x * n_blocks_y
    # block_size = int(float(original_grid_shape[0]) / n_blocks_y) + 1
    n_blocks_y = int(np.sqrt(original_grid_shape[0] / original_grid_shape[1] * n_ranks))
    n_blocks_x = int(np.sqrt(original_grid_shape[1] / original_grid_shape[0] * n_ranks))
    n_blocks = n_blocks_x * n_blocks_y
    block_size = ceil(max([original_grid_shape[0] / n_blocks_y, original_grid_shape[1] / n_blocks_x]))
    if rank == 0:
        print('n_blocks_y: ', n_blocks_y)
        print('n_blocks_x: ', n_blocks_x)
        print('n_blocks: ', n_blocks)
        print('block_size: ', block_size)

    this_pos_ind_ls = range(rank, n_blocks, n_ranks)
    block_delta_batch = np.zeros(
        [len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width,
         original_grid_shape[-1]])
    block_beta_batch = np.zeros(
        [len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width,
         original_grid_shape[-1]])
    block_probe_real_batch = np.zeros(
        [len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width])
    block_probe_imag_batch = np.zeros(
        [len(this_pos_ind_ls), block_size + 2 * safe_zone_width, block_size + 2 * safe_zone_width])

    for ind, i_pos in enumerate(this_pos_ind_ls):
        line_st = i_pos // n_blocks_x * block_size
        line_end = line_st + block_size
        px_st = i_pos % n_blocks_x * block_size
        px_end = px_st + block_size

        # sub_grids are memmaps
        sub_grid_delta = img[max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width,
                                                                     original_grid_shape[0]),
                         max([0, px_st - safe_zone_width]):min([px_end + safe_zone_width, original_grid_shape[1]])]
        sub_grid_beta = img[
                        max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, original_grid_shape[0]),
                        max([0, px_st - safe_zone_width]):min([px_end + safe_zone_width, original_grid_shape[1]])]
        sub_grid_delta = np.reshape(sub_grid_delta, [1, *sub_grid_delta.shape, 1]) * delta
        sub_grid_beta = np.reshape(sub_grid_beta, [1, *sub_grid_beta.shape, 1]) * beta
        sub_probe_real = np.ones(sub_grid_delta.shape[1:3])
        sub_probe_imag = np.zeros(sub_grid_delta.shape[1:3])

        # During padding, sub_grids are read into the RAM
        pad_top, pad_bottom, pad_left, pad_right = (0, 0, 0, 0)
        if line_st < safe_zone_width:
            pad_top = safe_zone_width - line_st
        if (original_grid_shape[0] - line_end) < safe_zone_width:
            pad_bottom = line_end + safe_zone_width - original_grid_shape[0]
        if px_st < safe_zone_width:
            pad_left = safe_zone_width - px_st
        if (original_grid_shape[1] - px_end) < safe_zone_width:
            pad_right = px_end + safe_zone_width - original_grid_shape[1]
        sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                                mode='edge')
        sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                               mode='edge')
        sub_probe_real = np.pad(sub_probe_real, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='edge')
        sub_probe_imag = np.pad(sub_probe_imag, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='edge')

        block_delta_batch[ind, :, :, :] = sub_grid_delta
        block_beta_batch[ind, :, :, :] = sub_grid_beta
        block_probe_real_batch[ind, :, :] = sub_probe_real
        block_probe_imag_batch[ind, :, :] = sub_probe_imag

    comm.Barrier()
    dt_read_div = time.time() - t_read_div_0

    # -------------------------------------

    if block_delta_batch.shape[0] > 0:
        comm.Barrier()
        # -----------------------------------------
        wavefield, dt = multislice_propagate_batch_numpy(block_delta_batch, block_beta_batch,
                                                         block_probe_real_batch, block_probe_imag_batch, energy_ev,
                                                         [psize_cm, psize_cm, slice_spacing_cm],
                                                         obj_batch_shape=block_delta_batch.shape,
                                                         return_fft_time=True, starting_slice=0, t_init=0,
                                                         debug=False, debug_save_path=None,
                                                         rank=rank, verbose=verbose, repeating_slice=n_slices)
        comm.Barrier()
        dt_fft_prop = dt
        # -----------------------------------------
        t_write_0 = time.time()
        comm.Barrier()
        f_out = h5py.File(os.path.join(path_prefix, 'size_{}'.format(this_size), 'nd_{}'.format(n_nodes),
                                      'pfft_nslices_{}_output_{}.h5'.format(n_slices, i)),
                                      'w', driver='mpio', comm=comm)
        print(wavefield.dtype)
        dset = f_out.create_dataset('wavefield', original_grid_shape[:-1], dtype='complex64', chunks=True)
        pos_ind_ls = range(rank, n_blocks, n_ranks)
        for ind, i_pos in enumerate(pos_ind_ls):
            line_st = i_pos // n_blocks_x * block_size
            line_end = min([line_st + block_size, original_grid_shape[0]])
            px_st = i_pos % n_blocks_x * block_size
            px_end = min([px_st + block_size, original_grid_shape[1]])
            dset[line_st:line_end, px_st:px_end] = wavefield[ind,
                                                              safe_zone_width:safe_zone_width + (line_end - line_st),
                                                              safe_zone_width:safe_zone_width + (px_end - px_st)]
        comm.Barrier()
        f_out.close()
        comm.Barrier()
        dt_write = time.time() - t_write_0
        # -----------------------------------------
        dt_total = time.time() - t_tot_0

    comm.Barrier()

    if rank == 0:
        print('PFFT: For size {}, average dt = {} s.'.format(this_size, dt_total))
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i, n_nodes, this_size, n_slices, safe_zone_width, n_blocks_y, n_blocks_x, block_size, n_ranks, dt_total, dt_read_div, dt_write, dt_fft_prop))
        f.flush()
        os.fsync(f.fileno())


comm.Barrier()


if rank == 0:
    f.close()
