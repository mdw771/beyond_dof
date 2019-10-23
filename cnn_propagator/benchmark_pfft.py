from mpi4py import MPI
import dxchange
import time
import warnings
import math
import numpy as np
from tqdm import trange
import sys
import os
import pickle
from math import ceil, floor
from propagation import multislice_propagate_cnn
from propagation_fft import multislice_propagate_batch_numpy


t_limit = 170
t_zero = time.time()

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


def save_checkpoint(this_size_ind, this_nslice_ind):
    np.savetxt(os.path.join(path_prefix, 'checkpoint.txt'), np.array([this_size_ind, this_nslice_ind]))
    return

path_prefix = os.path.join(os.getcwd(), 'zp')
######################################################################
size_ls = 4096 * np.array([1, 2, 4, 8, 16]).astype('int')
n_slices_ls = np.arange(10, 600, 5)
# size_ls = [4096]
# n_slices_ls = [50]
try:
    cp = np.loadtxt(os.path.join(path_prefix, 'checkpoint.txt'))
    i_starting_size = int(cp[0])
    i_starting_nslice = int(cp[1])
except:
    i_starting_size = 0
    i_starting_nslice = 0
#####################################################################
n_repeats = 1

# Create report
if rank == 0:
    f = open(os.path.join(path_prefix, 'report_pfft.csv'), 'a')
    if os.path.getsize(os.path.join(path_prefix, 'report_pfft.csv')) == 0:
        f.write('algorithm,object_size,n_slices,safezone_width,avg_working_time,avg_total_time\n')

# Benchmark partial FFT propagation
for this_size in np.take(size_ls, range(i_starting_size, len(size_ls))):

    for n_slices in np.take(n_slices_ls, range(i_starting_nslice, len(n_slices_ls))):
    
        parameters = pickle.load(open(os.path.join(path_prefix, 'size_{}'.format(this_size), 'parameters.pickle'), 'rb'))
        beta = parameters['beta']
        delta = parameters['delta']
        psize_cm = parameters['step_xy'] * 1e2
        lmbda_nm = parameters['wavelength in m'] * 1e9
        energy_ev = parameters['energy(in eV)']
        focal_len_m = parameters['focal_length']
        thick_zp_cm = 10e-4
        free_prop_cm = 0
        slice_spacing_cm = thick_zp_cm / n_slices

        if rank == 0: print('This size is {}. This n_slices is {}'.format(this_size, n_slices))
        img = np.load(os.path.join(path_prefix, 'size_{}', 'zp.npy').format(this_size))
        img_shape = img.shape
        img = np.reshape(img, [1, *img_shape, 1])
        grid_delta = np.ones([1, *img_shape, 1]) * img * delta
        grid_beta = np.ones([1, *img_shape, 1]) * img * beta
        # grid_delta = np.swapaxes(np.swapaxes(grid_delta, 0, 1), 1, 2)
        # grid_beta = np.swapaxes(np.swapaxes(grid_beta, 0, 1), 1, 2)
        # grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
        # grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
        n_batch = grid_delta.shape[0]
        original_grid_shape = grid_delta.shape[1:]

        size_factor = size_ls[-1] // this_size
        # psize_cm = psize_min_cm * size_factor

        probe_real = np.ones([*grid_delta.shape[1:3]])
        probe_imag = np.zeros([*grid_delta.shape[1:3]])

        ref = dxchange.read_tiff(
            os.path.join(path_prefix, 'size_{}'.format(this_size), 'fft_output.tiff'))

        safe_zone_width = ceil(
            4.0 * np.sqrt((slice_spacing_cm * 1e7 * n_slices + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7))
        if rank == 0: print('  Safe zone width is {}.'.format(safe_zone_width))

        # Must satisfy:
        # 1. n_block_x * n_block_y = n_ranks
        # 2. block_size * n_block_y = wave_shape[0]
        # 3. block_size * n_block_x = wave_shape[1]
        n_blocks_x = int(np.sqrt(float(n_ranks) * original_grid_shape[1] / original_grid_shape[0])) + 1
        n_blocks_y = int(np.sqrt(float(n_ranks) * original_grid_shape[0] / original_grid_shape[1])) + 1
        n_blocks = n_blocks_x * n_blocks_y
        block_size = int(float(original_grid_shape[0]) / n_blocks_y) + 1

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

        try:
            raise Exception
            dt_ls = np.loadtxt(os.path.join(path_prefix, 'size_{}'.format(this_size), 'dt_all_repeats.txt'))
        except:
            # 1st column excludes hard drive I/O, while 2nd column is the total time.
            dt_ls = np.zeros([n_repeats, 2])
        for i in range(n_repeats):
            t_tot_0 = time.time()
            wavefield, dt = multislice_propagate_batch_numpy(block_delta_batch, block_beta_batch, block_probe_real_batch, block_probe_imag_batch, energy_ev,
                                                             [psize_cm, psize_cm, slice_spacing_cm], obj_batch_shape=block_delta_batch.shape,
                                                             return_fft_time=True, starting_slice=0, t_init=0,
                                                             debug=False, debug_save_path=None,
                                                             rank=rank, verbose=True, repeating_slice=n_slices)

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
            dt_ls[i, 0] = dt
            dt_ls[i, 1] = time.time() - t_tot_0

            if rank == 0 and i == 0:
                dxchange.write_tiff(abs(full_wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size), 'pfft_nslices_{}_output.tiff'.format(n_slices)), dtype='float32', overwrite=True)
                np.save(os.path.join(path_prefix, 'size_{}'.format(this_size), 'pfft_nslices_{}_output'.format(n_slices)), full_wavefield)
            # if rank == 0:
            #     np.savetxt(os.path.join(path_prefix, 'size_{}'.format(this_size), 'dt_all_repeats.txt'), dt_ls)
            comm.Barrier()

        dt_avg, dt_tot_avg = np.mean(dt_ls, axis=0)
        if rank == 0:
            print('PFFT: For size {}, average dt = {} s.'.format(this_size, dt_avg))
            f.write('pfft,{},{},{},{},{}\n'.format(this_size, n_slices, safe_zone_width, dt_avg, dt_tot_avg))
            f.flush()
            os.fsync(f.fileno())
            i_starting_nslice += 1
            save_checkpoint(i_starting_size, i_starting_nslice)

        # Exit with status 0 before allocated time runs out
        if (time.time() - t_zero) / 60 >= t_limit: sys.exit()

        comm.Barrier()

    i_starting_nslice = 0
    i_starting_size += 1

if rank == 0:
    f.close()