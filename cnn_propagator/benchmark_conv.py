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
import pickle
from tqdm import trange
import sys
import os
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
    var_dict = {'this_size_ind': this_size_ind, 'this_nslice_ind': this_nslice_ind}
    pickle.dump(var_dict, open(os.path.join(path_prefix, 'checkpoint'), 'wb'))
    return

path_prefix = os.path.join(os.getcwd(), 'zp')
######################################################################
# size_ls = 4096 * np.array([1, 2, 4, 8, 16]).astype('int')
# n_slices_ls = np.arange(10, 600, 5)
size_ls = [4096]
n_slices_ls = [50]
try:
    cp = pickle.load(open(os.path.join(path_prefix, 'checkpoint'), 'rb'))
    i_starting_size = cp['this_size_ind']
    i_starting_nslice = cp['this_nslice_ind']
except:
    i_starting_size = 0
    i_starting_nslice = 0
#####################################################################
n_repeats = 1

# Create report
if rank == 0:
    f = open(os.path.join(path_prefix, 'report_conv.csv'), 'a')
    if os.path.getsize(os.path.join(path_prefix, 'report_conv.csv')) == 0:
        f.write('algorithm,object_size,kernel_size,n_slices,safezone_width,avg_working_time,avg_total_time\n')

# Benchmark convolution propagation
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
        kernel_size = 59
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
            4.0 * np.sqrt((slice_spacing_cm * 1e7 * n_slices + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7)) + (kernel_size // 2) + 1
        if rank == 0: print('  This kernel size is {}; safe zone width is {}.'.format(kernel_size, safe_zone_width))

        # Calculate the block range to be processed by each rank.
        # If the number of ranks is smaller than the number of lines, each rank will take 1 or more
        # whole lines.
        n_lines = grid_delta.shape[1]
        n_pixels_per_line = grid_delta.shape[2]
        if n_ranks <= n_lines:
            n_ranks_spill = n_lines % n_ranks
            line_st = n_lines // n_ranks * rank + min([n_ranks_spill, rank])
            line_end = line_st + n_lines // n_ranks
            if rank < n_ranks_spill:
                line_end += 1
            px_st = 0
            px_end = n_pixels_per_line
            safe_zone_width_side = 0
        else:
            n_lines_spill = n_ranks % n_lines
            n_ranks_per_line_base = n_ranks // n_lines
            n_ranks_spill = n_lines_spill * (n_ranks_per_line_base + 1)
            line_st = rank // (n_ranks_per_line_base + 1) if rank < n_ranks_spill else (rank - n_ranks_spill) // n_ranks_per_line_base + n_lines_spill
            line_end = line_st + 1
            n_ranks_per_line = n_ranks_per_line_base if line_st < n_lines_spill else n_ranks_per_line_base + 1
            i_seg = rank % n_ranks_per_line
            px_st = int(n_pixels_per_line * (float(i_seg) / n_ranks_per_line))
            px_end = min([int(n_pixels_per_line * (float(i_seg + 1) / n_ranks_per_line)) + 1, n_pixels_per_line])
            safe_zone_width_side = safe_zone_width

        # sub_grids are memmaps
        sub_grid_delta = grid_delta[:, max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, n_lines),
                         max([0, px_st - safe_zone_width_side]):min([px_end + safe_zone_width_side, n_pixels_per_line]),
                         :]
        sub_grid_beta = grid_beta[:, max([0, line_st - safe_zone_width]):min(line_end + safe_zone_width, n_lines),
                        max([0, px_st - safe_zone_width_side]):min([px_end + safe_zone_width_side, n_pixels_per_line]),
                        :]

        # During padding, sub_grids are read into the RAM
        pad_top, pad_bottom, pad_left, pad_right = (0, 0, 0, 0)
        if line_st < safe_zone_width:
            sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [safe_zone_width - line_st, 0], [0, 0], [0, 0]],
                                    mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [safe_zone_width - line_st, 0], [0, 0], [0, 0]],
                                   mode='constant', constant_values=0)
            probe_real = np.pad(probe_real, [[safe_zone_width - line_st, 0], [0, 0]], mode='edge')
            probe_imag = np.pad(probe_imag, [[safe_zone_width - line_st, 0], [0, 0]], mode='edge')
            pad_top = safe_zone_width - line_st
        if (n_lines - line_end + 1) < safe_zone_width:
            sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [0, line_end + safe_zone_width - n_lines], [0, 0], [0, 0]],
                                    mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [0, line_end + safe_zone_width - n_lines], [0, 0], [0, 0]],
                                   mode='constant', constant_values=0)
            probe_real = np.pad(probe_real, [[0, line_end + safe_zone_width - n_lines], [0, 0]], mode='edge')
            probe_imag = np.pad(probe_imag, [[0, line_end + safe_zone_width - n_lines], [0, 0]], mode='edge')
            pad_bottom = safe_zone_width
        if safe_zone_width_side > 0:
            if px_st < safe_zone_width_side:
                sub_grid_delta = np.pad(sub_grid_delta,
                                        [[0, 0], [0, 0], [safe_zone_width_side - px_st, 0], [0, 0]],
                                        mode='constant', constant_values=0)
                sub_grid_beta = np.pad(sub_grid_beta,
                                       [[0, 0], [0, 0], [safe_zone_width_side - px_st, 0], [0, 0]],
                                       mode='constant', constant_values=0)
                probe_real = np.pad(probe_real, [[0, 0], [safe_zone_width_side - px_st, 0, 0]], mode='edge')
                probe_imag = np.pad(probe_imag, [[0, 0], [safe_zone_width_side - px_st, 0, 0]], mode='edge')
                pad_left = safe_zone_width_side - px_st
            if (n_pixels_per_line - px_end + 1) < safe_zone_width_side:
                sub_grid_delta = np.pad(sub_grid_delta,
                                        [[0, 0], [0, 0], [0, px_end + safe_zone_width_side - n_pixels_per_line], [0, 0]],
                                        mode='constant', constant_values=0)
                sub_grid_beta = np.pad(sub_grid_beta,
                                       [[0, 0], [0, 0], [0, px_end + safe_zone_width_side - n_pixels_per_line], [0, 0]],
                                       mode='constant', constant_values=0)
                probe_real = np.pad(probe_real, [[0, 0], [0, px_end + safe_zone_width_side - n_pixels_per_line]], mode='edge')
                probe_imag = np.pad(probe_imag, [[0, 0], [0, px_end + safe_zone_width_side - n_pixels_per_line]], mode='edge')
                pad_right = px_end + safe_zone_width_side - n_pixels_per_line

        # Start from where it stopped
        i_repeat = 0
        verbose = True if rank == 0 else False
        debug_save_path = os.path.join(path_prefix, 'size_{}', 'debug').format(this_size)
        if rank == 0:
            if not os.path.exists(debug_save_path):
                try:
                    os.makedirs(debug_save_path)
                except:
                    warnings.warn('Failed to create debug_save_path.')
        comm.Barrier()
        # if os.path.exists(os.path.join(debug_save_path, 'current_irepeat_conv_kernel_{}.txt'.format(kernel_size))):
        #     i_repeat = np.loadtxt(os.path.join(debug_save_path, 'current_irepeat_conv_kernel_{}.txt'.format(kernel_size)))
        #     i_repeat = int(i_repeat)

        try:
            raise Exception
            dt_ls = np.loadtxt(os.path.join(path_prefix, 'size_{}'.format(this_size), 'dt_all_repeats.txt'))
            if dt_ls.ndim == 1:
                dt_ls = np.reshape(dt_ls, [1, 2])
        except:
            # 1st column excludes hard drive I/O, while 2nd column is the total time.
            dt_ls = np.zeros([n_repeats, 2])
        for i in range(i_repeat, n_repeats):
            if rank == 0: print('    This i_repeat is {}.'.format(i_repeat))
            # np.savetxt(os.path.join(debug_save_path, 'current_irepeat_conv_kernel_{}.txt'.format(kernel_size)), np.array([i]))
            t_tot_0 = time.time()
            # dxchange.write_tiff(sub_grid_delta[0, :, :, 0], 'zp/size_256/sub_grid_delta', dtype='float32', overwrite=True)
            # dxchange.write_tiff(sub_grid_beta[0, :, :, 0], 'zp/size_256/sub_grid_beta', dtype='float32', overwrite=True)
            # dxchange.write_tiff(probe_real, 'zp/size_256/probe_real', dtype='float32', overwrite=True)
            # dxchange.write_tiff(probe_imag, 'zp/size_256/probe_imag', dtype='float32', overwrite=True)
            wavefield, dt = multislice_propagate_cnn(sub_grid_delta, sub_grid_beta,
                                                 probe_real[
                                                 pad_top + line_st - safe_zone_width:pad_top + line_end + safe_zone_width,
                                                 pad_left + px_st - safe_zone_width_side:pad_left + px_end + safe_zone_width_side],
                                                 probe_imag[
                                                 pad_top + line_st - safe_zone_width:pad_top + line_end + safe_zone_width,
                                                 pad_left + px_st - safe_zone_width_side:pad_left + px_end + safe_zone_width_side],
                                                 energy_ev, [psize_cm, psize_cm, slice_spacing_cm], kernel_size=kernel_size, free_prop_cm=None,
                                                 debug=False,
                                                 return_fft_time=True,
                                                 debug_save_path=debug_save_path,
                                                 original_kernel_shape=original_grid_shape[:2],
                                                 rank=rank, t_init=0, verbose=verbose, starting_slice=0, repeating_slice=n_slices)

            t0 = time.time()
            this_full_wavefield = np.zeros([n_batch, *original_grid_shape[:-1]], dtype='complex64')
            this_full_wavefield[:, line_st:line_end, px_st:px_end] = wavefield[:,
                                                                     safe_zone_width:safe_zone_width + (line_end - line_st),
                                                                     safe_zone_width_side:safe_zone_width_side + (px_end - px_st)]
            full_wavefield = np.zeros_like(this_full_wavefield, dtype='complex64')
            comm.Allreduce(this_full_wavefield, full_wavefield)
            dt += (time.time() - t0)
            dt_ls[i, 0] = dt
            dt_ls[i, 1] = time.time() - t_tot_0

            if rank == 0 and i == 0:
                dxchange.write_tiff(abs(full_wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size), 'conv_kernel_{}_nslices_{}_output.tiff'.format(kernel_size, n_slices)), dtype='float32', overwrite=True)
                np.save(os.path.join(path_prefix, 'size_{}'.format(this_size), 'conv_kernel_{}_nslices_{}_output'.format(kernel_size, n_slices)), full_wavefield)
            # if rank == 0:
            #     np.savetxt(os.path.join(path_prefix, 'size_{}'.format(this_size), 'dt_all_repeats.txt'), dt_ls)
            comm.Barrier()

        dt_avg, dt_tot_avg = np.mean(dt_ls, axis=0)
        if rank == 0:
            print('CONV: For n_ranks {} and kernel n_ranks {}, average dt = {} s.'.format(this_size, kernel_size, dt_avg))
            # img = dxchange.read_tiff(os.path.join(path_prefix, 'size_{}'.format(this_size), 'conv_kernel_{}_nslices_{}_output.tiff'.format(kernel_size, n_slices)))
            f.write('conv,{},{},{},{},{},{}\n'.format(this_size, kernel_size, n_slices, safe_zone_width, dt_avg, dt_tot_avg))
            f.flush()
            os.fsync(f.fileno())
            # Save checkpoint
            i_starting_nslice += 1
            save_checkpoint(i_starting_size, i_starting_nslice)

        # Exit with status 0 before allocated time runs out
        if (time.time() - t_zero) / 60 >= t_limit: sys.exit()

        comm.Barrier()

    i_starting_nslice = 0
    i_starting_size += 1

if rank == 0:
    f.close()