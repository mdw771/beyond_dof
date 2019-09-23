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
    size = comm.Get_size()
    rank = comm.Get_rank()
    mpi_ok = True
except:
    from pseudo import Mpi
    comm = Mpi()
    size = 1
    rank = 0
    mpi_ok = False

energy_ev = 3000
lmbda_nm = 1.24 / (energy_ev / 1e3)
psize_min_cm = 100e-7
kernel_size_ls = 2 ** np.array([2, 3, 4, 5, 6]) + 1
free_prop_cm = None
######################################################################
size_ls = np.array([256, 512, 1024, 2048, 4096])
#####################################################################
path_prefix = os.path.join(os.getcwd(), 'charcoal')
n_repeats = 100

# Start from where it stopped
i_st = 0
try:
    f = open(os.path.join(path_prefix, 'size_{}'.format(size_ls[0]), 'timing.csv'), 'r')
    a = f.readlines()[-1]
    i_st = int(a[:a.find(',')])
except:
    pass

# Create report
if rank == 0:
    f = open(os.path.join(path_prefix, 'report.csv'), 'a')
#    f.write('algorithm,object_size,kernel_size,safezone_width,avg_time,mse_with_fft\n')

# Do a FFT based propagation
for this_size in size_ls:
    # f_temp = open(os.path.join(path_prefix, 'size_{}'.format(this_size), 'timing.csv'), 'a')
    grid_delta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_delta.tiff').format(this_size))
    grid_beta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_beta.tiff').format(this_size))
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])
    size_factor = size_ls[-1] // this_size
    psize_cm = psize_min_cm * size_factor

    for i in trange(i_st, n_repeats):
        t0 = time.time()
        wavefield = multislice_propagate_batch_numpy(grid_delta, grid_beta, probe_real, probe_imag, energy_ev, [psize_cm] * 3, obj_batch_shape=grid_delta.shape)
        dt = time.time() - t0
        if i == 0:
            dxchange.write_tiff(abs(wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size), 'fft_output'), dtype='float32', overwrite=True)
        # f_temp.write('{},{}\n'.format(i, dt))
        # f_temp.flush()
        # os.fsync(f_temp.fileno())
    dt_avg = 0
    print('FFT (rank {}): For size {}, dt = {} s.'.format(rank, this_size, dt))
    comm.Allreduce(dt, dt_avg)
    dt_avg /= size

    if rank == 0:
        f.write('fft,{},0,0,{},0\n'.format(this_size, dt_avg))
        f.flush()
        os.fsync(f.fileno())

comm.Barrier()

sys.exit()

for this_size in size_ls:

    grid_delta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_delta.tiff').format(this_size))
    grid_beta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_beta.tiff').format(this_size))
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
    n_batch = grid_delta.shape[0]
    original_grid_shape = grid_delta.shape[1:]

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    ref = dxchange.read_tiff(
        os.path.join(path_prefix, 'size_{}'.format(this_size), 'fft_output.tiff'))

    for kernel_size in kernel_size_ls:
        safe_zone_width = ceil(
            4.0 * np.sqrt((psize_cm * 1e7 * grid_delta.shape[-1] + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7)) + (
                                      kernel_size // 2) + 1

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
            line_st = rank // (n_ranks_per_line_base + 1) if rank < n_ranks_spill else (
                                                                                                   rank - n_ranks_spill) // n_ranks_per_line_base + n_lines_spill
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
        pad_top, pad_bottom = (0, 0)
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
            sub_grid_delta = np.pad(sub_grid_delta,
                                    [[0, 0], [0, 0], [safe_zone_width_side, safe_zone_width_side], [0, 0]],
                                    mode='constant', constant_values=0)
            sub_grid_beta = np.pad(sub_grid_beta,
                                   [[0, 0], [0, 0], [safe_zone_width_side, safe_zone_width_side], [0, 0]],
                                   mode='constant', constant_values=0)
            probe_real = np.pad(probe_real, [[0, 0], [safe_zone_width_side, safe_zone_width_side]], mode='edge')
            probe_imag = np.pad(probe_imag, [[0, 0], [safe_zone_width_side, safe_zone_width_side]], mode='edge')

        dt_ls = np.zeros(n_repeats)
        for i in range(n_repeats):
            t0 = time.time()
            wavefield = multislice_propagate_cnn(sub_grid_delta, sub_grid_beta,
                                                 probe_real[
                                                 pad_top + line_st - safe_zone_width:pad_top + line_end + safe_zone_width,
                                                 px_st:px_end + 2 * safe_zone_width_side],
                                                 probe_imag[
                                                 pad_top + line_st - safe_zone_width:pad_top + line_end + safe_zone_width,
                                                 px_st:px_end + 2 * safe_zone_width_side],
                                                 energy_ev, [psize_cm] * 3, kernel_size=kernel_size, free_prop_cm=free_prop_cm,
                                                 debug=False,
                                                 original_grid_shape=original_grid_shape)

            this_full_wavefield = np.zeros([n_batch, *original_grid_shape[:-1]], dtype='complex64')
            this_full_wavefield[:, line_st:line_end, px_st:px_end] = wavefield[:,
                                                                     safe_zone_width:safe_zone_width + (line_end - line_st),
                                                                     safe_zone_width_side:safe_zone_width_side + (px_end - px_st)]
            full_wavefield = np.zeros_like(this_full_wavefield, dtype='complex64')
            comm.Allreduce(this_full_wavefield, full_wavefield)
            dt = time.time() - t0
            dt_ls[i] = dt

            if rank == 0 and i == 0:
                dxchange.write_tiff(abs(full_wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size), 'conv_kernel_{}_output'.format(kernel_size)), dtype='float32', overwrite=True)
            comm.Barrier()

        dt_avg = np.mean(dt_ls)
        if rank == 0:
            print('CONV: For size {} and kernel size {}, average dt = {} s.'.format(this_size, kernel_size, dt_avg))
            img = dxchange.read_tiff(os.path.join(path_prefix, 'size_{}'.format(this_size), 'conv_kernel_{}_output.tiff'.format(kernel_size)))
            f.write('conv,{},{},{},{},{}\n'.format(this_size, kernel_size, safe_zone_width, dt_avg, np.mean((img - ref) ** 2)))
            f.flush()
            os.fsync(f.fileno())

        comm.Barrier()

if rank == 0:
    f.close()