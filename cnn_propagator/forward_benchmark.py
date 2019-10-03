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
    f = open(os.path.join(path_prefix, 'report_fft.csv'), 'a')
    f.write('algorithm,object_size,kernel_size,safezone_width,avg_time,mse_with_fft\n')

# Do a FFT based propagation
for this_size in size_ls:
    is_large_array = False
    debug_save_path = None
    grid_delta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_delta.tiff').format(this_size))
    grid_beta = dxchange.read_tiff(os.path.join(path_prefix, 'phantom', 'size_{}', 'grid_beta.tiff').format(this_size))
    grid_delta = np.swapaxes(np.swapaxes(grid_delta, 0, 1), 1, 2)
    grid_beta = np.swapaxes(np.swapaxes(grid_beta, 0, 1), 1, 2)
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    # Start from where it stopped
    i_st = 0
    t_init = 0
    verbose = True if rank == 0 else False
    if this_size == 4096:
        is_large_array = True
        debug_save_path = os.path.join(path_prefix, 'size_{}', 'debug').format(this_size)
        if rank == 0:
            if not os.path.exists(debug_save_path):
                try:
                    os.makedirs(debug_save_path)
                except:
                    warnings.warn('Failed to create debug_save_path.')
        comm.Barrier()
        if os.path.exists(os.path.join(debug_save_path, 'current_islice_rank_{}.txt'.format(rank))):
            i_st, t_init = np.loadtxt(os.path.join(debug_save_path, 'current_islice_rank_{}.txt'.format(rank)))
            i_st = int(i_st)
            probe_real = dxchange.read_tiff(os.path.join(debug_save_path, 'probe_real_rank_{}.tiff'.format(rank)))
            probe_imag = dxchange.read_tiff(os.path.join(debug_save_path, 'probe_imag_rank_{}.tiff'.format(rank)))
    size_factor = size_ls[-1] // this_size
    psize_cm = psize_min_cm * size_factor

    dt_ls = np.zeros(n_ranks)
    dt_ls_final = np.zeros(n_ranks)

    # t0 = time.time()
    wavefield, dt = multislice_propagate_batch_numpy(grid_delta, grid_beta, probe_real, probe_imag, energy_ev, [psize_cm] * 3, obj_batch_shape=grid_delta.shape, return_fft_time=True, starting_slice=i_st, t_init=t_init, debug=is_large_array, debug_save_path=debug_save_path, rank=rank, verbose=verbose)
    # dt = time.time() - t0
    dt_ls[rank] = dt
    dxchange.write_tiff(abs(wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size), 'fft_output'), dtype='float32', overwrite=True)

    print('FFT (rank {}): For n_ranks {}, dt = {} s.'.format(rank, this_size, dt))
    comm.Allreduce(dt_ls, dt_ls_final)

    if rank == 0:
        np.savetxt(os.path.join(path_prefix, 'size_{}', 'all_rank_timing.txt').format(this_size), dt_ls_final, delimiter=',')
        dt_avg = np.mean(dt_ls_final)
        f.write('fft,{},0,0,{},0\n'.format(this_size, dt_avg))
        f.flush()
        os.fsync(f.fileno())

comm.Barrier()

if rank == 0:
    f.close()