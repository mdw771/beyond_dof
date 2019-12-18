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
import util

t_limit = 160
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


if rank == 0: print('Numpy is ', np.__file__)


def save_checkpoint(this_size_ind, this_nslice_ind):
    np.savetxt(os.path.join(path_prefix, 'checkpoint.txt'), np.array([this_size_ind, this_nslice_ind]), fmt='%d')
    return


def get_interpolated_slice(i_slice, n_repeats, n_slices_max, n_slices, slc=None, mask=False):
    """
    Get weighted summation of slice values, where each individual slice contained in a tiff file is normalized to have
    a mean of 1. The mean of the returned array should be equal to the thickness of the slab in the unit of
    sample_thickness / n_slices_max.
    :param i_slice: the current slice's index within range(0, n_slices).
    :param n_repeats: the number of slices in a period (either ascending or descending). Note that the number of tiff
                      files needed is n_repeats + 1, because the cycle goes like 0, 1, ..., 49 || 50, 49, ..., 1 || 0, ...
    :param n_slices_max: the number of original slices that compose the object. n_slices shouldn't go beyond this value.
    :param n_slices: the number of slices in the current case.
    :param slc: the lateral region of the slices to be summed. Should have the format of
                ((y_start, y_end), (x_start, x_end)). If None, then return the full slice.
    :return: weighted summation of the slice value.
    """

    prefix = 'mask' if mask else 'img'
    slice_full_ls = np.concatenate((range(n_repeats), range(n_repeats, 0, -1)))
    slice_full_ls = np.tile(slice_full_ls, n_slices_max // len(slice_full_ls))
    if n_slices_max % len(slice_full_ls) > 0:
        slice_full_ls = np.append(slice_full_ls, slice_full_ls[:n_slices_max % len(slice_full_ls)])
    # slice_ind_ls is always ascending, e.g. [0, 2.5, 5, 7.5, ..., 500]
    slice_ind_ls = np.linspace(0, n_slices_max, n_slices + 1) # The last index is not actually propagated
    # slice_ls is the numbers in filenames
    this_slice_ind = slice_ind_ls[i_slice] # Say, 2.5
    final_slice_ind = slice_ind_ls[i_slice + 1] # Say, 5
    if slc is None:
        shape = original_grid_shape[:2]
    else:
        shape = [slc[0][1] - slc[0][0], slc[1][1] - slc[1][0]]
    ri_slice = np.zeros(shape)
    total_dist = final_slice_ind - this_slice_ind

    dist_px = 0
    if np.ceil(this_slice_ind) - this_slice_ind < 1e-8: # If this_slice_ind is an integer itself
        pass
    else:
        this_ri_slice = dxchange.read_tiff(os.path.join(path_prefix, 'size_{}'.format(this_size), 'phantom', '{}_{:05}.tiff'.format(prefix, slice_full_ls[int(this_slice_ind)])))
        if slc is not None:
            this_ri_slice = this_ri_slice[slc[0][0]:slc[0][1], slc[1][0]:slc[1][1]]
        ri_slice += this_ri_slice * (np.ceil(this_slice_ind) - this_slice_ind)
        # if rank == 0: print('{}_{:05}.tiff * {}'.format(prefix, slice_full_ls[int(this_slice_ind)], np.ceil(this_slice_ind) - this_slice_ind))
        dist_px += (np.ceil(this_slice_ind) - this_slice_ind)
        this_slice_ind = np.ceil(this_slice_ind)
    while this_slice_ind + 1 <= final_slice_ind:
        this_ri_slice = dxchange.read_tiff(os.path.join(path_prefix, 'size_{}'.format(this_size), 'phantom', '{}_{:05}.tiff'.format(prefix, slice_full_ls[int(this_slice_ind)])))
        if slc is not None:
            this_ri_slice = this_ri_slice[slc[0][0]:slc[0][1], slc[1][0]:slc[1][1]]
        ri_slice += this_ri_slice
        # if rank == 0: print('{}_{:05}.tiff * {}'.format(prefix, slice_full_ls[int(this_slice_ind)], 1))
        this_slice_ind += 1
        dist_px += 1
    if final_slice_ind - this_slice_ind > 1e-8:
        this_ri_slice = dxchange.read_tiff(os.path.join(path_prefix, 'size_{}'.format(this_size), 'phantom', '{}_{:05}.tiff'.format(prefix, slice_full_ls[int(this_slice_ind)])))
        if slc is not None:
            this_ri_slice = this_ri_slice[slc[0][0]:slc[0][1], slc[1][0]:slc[1][1]]
        ri_slice += this_ri_slice * (final_slice_ind - np.floor(final_slice_ind))
        # if rank == 0: print('{}_{:05}.tiff * {}'.format(prefix, slice_full_ls[int(this_slice_ind)], final_slice_ind - np.floor(final_slice_ind)))
        this_slice_ind += 1
        dist_px += (final_slice_ind - np.floor(final_slice_ind))

    # Normalize over distance
    if rank == 0: print('Distance: {}'.format(dist_px))
    ri_slice /= total_dist
    return ri_slice


def get_padding_lengths(line_st, line_end, px_st, px_end, original_grid_shape, safe_zone_width):

    pad_top, pad_bottom, pad_left, pad_right = (0, 0, 0, 0)
    if line_st < safe_zone_width:
        pad_top = safe_zone_width - line_st
    if (original_grid_shape[0] - line_end) < safe_zone_width:
        pad_bottom = line_end + safe_zone_width - original_grid_shape[0]
    if px_st < safe_zone_width:
        pad_left = safe_zone_width - px_st
    if (original_grid_shape[1] - px_end) < safe_zone_width:
        pad_right = px_end + safe_zone_width - original_grid_shape[1]

    return pad_top, pad_bottom, pad_left, pad_right


path_prefix = os.path.join(os.getcwd(), 'charcoal')
######################################################################
psize_cm = 1e-7
energy_ev = 25000

# import xommons
# delta1 = xommons.ri_delta('Al', energy_ev / 1e3, 2.7)
# beta1 = xommons.ri_beta('Al', energy_ev / 1e3, 2.7)
# delta2 = xommons.ri_delta('Au', energy_ev / 1e3, 19.32)
# beta2 = xommons.ri_beta('Au', energy_ev / 1e3, 19.32)
delta1 = 8.666320754358026e-07
beta1 = 1.95600602233921e-09
delta2 = 5.1053512407639445e-06
beta2 = 3.3630855527288826e-07
# print(delta, beta)
# delta = 6.638119376400908e-07
# beta = 2.4754720576473264e-10
# delta = 5.1053512407639445e-06
# beta = 3.3630855527288826e-07
if rank == 0: print('Refractive indices:', delta1, beta1)
if rank == 0: print('Refractive indices:', delta2, beta2)
safe_zone_factor = 2
safe_zone_width = 240

lmbda_nm = 1240. / energy_ev
n_slices_repeating = 50
n_slices_max = 1000
# size_ls = 4096 * np.array([1, 2, 4, 8, 16]).astype('int')
size_ls = [4096]
n_slices_ls = np.arange(25, 1001, 25)
# n_slices_ls = [99, 100]
# n_slices_ls = [1000]
# n_slices_ls = list(range(10, 100, 5)) + list(range(100, 600, 25))
# size_ls = [256]
# n_slices_ls = [10]
# thick_zp_cm = n_slices_max * psize_cm
thick_zp_cm = 110e-4

try:
    cp = np.loadtxt(os.path.join(path_prefix, 'checkpoint.txt'))
    i_starting_size = int(cp[0])
    i_starting_nslice = int(cp[1])
except:
    i_starting_size = 0
    i_starting_nslice = 0
#####################################################################
n_repeats = 1
verbose = True if rank == 0 else False

# Create report
if rank == 0:
    f = open(os.path.join(path_prefix, 'report_pfft.csv'), 'a')
    if os.path.getsize(os.path.join(path_prefix, 'report_pfft.csv')) == 0:
        f.write('algorithm,object_size,n_slices,safezone_width,total_time,propagation_time,reading_time,writing_time\n')

# Benchmark partial FFT propagation
for this_size in np.take(size_ls, range(i_starting_size, len(size_ls))):

    for n_slices in np.take(n_slices_ls, range(i_starting_nslice, len(n_slices_ls))):

        free_prop_cm = 0
        slice_spacing_cm = thick_zp_cm / n_slices

        if rank == 0: print('This size is {}. This n_slices is {}'.format(this_size, n_slices))
        # grid_delta = np.ones([1, *img_shape, 1]) * img * delta
        # grid_beta = np.ones([1, *img_shape, 1]) * img * beta
        # grid_delta = np.swapaxes(np.swapaxes(grid_delta, 0, 1), 1, 2)
        # grid_beta = np.swapaxes(np.swapaxes(grid_beta, 0, 1), 1, 2)
        # grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
        # grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])
        n_batch = 1
        original_grid_shape = [this_size, this_size, 1]

        size_factor = size_ls[-1] // this_size
        # psize_cm = psize_min_cm * size_factor

        ref = dxchange.read_tiff(
            os.path.join(path_prefix, 'size_{}'.format(this_size), 'fft_output.tiff'))

        # safe_zone_width = ceil(
        #     safe_zone_factor * np.sqrt((slice_spacing_cm * 1e7 * n_slices + free_prop_cm * 1e7) * lmbda_nm) / (psize_cm * 1e7))
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
            print('n_ranks: ', n_ranks)
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

        # Crop and/or pad incident waves
        for ind, i_pos in enumerate(this_pos_ind_ls):
            line_st = i_pos // n_blocks_x * block_size
            line_end = line_st + block_size
            px_st = i_pos % n_blocks_x * block_size
            px_end = px_st + block_size

            pad_top, pad_bottom, pad_left, pad_right = get_padding_lengths(line_st, line_end, px_st, px_end,
                                                                           original_grid_shape, safe_zone_width)

            sub_grid_shape = [min(line_end + safe_zone_width, original_grid_shape[0]) - max([0, line_st - safe_zone_width]),
                              min([px_end + safe_zone_width, original_grid_shape[1]]) - max([0, px_st - safe_zone_width])]

            sub_probe_real = np.ones(sub_grid_shape)
            sub_probe_imag = np.zeros(sub_grid_shape)

            sub_probe_real = np.pad(sub_probe_real, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='edge')
            sub_probe_imag = np.pad(sub_probe_imag, [[pad_top, pad_bottom], [pad_left, pad_right]], mode='edge')

            block_probe_real_batch[ind, :, :] = sub_probe_real
            block_probe_imag_batch[ind, :, :] = sub_probe_imag

        # Multislice expanded here

        h = util.get_kernel(slice_spacing_cm * 1e7, lmbda_nm, [psize_cm * 1e7, psize_cm * 1e7, slice_spacing_cm * 1e7], block_probe_real_batch.shape[1:], fresnel_approx=False)
        dt_ls = np.zeros([1, 2])

        dt_prop = 0
        dt_tot = 0
        dt_reading = 0
        dt_writing = 0
        t_tot_0 = time.time()
        for i_slice in trange(n_slices, disable=True):
            if rank == 0: print('Slice: {} / {}'.format(i_slice + 1, n_slices))
            for ind, i_pos in enumerate(this_pos_ind_ls):

                t_read_0 = time.time()
                sub_grid = get_interpolated_slice(i_slice, n_slices_repeating, n_slices_max, n_slices, slc=((max([0, line_st - safe_zone_width]), min([line_end + safe_zone_width, original_grid_shape[0]])),
                                                                                                            (max([0, px_st - safe_zone_width]), min([px_end + safe_zone_width, original_grid_shape[1]]))))
                mask = get_interpolated_slice(i_slice, n_slices_repeating, n_slices_max, n_slices, slc=((max([0, line_st - safe_zone_width]), min([line_end + safe_zone_width, original_grid_shape[0]])),
                                                                                                            (max([0, px_st - safe_zone_width]), min([px_end + safe_zone_width, original_grid_shape[1]]))), mask=True)
                dt_reading += (time.time() - t_read_0)

                # sub_grid_delta = sub_grid * delta
                # sub_grid_beta = sub_grid * beta
                # dxchange.write_tiff(mask, 'charcoal/size_4096/mask', dtype='float32')
                sub_grid_delta = sub_grid * (1 - mask) * delta1 + sub_grid * mask * delta2
                sub_grid_beta = sub_grid * (1 - mask) * beta1 + sub_grid * mask * beta2
                # dxchange.write_tiff(sub_grid_delta, os.path.join(path_prefix, 'size_4096', 'delta'), dtype='float32')
                # dxchange.write_tiff(sub_grid_beta, os.path.join(path_prefix, 'size_4096', 'beta'), dtype='float32')

                sub_grid_delta = np.reshape(sub_grid_delta, [1, *sub_grid_delta.shape, 1])
                sub_grid_beta = np.reshape(sub_grid_beta, [1, *sub_grid_beta.shape, 1])

                # During padding, sub_grids are read into the RAM
                pad_top, pad_bottom, pad_left, pad_right = get_padding_lengths(line_st, line_end, px_st, px_end, original_grid_shape, safe_zone_width)
                sub_grid_delta = np.pad(sub_grid_delta, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                                        mode='edge')
                sub_grid_beta = np.pad(sub_grid_beta, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                                       mode='edge')
                block_delta_batch[ind] = sub_grid_delta
                block_beta_batch[ind] = sub_grid_beta


            wavefield, dt = multislice_propagate_batch_numpy(block_delta_batch, block_beta_batch,
                                                             block_probe_real_batch, block_probe_imag_batch, energy_ev,
                                                             [psize_cm, psize_cm, slice_spacing_cm],
                                                             h=h,
                                                             obj_batch_shape=block_delta_batch.shape,
                                                             return_fft_time=True, starting_slice=0, t_init=0,
                                                             debug=False, debug_save_path=None,
                                                             rank=rank, verbose=False, repeating_slice=1)
            block_probe_real_batch[:, :, :] = wavefield.real
            block_probe_imag_batch[:, :, :] = wavefield.imag

            t0 = time.time()
            dt_prop += dt

            # if rank == 0: dxchange.write_tiff(abs(wavefield[0]), 'charcoal/size_4096/partial0.tiff', dtype='float32', overwrite=True)
        # if rank == 1: dxchange.write_tiff(abs(wavefield[0]), 'charcoal/size_4096/partial0.tiff', dtype='float32', overwrite=True)

        t_write_0 = time.time()
        if hdf5:
            f_out = h5py.File(os.path.join(path_prefix, 'size_{}'.format(this_size),
                                          'pfft_nslices_{}_output.h5'.format(n_slices)),
                              'w', driver='mpio', comm=comm)
            dset = f_out.create_dataset('wavefield', original_grid_shape[:-1], dtype='complex64')
            pos_ind_ls = range(rank, n_blocks, n_ranks)
            for ind, i_pos in enumerate(pos_ind_ls):
                line_st = i_pos // n_blocks_x * block_size
                line_end = min([line_st + block_size, original_grid_shape[0]])
                px_st = i_pos % n_blocks_x * block_size
                px_end = min([px_st + block_size, original_grid_shape[1]])
                dset[line_st:line_end, px_st:px_end] += wavefield[ind,
                                                                  safe_zone_width:safe_zone_width + (line_end - line_st),
                                                                  safe_zone_width:safe_zone_width + (px_end - px_st)]
            f_out.close()

        else:
            block_ls = comm.gather(wavefield, root=0)
            if rank == 0:
                full_wavefield = np.zeros([n_batch, *original_grid_shape[:-1]], dtype=np.complex64)
                for i_src_rank in range(len(block_ls)):
                    pos_ind_ls = range(i_src_rank, n_blocks, n_ranks)
                    for ind, i_pos in enumerate(pos_ind_ls):
                        line_st = i_pos // n_blocks_x * block_size
                        line_end = min([line_st + block_size, original_grid_shape[0]])
                        px_st = i_pos % n_blocks_x * block_size
                        px_end = min([px_st + block_size, original_grid_shape[1]])
                        full_wavefield[0, line_st:line_end, px_st:px_end] += block_ls[i_src_rank][ind,
                                                                             safe_zone_width:safe_zone_width + (
                                                                                         line_end - line_st),
                                                                             safe_zone_width:safe_zone_width + (
                                                                                         px_end - px_st)]


                    # dxchange.write_tiff(abs(full_wavefield), os.path.join(path_prefix, 'size_{}'.format(this_size),
                    #                                                       'pfft_nslices_{}_output.tiff'.format(
                    #                                                           n_slices)), dtype='float32',
                    #                     overwrite=True)
                np.save(os.path.join(path_prefix, 'size_{}'.format(this_size),
                                     'pfft_nslices_{}_output'.format(n_slices)), np.squeeze(full_wavefield))
                # if rank == 0:
                #     np.savetxt(os.path.join(path_prefix, 'size_{}'.format(this_size), 'dt_all_repeats.txt'), dt_ls)
            comm.Barrier()

        dt_writing = time.time() - t_write_0
        dt_tot = time.time() - t_tot_0

        if rank == 0:
            print('PFFT: For size {}, average dt = {} s.'.format(this_size, dt_tot))
            f.write('pfft,{},{},{},{},{},{},{}\n'.format(this_size, n_slices, safe_zone_width, dt_tot, dt_prop, dt_reading, dt_writing))
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
