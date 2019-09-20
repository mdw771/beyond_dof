import numpy as np
from scipy.ndimage import gaussian_filter
import dxchange
import glob
import os
from tqdm import trange, tqdm
from xommons import *


energy_kev = 3
max_size = 4096
resize_level = 4

src_path = '/data/datasets/charcoal/recon_crop_8_bgrm'
dest_path = '/data/datasets/charcoal/grids'
# obj = dxchange.read_tiff_stack(os.path.join(src_path, 'recon_00000.tiff'), range(len(glob.glob(os.path.join(src_path, 'recon_*')))), digit=5)
#
# obj_0 = np.zeros([max_size] * 3, dtype='float32')
# obj_0_start = (max_size - np.array(obj.shape)) // 2
# obj_0[obj_0_start[0]:obj_0_start[0] + obj.shape[0], obj_0_start[1]:obj_0_start[1] + obj.shape[1], obj_0_start[2]:obj_0_start[2] + obj.shape[2]] = obj
#
# base_delta = ri_delta('C', energy_kev, 0.5)
# base_beta = ri_beta('C', energy_kev, 0.5)
#
# dxchange.write_tiff((obj_0 / 44.) * base_delta, os.path.join(dest_path, 'size_{}/grid_delta'.format(max_size)), dtype='float32')
# dxchange.write_tiff((obj_0 / 44.) * base_beta, os.path.join(dest_path, 'size_{}/grid_beta'.format(max_size)), dtype='float32')

base_delta = ri_delta('C', energy_kev, 0.5)
base_beta = ri_beta('C', energy_kev, 0.5)

print('reading')
obj_delta = dxchange.read_tiff(os.path.join(dest_path, 'size_{}/grid_delta.tiff'.format(max_size)))
print('reading done')

for size_divider in range(2, resize_level + 1):

    this_size = max_size // (2 ** size_divider)
    print(this_size)

    padding = 5
    if size_divider == 1:
        n_batches = len(range(0, obj_delta.shape[0], 64))
        for ind, i in tqdm(enumerate(range(0, obj_delta.shape[0], 64)), total=n_batches):
            obj_chunk = obj_delta[max([0, i - padding]):min([i + 64 + padding, obj_delta.shape[0]])]
            if ind == 0:
                obj_chunk = np.pad(obj_chunk, [[padding, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
            elif ind == n_batches - 1:
                obj_chunk = np.pad(obj_chunk, [[0, padding], [0, 0], [0, 0]], mode='constant', constant_values=0)

            obj_filt = gaussian_filter(obj_chunk, 1)

            dxchange.write_tiff_stack(obj_filt[padding:padding + 64][::2 ** size_divider, ::2 ** size_divider, ::2 ** size_divider],
                                os.path.join(dest_path, 'size_{}/grid_delta'.format(this_size)), start=i // 2,
                                dtype='float32', overwrite=True)
            dxchange.write_tiff_stack(obj_filt[padding:padding + 64][::2 ** size_divider, ::2 ** size_divider, ::2 ** size_divider]
                                / base_delta * base_beta,
                                os.path.join(dest_path, 'size_{}/grid_beta'.format(this_size)), start=i // 2,
                                dtype='float32', overwrite=True)


    elif size_divider == 2:
        obj_delta = dxchange.read_tiff_stack(os.path.join(dest_path, 'size_{}'.format(max_size // (2 ** (size_divider - 1))), 'grid_delta_00000.tiff'), ind=range(max_size // (2 ** (size_divider - 1))))
        n_batches = len(range(0, obj_delta.shape[0], 64))
        print(obj_delta.shape)
        for ind, i in tqdm(enumerate(range(0, obj_delta.shape[0], 64)), total=n_batches):
            obj_chunk = obj_delta[max([0, i - padding]):min([i + 64 + padding, obj_delta.shape[0]])]
            if ind == 0:
                obj_chunk = np.pad(obj_chunk, [[padding, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
            elif ind == n_batches - 1:
                obj_chunk = np.pad(obj_chunk, [[0, padding], [0, 0], [0, 0]], mode='constant', constant_values=0)

            obj_filt = gaussian_filter(obj_chunk, 1)

            dxchange.write_tiff_stack(obj_filt[padding:padding + 64][::2, ::2, ::2],
                                os.path.join(dest_path, 'size_{}/grid_delta'.format(this_size)), start=i // 2,
                                dtype='float32', overwrite=True)
            dxchange.write_tiff_stack(obj_filt[padding:padding + 64][::2, ::2, ::2]
                                / base_delta * base_beta,
                                os.path.join(dest_path, 'size_{}/grid_beta'.format(this_size)), start=i // 2,
                                dtype='float32', overwrite=True)
        # merge into a single file
        a = dxchange.read_tiff_stack(os.path.join(dest_path, 'size_{}/grid_delta_00000.tiff'.format(this_size)), range(this_size))
        dxchange.write_tiff(a, os.path.join(dest_path, 'size_{}/grid_delta.tiff'.format(this_size)), dtype='float32', overwrite=True)
        a = dxchange.read_tiff_stack(os.path.join(dest_path, 'size_{}/grid_beta_00000.tiff'.format(this_size)), range(this_size))
        dxchange.write_tiff(a, os.path.join(dest_path, 'size_{}/grid_beta.tiff'.format(this_size)), dtype='float32', overwrite=True)

    else:
        obj_temp = dxchange.read_tiff(os.path.join(dest_path, 'size_{}'.format(max_size // (2 ** (size_divider - 1))), 'grid_delta.tiff'))
        obj_temp = gaussian_filter(obj_temp, 1)
        obj_temp = obj_temp[::2, ::2, ::2]
        dxchange.write_tiff(obj_temp, os.path.join(dest_path, 'size_{}/grid_delta.tiff'.format(this_size)), dtype='float32', overwrite=True)

        obj_temp = dxchange.read_tiff(os.path.join(dest_path, 'size_{}'.format(max_size // (2 ** (size_divider - 1))), 'grid_beta.tiff'))
        obj_temp = gaussian_filter(obj_temp, 1)
        obj_temp = obj_temp[::2, ::2, ::2]
        dxchange.write_tiff(obj_temp, os.path.join(dest_path, 'size_{}/grid_beta.tiff'.format(this_size)), dtype='float32', overwrite=True)



