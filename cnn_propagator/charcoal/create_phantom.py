import numpy as np
from mpi4py import MPI
import xommons
import tomosaic
from tomosaic import blend, arrange_image
import time
from glob import glob
import dxchange
import os
from tqdm import tqdm

def build_panorama(file_grid, shift_grid, tile, frame=0, method='max', method2=None, blend_options={}, blend_options2={},
                   blur=None, color_correction=False, margin=100, data_format='aps_32id', verbose=True):

    buff = np.zeros([1, 1])
    last_none = False
    if method2 is None:
        for (y, x), _ in np.ndenumerate(file_grid):
            print(y, x)
            buff = blend(buff, np.squeeze(tile), shift_grid[y, x, :], method=method, color_correction=color_correction, **blend_options)
            if last_none:
                buff[margin:, margin:-margin][np.isnan(buff[margin:, margin:-margin])] = 0
                last_none = False
    else:
        for y in range(file_grid.shape[0]):
            temp_grid = file_grid[y:y+1, :]
            temp_shift = np.copy(shift_grid[y:y+1, :, :])
            offset = np.min(temp_shift[:, :, 0])
            temp_shift[:, :, 0] = temp_shift[:, :, 0] - offset
            row_buff = np.zeros([1, 1])
            row_buff, _ = arrange_image(row_buff, np.squeeze(tile), temp_shift[0, 0, :], order=1)
            for x in range(1, temp_grid.shape[1]):
                print('Rank {} y {} x {}'.format(rank, y, x))
                row_buff = blend(row_buff, np.squeeze(tile), temp_shift[0, x, :], method=method, color_correction=color_correction, **blend_options)
                if last_none:
                    row_buff[margin:, margin:-margin][np.isnan(row_buff[margin:, margin:-margin])] = 0
                    last_none = False
            buff = blend(buff, row_buff, [offset, 0], method=method2, color_correction=False, **blend_options2)
    return buff


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()
name = MPI.Get_processor_name()

delta = xommons.ri_delta('Au', 5, 19.3)
beta = xommons.ri_beta('Au', 5, 19.3)

tile_size = 2048
########################################################################
tile_grid = [32] * 2
interval = 50
n_slices = 51
########################################################################
img_size = [tile_size * i for i in tile_grid]
roi_lt_corner = (2232, 3510)
blend_len = 200
empty_ratio = 0.15
empty_len = int(img_size[0] * empty_ratio)

src_folder = '/data/datasets/charcoal/recon_crop_8_med'
dest_folder = 'size_{}/phantom'.format(img_size[0])

src_img_list = glob(os.path.join(src_folder, 'recon_*.tiff'))
src_img_list.sort()
src_img_list = src_img_list[184:184 + interval * n_slices:interval]

file_grid = np.ones(tile_grid)
shift_grid = tomosaic.start_shift_grid(file_grid, tile_size, tile_size)

tile0 = dxchange.read_tiff(src_img_list[0], slc=((roi_lt_corner[0], roi_lt_corner[0] + blend_len * 2), (roi_lt_corner[1], roi_lt_corner[1] + blend_len * 2)))
dc = np.mean(tile0)

for i, src_img in tqdm(enumerate(src_img_list[rank+43:len(src_img_list):n_ranks]), total=len(src_img_list[rank:len(src_img_list):n_ranks])):
    ind = rank + i * n_ranks+43
    print(ind)
    tile = dxchange.read_tiff(src_img, slc=((roi_lt_corner[0], roi_lt_corner[0] + tile_size + blend_len * 2), (roi_lt_corner[1], roi_lt_corner[1] + tile_size + blend_len * 2)))
    slice = build_panorama(file_grid, shift_grid, tile, method='pyramid', blend_options={'depth': 3, 'blur': 0.4})
    slice = slice[blend_len:-blend_len, blend_len:-blend_len]
    slice[:empty_len, :] = 0
    slice[-empty_len:, :] = 0
    slice[:, :empty_len] = 0
    slice[:, -empty_len:] = 0
    slice /= dc
    slice[slice < 0] = 0
    dxchange.write_tiff(slice, os.path.join(dest_folder, 'img_{:05d}'.format(ind)), overwrite=True, dtype='float32')

