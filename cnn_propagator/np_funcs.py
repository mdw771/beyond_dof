import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from util import *



def gen_mesh(max, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, shape):

    k = 2 * np.pi / lmbda_nm
    x, y = gen_mesh((np.array(shape) - 1.) / 2 * np.array(voxel_nm[:2]), shape)
    kernel = np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbda_nm * dist_nm))
    return kernel


def get_kernel_tf(dist_nm, lmbda_nm, voxel_nm, grid_shape):

    k = 2 * np.pi / lmbda_nm
    u_max = 1. / (2. * voxel_nm[0])
    v_max = 1. / (2. * voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], grid_shape[0:2])
    # H = np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm**2 * (u**2 + v**2)))
    H = np.exp(1j * k * dist_nm) * np.exp(-1j * np.pi * lmbda_nm * dist_nm * (u ** 2 + v ** 2))

    return H


def multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, energy_ev, psize_cm, kernel_size=256, free_prop_cm=None):

    n_batch, shape_y, shape_x, n_slice = grid_delta.shape
    lmbda_nm = 1240. / energy_ev
    voxel_nm = np.array(psize_cm) * 1.e7
    delta_nm = voxel_nm[-1]
    k = 2. * np.pi * delta_nm / lmbda_nm

    kernel_complex = get_kernel_tf(delta_nm, lmbda_nm, voxel_nm, [kernel_size] * 2)

    kernel_complex = np.fft.ifft2(kernel_complex)

    probe_complex = probe_real + 1j * probe_imag
    probe_complex = np.tile(probe_complex[np.newaxis, :, :], [n_batch, 1, 1])

    for i_slice in range(n_slice):
        this_delta_batch = grid_delta[:, :, :, i_slice]
        this_beta_batch = grid_beta[:, :, :, i_slice]
        c = np.exp(1j * k * this_delta_batch - k * this_beta_batch)
        probe_complex = probe_complex * c

        # probe_complex = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(probe_complex)) * kernel_complex))
        probe_complex = scipy
        # print(probe_complex)

    return probe_complex

    # for i_slice in range(n_slice):
    #     this_delta_batch = grid_delta[:, :, :, i_slice]
    #     this_beta_batch = grid_beta[:, :, :, i_slice]
    #     c = np.exp(1j * k * this_delta_batch - k * this_beta_batch)
    #     this_probe_complex = (np.cast(probe_real, np.complex128) + 1j * np.cast(probe_imag, np.complex128)) * c
    #     probe_real, probe_imag = (np.real(this_probe_complex), np.imag(this_probe_complex))
    #
    #     wavefield_array = np.expand_dims(np.concat([probe_real, probe_imag], axis=0), -1)
    #     # print(wavefield_array)
    #     this_probe_complex = np.nn.conv2d(wavefield_array, kernel, (1, 1, 1, 1), 'SAME')
    #     ac = this_probe_complex[0:n_batch, :, :, 0]
    #     ad = this_probe_complex[0:n_batch, :, :, 1]
    #     bc = this_probe_complex[n_batch:, :, :, 0]
    #     bd = this_probe_complex[n_batch, :, :, 1]
    #     probe_real = ac - bd
    #     probe_imag = ad + bc
    #
    #     this_wavefield = np.exp(1j * np.cast(np.math.atan(probe_imag / probe_real), dtype=np.complex128))
    #     probe_real = np.real(this_wavefield)
    #     probe_imag = np.imag(this_wavefield)
    #
    #     print(this_wavefield)
    #
    # return np.cast(probe_real, np.complex128) + 1j *  np.cast(probe_imag, np.complex128)


if __name__ == '__main__':

    grid_delta = np.load('cone_256_foam/phantom/grid_delta.npy')
    grid_beta = np.load('cone_256_foam/phantom/grid_beta.npy')
    grid_delta = np.reshape(grid_delta, [1, *grid_delta.shape])
    grid_beta = np.reshape(grid_beta, [1, *grid_beta.shape])

    probe_real = np.ones([*grid_delta.shape[1:3]])
    probe_imag = np.zeros([*grid_delta.shape[1:3]])

    wavefield = multislice_propagate_cnn(grid_delta, grid_beta, probe_real, probe_imag, 5000, [1e-7, 1e-7, 1e-7])
    print(wavefield.shape)
    plt.imshow(abs(wavefield)[0])
    plt.show()

