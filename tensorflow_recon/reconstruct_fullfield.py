from fullfield import reconstruct_fullfield
import numpy as np
from constants import *

# init_delta_adhesin = np.zeros([64, 64, 64])
# init_delta_adhesin[...] = 8e-7
# init_beta_adhesin = np.zeros([64, 64, 64])
# init_beta_adhesin[...] = 8e-8
# init_delta_adhesin = np.load('adhesin/phantom/grid_delta.npy')
# init_beta_adhesin = np.load('adhesin/phantom/grid_beta.npy')
init_delta = np.load('2d_512/phantom/grid_delta.npy')
init_beta = np.load('2d_512/phantom/grid_beta.npy')
init = [init_delta, init_beta]


params_adhesin = {'fname': 'data_adhesin_360_soft.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_epochs': 1,
                  'alpha_d': 1.e-9,
                  'alpha_b': 1.e-10,
                  'gamma': 0,
                  'learning_rate': 1e-7,
                  'center': 32,
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'batch_size': 10,
                  'theta_downsample': None,
                  'n_epochs_mask_release': 200,
                  'shrink_cycle': 10,
                  'free_prop_cm': None,
                  'n_batch_per_update': 1,
                  'output_folder': 'test',
                  'cpu_only': True,
                  'save_folder': 'adhesin',
                  'phantom_path': 'adhesin/phantom',
                  'multiscale_level': 1,
                  'n_epoch_final_pass': None,
                  'save_intermediate': True,
                  'full_intermediate': True,
                  'initial_guess': None,
                  'probe_type': 'plane',
                  'forward_algorithm': 'fresnel',
                  'kwargs': {}}

params_cone = {'fname': 'data_cone_256_1nm_1um.h5',
               'theta_st': 0,
               'theta_end': 2 * np.pi,
               'n_epochs': 10,
               'alpha_d': 1.5e-7,
               'alpha_b': 1.5e-8,
               'gamma': 5e-8,
               'learning_rate': 1e-7,
               'center': 128,
               'energy_ev': 5000,
               'psize_cm': 1.e-7,
               'batch_size': 10,
               'theta_downsample': None,
               'n_epochs_mask_release': 10,
               'shrink_cycle': 1,
               'free_prop_cm': 1e-4,
               'n_batch_per_update': 1,
               'output_folder': 'test',
               'cpu_only': True,
               'save_folder': 'cone_256_filled/new',
               'phantom_path': 'cone_256_filled/phantom',
               'multiscale_level': 3,
               'n_epoch_final_pass': 6,
               'save_intermediate': True,
               'full_intermediate': True,
               'initial_guess': None,
               'probe_type': 'plane',
               'forward_algorithm': 'fresnel',
               'kwargs': {}}

params_cone_far = {'fname': 'data_cone_1nm_1um_far.h5',
               'theta_st': 0,
               'theta_end': 2 * np.pi,
               'n_epochs': 10,
               'alpha_d': 0,
               'alpha_b': 0,
               'gamma': 0,
               'learning_rate': 1e-7,
               'center': 128,
               'energy_ev': 5000,
               'psize_cm': 1.e-7,
               'batch_size': 10,
               'theta_downsample': None,
               'n_epochs_mask_release': 10,
               'shrink_cycle': 1,
               'free_prop_cm': 'inf',
               'n_batch_per_update': 1,
               'output_folder': 'far',
               'cpu_only': True,
               'save_folder': 'cone_256_filled/new',
               'phantom_path': 'cone_256_filled/phantom',
               'multiscale_level': 3,
               'n_epoch_final_pass': 6,
               'save_intermediate': True,
               'full_intermediate': True,
               'initial_guess': None,
               'probe_type': 'plane',
               'forward_algorithm': 'fresnel',
               'kwargs': {}}

params_2d = {'fname': 'data_cone_2d_1nm_1um_far.h5',
               'theta_st': 0,
               'theta_end': 0,
               'n_epochs': 1000,
               'alpha_d': 0,
               'alpha_b': 0,
               'gamma': 0,
               'learning_rate': 1e-6,
               'center': 128,
               'energy_ev': 5000,
               'psize_cm': 1.e-7,
               'batch_size': 1,
               'theta_downsample': None,
               'n_epochs_mask_release': 1000,
               'shrink_cycle': 9999,
               'free_prop_cm': 'inf',
               'n_batch_per_update': 1,
               'output_folder': 'fullfield/0_0_1e-6_far',
               'cpu_only': True,
               'save_folder': '2d_1024',
               'phantom_path': '2d_1024',
               'multiscale_level': 1,
               'n_epoch_final_pass': None,
               'save_intermediate': True,
               'full_intermediate': True,
               'initial_guess': None,
               'probe_type': 'gaussian',
               'forward_algorithm': 'fresnel',
               'kwargs': {'probe_mag_sigma': 100,
                               'probe_phase_sigma': 100,
                               'probe_phase_max': 0.5},
             }
params_cone_noisy = {'fname': 'data_cone_256_1nm_1um_180.h5',
                     'theta_st': 0,
                     'theta_end': 2 * np.pi,
                     'n_epochs': 10,
                     'alpha_d': 1.5e-7,
                     'alpha_b': 1.5e-8,
                     'gamma': 5e-8,
                     'learning_rate': 1e-7,
                     'center': 128,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 10,
                     'theta_downsample': None,
                     'n_epochs_mask_release': 10,
                     'shrink_cycle': 1,
                     'free_prop_cm': 1e-4,
                     'n_batch_per_update': 1,
                     'output_folder': '180_ref',
                     'cpu_only': True,
                     'save_folder': 'cone_256_filled/new',
                     'phantom_path': 'cone_256_filled/phantom',
                     'multiscale_level': 3,
                     'n_epoch_final_pass': 6,
                     'save_intermediate': True,
                     'full_intermediate': True,
                     'initial_guess': None,
                     'probe_type': 'plane',
                     'forward_algorithm': 'fresnel',
                     'kwargs': {}}

params_cone_pp = {'fname': 'data_cone_256_1nm_1um.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_epochs': 7,
                  'alpha_d': 1.5e-7,
                  'alpha_b': 1.5e-8,
                  'gamma': 1e-7,
                  'learning_rate': 1e-7,
                  'center': 128,
                  'energy_ev': 5000,
                  'psize_cm': 1.e-7,
                  'batch_size': 5,
                  'theta_downsample': None,
                  'n_epochs_mask_release': None,
                  'shrink_cycle': 1,
                  'free_prop_cm': 1e-4,
                  'n_batch_per_update': 1,
                  'output_folder': None,
                  'cpu_only': True,
                  'save_folder': 'cone_256_filled_pp',
                  'phantom_path': 'cone_256_filled_pp/phantom',
                  'multiscale_level': 3,
                  'n_epoch_final_pass': 6,
                  'save_intermediate': True,
                  'full_intermediate': True,
                  'initial_guess': None,
                  'probe_type': 'point',
                  'forward_algorithm': 'fresnel',
                  'kwargs': {'dist_to_source_cm': 1e-4,
                             'det_psize_cm': 3e-7,
                             'theta_max': PI / 15,
                             'phi_max': PI / 15}}

params = params_2d
# init_delta = np.load('cone_256_filled_pp/phantom/grid_delta.npy')
# init_beta = np.load('cone_256_filled_pp/phantom/grid_beta.npy')
# init = [init_delta, init_beta]

reconstruct_fullfield(fname=params['fname'],
                      theta_st=0,
                      theta_end=params['theta_end'],
                      n_epochs=params['n_epochs'],
                      n_epochs_mask_release=params['n_epochs_mask_release'],
                      shrink_cycle=params['shrink_cycle'],
                      crit_conv_rate=0.03,
                      max_nepochs=200,
                      alpha_d=params['alpha_d'],
                      alpha_b=params['alpha_b'],
                      gamma=params['gamma'],
                      free_prop_cm=params['free_prop_cm'],
                      learning_rate=params['learning_rate'],
                      output_folder=params['output_folder'],
                      minibatch_size=params['batch_size'],
                      theta_downsample=params['theta_downsample'],
                      save_intermediate=params['save_intermediate'],
                      full_intermediate=params['full_intermediate'],
                      energy_ev=params['energy_ev'],
                      psize_cm=params['psize_cm'],
                      cpu_only=params['cpu_only'],
                      save_path=params['save_folder'],
                      phantom_path=params['phantom_path'],
                      multiscale_level=params['multiscale_level'],
                      n_epoch_final_pass=params['n_epoch_final_pass'],
                      initial_guess=params['initial_guess'],
                      n_batch_per_update=params['n_batch_per_update'],
                      dynamic_rate=True,
                      probe_type=params['probe_type'],
                      probe_initial=None,
                      probe_learning_rate=1e-3,
                      pupil_function=None,
                      forward_algorithm=params['forward_algorithm'],
                      **params['kwargs'])
