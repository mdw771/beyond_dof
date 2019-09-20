from fullfield import reconstruct_fullfield
import autograd.numpy as np
from constants import *
import dxchange
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

warnings.filterwarnings("ignore")

params_adhesin = {'fname': 'data_adhesin_360_soft.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_epochs': 5,
                  'alpha_d': 1.e-9,
                  'alpha_b': 1.e-10,
                  'gamma': 0,
                  'learning_rate': 1e-7,
                  'center': 32,
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'minibatch_size': 10,
                  'theta_downsample': None,
                  'n_epochs_mask_release': 200,
                  'shrink_cycle': 9999,
                  'free_prop_cm': None,
                  'n_batch_per_update': 1,
                  'output_folder': 'test',
                  'cpu_only': True,
                  'save_path': 'adhesin',
                  'phantom_path': 'adhesin/phantom',
                  'multiscale_level': 1,
                  'n_epoch_final_pass': None,
                  'save_intermediate': True,
                  'full_intermediate': True,
                  'initial_guess': None,
                  'probe_type': 'plane',
                  'forward_algorithm': 'fresnel',
                  'kernel_size': 5,
                  'kwargs': {}}

params_cone = {'fname': 'data_cone_256_1nm_1um.h5',
               'theta_st': 0,
               'theta_end': 2 * np.pi,
               'n_epochs': 10,
               'alpha_d': 1.5e-8,
               'alpha_b': 1.5e-9,
               'gamma': 1e-11,
               'learning_rate': 1e-7,
               'center': 128,
               'energy_ev': 5000,
               'psize_cm': 1.e-7,
               'minibatch_size': 10,
               # 'minibatch_size': 1,
               'theta_downsample': None,
               'n_epochs_mask_release': 10,
               'shrink_cycle': None,
               'free_prop_cm': 1e-4,
               'n_batch_per_update': 1,
               'output_folder': 'n1_auto',
               'cpu_only': True,
               'save_path': 'cone_256_foam',
               'phantom_path': 'cone_256_foam/phantom',
               'multiscale_level': 1,
               'n_epoch_final_pass': 10,
               'save_intermediate': True,
               'object_type': 'normal',
               'full_intermediate': True,
               'initial_guess': [dxchange.read_tiff('cone_256_foam/init_delta.tiff'), dxchange.read_tiff('cone_256_foam/init_beta.tiff')],
               'probe_type': 'plane',
               'debug': True,
               'forward_algorithm': 'conv',
               'safe_zone_width': 'auto',  # 'auto'
               'kernel_size': 17,
               }

params = params_cone

reconstruct_fullfield(**params)
